import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_19_l453_45354

-- Define the game rules
def valid_move (remaining : List Nat) (n : Nat) : Bool :=
  remaining.elem n && remaining.any (fun m => m ≠ n && n % m = 0)

def remove_divisors (remaining : List Nat) (n : Nat) : List Nat :=
  remaining.filter (fun m => m = n || n % m ≠ 0)

-- Define Carolyn's moves
def carolyn_moves : List Nat :=
  [3, 6, 10]

-- Theorem statement
theorem carolyn_sum_is_19 (n : Nat) (h : n = 10) :
  let initial_list := List.range n
  let first_move := 3
  let moves := carolyn_moves
  moves.head? = some first_move →
  (moves.foldl
    (fun remaining move =>
      if valid_move remaining move
      then remove_divisors remaining move
      else remaining)
    initial_list).all (fun m => ¬valid_move (remove_divisors initial_list first_move) m) →
  moves.sum = 19 := by
  sorry

#eval carolyn_moves.sum -- To check if the sum is indeed 19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_19_l453_45354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_no_zeros_l453_45385

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * log x

-- State the theorem
theorem min_a_for_no_zeros (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (1/2), f a x ≠ 0) → a ≥ 2 - 4 * log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_no_zeros_l453_45385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_theorem_l453_45353

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The focal distance of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  2 * Real.sqrt (h.a^2 + h.b^2)

/-- The length of the real axis of a hyperbola -/
def real_axis_length (h : Hyperbola) : ℝ := 2 * h.a

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

theorem hyperbola_asymptote_theorem (h : Hyperbola) :
  real_axis_length h = (1/2) * focal_distance h →
  ∀ x y, asymptote_equation h x y ↔ 
    (∃ t, t ≠ 0 ∧ hyperbola_equation h (t*x) (t*y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_theorem_l453_45353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_school_time_l453_45336

/-- Represents the walking characteristics of a person -/
structure Walker where
  steps_per_minute : ℝ
  step_length : ℝ

/-- Calculates the walking speed in cm/min -/
def walking_speed (w : Walker) : ℝ := w.steps_per_minute * w.step_length

/-- Calculates the distance traveled in a given time -/
def distance_traveled (w : Walker) (time : ℝ) : ℝ := walking_speed w * time

theorem mike_school_time (dave : Walker) (mike : Walker) (dave_time : ℝ) :
  dave.steps_per_minute = 85 →
  dave.step_length = 70 →
  dave_time = 18 →
  mike.steps_per_minute = 95 →
  mike.step_length = 65 →
  ∃ (mike_time : ℝ), 
    distance_traveled dave dave_time = distance_traveled mike mike_time ∧
    (mike_time ≥ 17.33 ∧ mike_time ≤ 17.35) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_school_time_l453_45336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_decreasing_numbers_l453_45337

theorem six_digit_decreasing_numbers : ∃ n : ℕ, 
  (n = Nat.choose 10 6) ∧ 
  (n = (Finset.filter (fun x : Fin 1000000 => 
    ∀ i j : Fin 6, i < j → 
      ((x : ℕ) / 10^(5 - i.val) % 10) > ((x : ℕ) / 10^(5 - j.val) % 10))
    (Finset.univ)).card) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_decreasing_numbers_l453_45337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_radial_axis_l453_45367

/-- A circle in polar form -/
structure PolarCircle where
  D : ℝ
  E : ℝ

/-- Predicate for a circle being tangent to the radial axis -/
def isTangentToRadialAxis (c : PolarCircle) : Prop :=
  c.D = 0 ∧ c.E ≠ 0

/-- Theorem stating the necessary and sufficient condition for a circle to be tangent to the radial axis -/
theorem circle_tangent_to_radial_axis (c : PolarCircle) :
  isTangentToRadialAxis c ↔ ∀ θ : ℝ, Real.cos θ * c.D + Real.sin θ * c.E ≥ 0 ∧ ∃ θ₀ : ℝ, Real.cos θ₀ * c.D + Real.sin θ₀ * c.E = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_radial_axis_l453_45367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l453_45391

theorem f_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(abs x)
  f (a + 1) > f 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l453_45391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_schemeA_more_cost_effective_savings_amount_l453_45392

/-- Regular ticket price in yuan -/
def a : ℝ := 1

/-- Cost calculation for Scheme A -/
def schemeACost (x : ℕ) : ℝ := (1 - 0.25) * ((x : ℝ) - 2) * a + 2 * a

/-- Cost calculation for Scheme B -/
def schemeBCost (x : ℕ) : ℝ := (1 - 0.20) * (x : ℝ) * a

/-- Theorem stating that Scheme A is more cost-effective for a group of 18 people -/
theorem schemeA_more_cost_effective :
  schemeACost 18 < schemeBCost 18 ∧
  schemeBCost 18 - schemeACost 18 = 0.4 * a := by
  sorry

/-- Theorem proving the savings amount -/
theorem savings_amount :
  schemeBCost 18 - schemeACost 18 = 0.4 * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_schemeA_more_cost_effective_savings_amount_l453_45392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_l453_45384

noncomputable def p1 : ℝ × ℝ := (0, 30)
noncomputable def p2 : ℝ × ℝ := (150, 0)
noncomputable def p3 : ℝ × ℝ := (0, 0)
noncomputable def p4 : ℝ × ℝ := (150, 90)

noncomputable def line1 (x : ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1) * (x - p1.1) + p1.2
noncomputable def line2 (x : ℝ) : ℝ := (p4.2 - p3.2) / (p4.1 - p3.1) * (x - p3.1) + p3.2

theorem intersection_height :
  ∃ x : ℝ, line1 x = line2 x ∧ line1 x = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_l453_45384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l453_45394

/-- The range of values for the real number a given the specified conditions -/
theorem range_of_a : 
  (∀ x₀ : ℝ, x₀^2 - a*x₀ + 1 ≠ 0) →
  (∃ x : ℝ, x ≥ 0 ∧ ((x^2 - 2*a*x + a^2 + 1 < 1) ∨ 
    (∃ y : ℝ, y ≥ 1 ∧ x^2 - 2*a*x + a^2 + 1 > y))) →
  Set.Ioo (-2 : ℝ) 0 = {a : ℝ | true} := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l453_45394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_left_to_read_l453_45320

/-- Given a 400-page book and having read 20% of it, prove that the number of pages left to read is 320. -/
theorem pages_left_to_read (total_pages : ℕ) (percentage_read : ℚ) (h1 : total_pages = 400) (h2 : percentage_read = 1/5) :
  total_pages - (percentage_read * ↑total_pages).floor = 320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_left_to_read_l453_45320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_prism_volume_l453_45325

/-- The volume of a regular quadrilateral prism -/
noncomputable def prismVolume (a : ℝ) (α : ℝ) : ℝ :=
  (a^3 * Real.sqrt (2 * Real.cos α)) / (2 * Real.sin (α / 2))

/-- Theorem: The volume of a regular quadrilateral prism with base side length a
    and angle α between intersecting diagonals of two adjacent lateral faces -/
theorem regular_quadrilateral_prism_volume
  (a : ℝ) (α : ℝ) (h1 : 0 < a) (h2 : 0 < α ∧ α < π) :
  let V := prismVolume a α
  ∃ (h : ℝ), V = a^2 * h ∧ h = (a * Real.sqrt (2 * Real.cos α)) / (2 * Real.sin (α / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_prism_volume_l453_45325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l453_45359

open Real

noncomputable def f (x : ℝ) : ℝ := tan x ^ 2 - 4 * tan x - 8 * (1 / tan x) + 4 * (1 / tan x) ^ 2 + 5

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), π / 2 < x ∧ x < π ∧
  (∀ (y : ℝ), π / 2 < y ∧ y < π → f y ≥ f x) ∧
  f x = 9 - 8 * sqrt 2 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l453_45359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_increase_l453_45351

-- Define the original trapezoid dimensions
noncomputable def original_base1 : ℝ := 10
noncomputable def original_base2 : ℝ := 20
noncomputable def original_height : ℝ := 5

-- Define the increase percentages
noncomputable def base1_increase : ℝ := 0.15
noncomputable def base2_increase : ℝ := 0.25
noncomputable def height_increase : ℝ := 0.10

-- Define the function to calculate the area of a trapezoid
noncomputable def trapezoid_area (b1 b2 h : ℝ) : ℝ := (1/2) * (b1 + b2) * h

-- Define the new dimensions after increase
noncomputable def new_base1 : ℝ := original_base1 * (1 + base1_increase)
noncomputable def new_base2 : ℝ := original_base2 * (1 + base2_increase)
noncomputable def new_height : ℝ := original_height * (1 + height_increase)

-- Theorem statement
theorem trapezoid_area_increase :
  trapezoid_area new_base1 new_base2 new_height - trapezoid_area original_base1 original_base2 original_height = 25.375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_increase_l453_45351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_of_cubic_l453_45388

-- Define the polynomial type
def cubic_polynomial (a b c : ℚ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

-- State the theorem
theorem integer_root_of_cubic (a b c : ℚ) :
  (∃ x : ℝ, cubic_polynomial a b c x = 0 ∧ x = 3 - Real.sqrt 5) →
  (∃ x y z : ℝ, cubic_polynomial a b c x = 0 ∧ 
                cubic_polynomial a b c y = 0 ∧ 
                cubic_polynomial a b c z = 0 ∧ 
                x + y + z = -2) →
  cubic_polynomial a b c (-8) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_of_cubic_l453_45388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_decaf_percentage_l453_45316

theorem coffee_decaf_percentage 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ) 
  (new_batch : ℝ) 
  (final_decaf_percent : ℝ) 
  (x : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 25)
  (h3 : new_batch = 100)
  (h4 : final_decaf_percent = 32)
  (h5 : initial_stock * (initial_decaf_percent / 100) + new_batch * (x / 100) = 
        (initial_stock + new_batch) * (final_decaf_percent / 100)) :
  x = 60 := by
  sorry

#check coffee_decaf_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_decaf_percentage_l453_45316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l453_45339

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

theorem symmetry_of_f :
  (∀ x, f (2 * Real.pi / 3 - x) = f x) ∧
  (∀ x, f (Real.pi / 6 + x) = f (Real.pi / 6 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l453_45339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_for_negative_tangent_l453_45318

/-- 
Given:
- α is the inclination angle of a line
- tan(α) = -2

Prove:
α = π - arctan(2)
-/
theorem inclination_angle_for_negative_tangent (α : Real) :
  (∃ k b : Real, α = Real.arctan k ∧ k * Real.cos α = Real.sin α) →  -- α is inclination angle
  Real.tan α = -2 →                                                  -- tan(α) = -2
  α = Real.pi - Real.arctan 2 :=                                     -- conclusion
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_for_negative_tangent_l453_45318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_b_same_order_option_a_different_order_option_c_different_order_l453_45301

-- Define the order of operations for arithmetic expressions
inductive ArithExpr
  | num : ℤ → ArithExpr
  | add : ArithExpr → ArithExpr → ArithExpr
  | sub : ArithExpr → ArithExpr → ArithExpr
  | mul : ArithExpr → ArithExpr → ArithExpr
  | div : ArithExpr → ArithExpr → ArithExpr

-- Define a function to evaluate arithmetic expressions
def evaluate : ArithExpr → ℤ
  | ArithExpr.num n => n
  | ArithExpr.add a b => evaluate a + evaluate b
  | ArithExpr.sub a b => evaluate a - evaluate b
  | ArithExpr.mul a b => evaluate a * evaluate b
  | ArithExpr.div a b => evaluate a / evaluate b

-- Define a function to check if two expressions have the same order of operations
def sameOrderOfOperations (e1 e2 : ArithExpr) : Prop :=
  ∀ (n1 n2 n3 : ℤ), 
    evaluate (ArithExpr.sub (ArithExpr.sub (ArithExpr.num n1) (ArithExpr.num n2)) (ArithExpr.num n3)) = 
    evaluate (ArithExpr.mul (ArithExpr.div (ArithExpr.num n1) (ArithExpr.num n2)) (ArithExpr.num n3))

-- Define the expressions for option B
def optionB1 : ArithExpr := ArithExpr.sub (ArithExpr.sub (ArithExpr.num 36) (ArithExpr.num 9)) (ArithExpr.num 5)
def optionB2 : ArithExpr := ArithExpr.mul (ArithExpr.div (ArithExpr.num 36) (ArithExpr.num 6)) (ArithExpr.num 5)

-- Theorem statement
theorem option_b_same_order : sameOrderOfOperations optionB1 optionB2 := by
  sorry

-- Define the expressions for options A and C
def optionA1 : ArithExpr := ArithExpr.div (ArithExpr.mul (ArithExpr.num 2) (ArithExpr.num 9)) (ArithExpr.num 3)
def optionA2 : ArithExpr := ArithExpr.add (ArithExpr.num 2) (ArithExpr.mul (ArithExpr.num 9) (ArithExpr.num 3))

def optionC1 : ArithExpr := ArithExpr.mul (ArithExpr.div (ArithExpr.num 56) (ArithExpr.num 7)) (ArithExpr.num 5)
def optionC2 : ArithExpr := ArithExpr.add (ArithExpr.num 56) (ArithExpr.mul (ArithExpr.num 7) (ArithExpr.num 5))

-- Theorem statements for options A and C
theorem option_a_different_order : ¬ sameOrderOfOperations optionA1 optionA2 := by
  sorry

theorem option_c_different_order : ¬ sameOrderOfOperations optionC1 optionC2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_b_same_order_option_a_different_order_option_c_different_order_l453_45301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_three_roots_l453_45397

theorem equation_three_roots (m : ℂ) : 
  (∃! (roots : Finset ℂ), 
    (roots.card = 3) ∧ 
    (∀ x ∈ roots, (2 * x) / (x + 1) + (4 * x) / (x + 3) = m * x)) ↔ 
  (m = 1 + 2 * Complex.I * Real.sqrt 2 ∨ m = 1 - 2 * Complex.I * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_three_roots_l453_45397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_monotonically_increasing_l453_45338

def a (n : ℕ+) (lambda : ℝ) := n^2 + lambda * n

theorem sequence_monotonically_increasing (lambda : ℝ) :
  (∀ n : ℕ+, a (n + 1) lambda > a n lambda) ↔ lambda > -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_monotonically_increasing_l453_45338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l453_45375

noncomputable def f (x : ℝ) : ℝ := (2 * x^2 - 1) / (x^2 + 2)

theorem f_range : Set.range f = Set.Icc (-1/2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l453_45375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_final_distance_l453_45379

/-- Represents the distance traveled by the ant in each step -/
noncomputable def antStep (n : ℕ) : ℝ := (1/2)^n

/-- Represents the direction of the ant's movement at each step -/
inductive Direction
| East
| Northeast

/-- Determines the direction of the ant's movement at step n -/
def stepDirection (n : ℕ) : Direction :=
  if n % 2 = 0 then Direction.East else Direction.Northeast

/-- Calculates the horizontal distance traveled in a single step -/
noncomputable def horizontalDistance (n : ℕ) : ℝ :=
  match stepDirection n with
  | Direction.East => antStep n
  | Direction.Northeast => antStep n / Real.sqrt 2

/-- Calculates the vertical distance traveled in a single step -/
noncomputable def verticalDistance (n : ℕ) : ℝ :=
  match stepDirection n with
  | Direction.East => 0
  | Direction.Northeast => antStep n / Real.sqrt 2

/-- The theorem stating the final distance of the ant from its starting point -/
theorem ant_final_distance :
  (∑' n, horizontalDistance n)^2 + (∑' n, verticalDistance n)^2 =
  (2/3 * Real.sqrt (2 * Real.sqrt 2 + 5))^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_final_distance_l453_45379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_energetic_triplets_no_specific_energetic_triplets_l453_45315

def is_n_energetic (a b c n : ℕ) : Prop :=
  (0 < a) ∧ (a ≤ b) ∧ (b ≤ c) ∧ (Nat.gcd a (Nat.gcd b c) = 1) ∧
  (a^n + b^n + c^n) % (a + b + c) = 0

theorem n_energetic_triplets :
  ∀ a b c : ℕ, (∀ n : ℕ, n > 0 → is_n_energetic a b c n) →
  ((a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 2 ∧ c = 3)) :=
sorry

-- Additional theorem for the second part of the problem
theorem no_specific_energetic_triplets :
  ¬ ∃ a b c : ℕ, 
    (is_n_energetic a b c 2004 ∧ is_n_energetic a b c 2005 ∧ ¬is_n_energetic a b c 2007) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_energetic_triplets_no_specific_energetic_triplets_l453_45315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l453_45302

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else -x * (x + 2)

-- State the theorem
theorem f_has_three_zeros :
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
    (∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l453_45302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_implies_a_value_l453_45303

theorem constant_term_implies_a_value (a : ℝ) : 
  (∃ (x : ℝ), (∀ (n : ℕ), n ≠ 2 → 
    (Finset.sum (Finset.range 7) (λ k ↦ (-1)^k * Nat.choose 6 k * a^(k/2) * x^((6-3*k)/2))) = 0) ∧
    (Finset.sum (Finset.range 7) (λ k ↦ (-1)^k * Nat.choose 6 k * a^(k/2) * x^((6-3*k)/2))) = 60) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_implies_a_value_l453_45303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_solution_set_l453_45393

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x - 1 else -x - 1

-- State the theorem
theorem f_negative_solution_set :
  (∀ x : ℝ, f (-x) = f x) →  -- f is even
  (∀ x : ℝ, x ≥ 0 → f x = x - 1) →  -- f(x) = x - 1 for x ≥ 0
  {x : ℝ | f x < 0} = {x : ℝ | -1 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_solution_set_l453_45393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_sqrt_16_l453_45328

theorem sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_sqrt_16_l453_45328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_well_defined_and_decreasing_l453_45396

noncomputable def u : ℕ → ℝ
  | 0 => 3
  | n + 1 => Real.sqrt (2 + u n)

theorem u_well_defined_and_decreasing :
  (∀ n, u n ≥ 0) ∧ (∀ n, u (n + 1) < u n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_well_defined_and_decreasing_l453_45396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_reciprocal_sum_l453_45324

theorem min_value_reciprocal_sum (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 + Real.log (2^x * 4^y) = Real.log 2) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 
    Real.log 2 + Real.log (2^a * 4^b) = Real.log 2 → 
    1/x + 1/y ≤ 1/a + 1/b) ∧ 
  1/x + 1/y = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_reciprocal_sum_l453_45324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l453_45390

noncomputable def P : ℕ → ℝ × ℝ
| 0 => (3, 1)
| n + 1 => let (x, y) := P n; (-(3*x - y)/2, -(x + y)/2)

noncomputable def quadrilateral_area (p₁ p₂ p₃ p₄ : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  let (x₄, y₄) := p₄
  (1/2) * abs ((x₁*y₂ + x₂*y₃ + x₃*y₄ + x₄*y₁) - (y₁*x₂ + y₂*x₃ + y₃*x₄ + y₄*x₁))

theorem area_of_quadrilateral :
  quadrilateral_area (P 96) (P 97) (P 98) (P 99) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l453_45390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_problem_statement_l453_45327

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
noncomputable def y_intercept (a b c : ℝ) : ℝ := -c / b

/-- The line equation is in the form ax + by + c = 0 -/
theorem y_intercept_of_line (a b c : ℝ) (hb : b ≠ 0) :
  y_intercept a b c = -c / b :=
by sorry

theorem problem_statement :
  y_intercept 1 2 2 = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_problem_statement_l453_45327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_questionnaire_C_count_l453_45369

/-- Systematic sampling parameters -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  first_number : ℕ
  interval : ℕ

/-- Define the systematic sample for the given problem -/
def questionnaire_sample : SystematicSample :=
  { population := 1000
  , sample_size := 50
  , first_number := 8
  , interval := 20 }

/-- The number of people in the sample with numbers in a given range -/
def count_in_range (s : SystematicSample) (lower : ℕ) (upper : ℕ) : ℕ :=
  List.length (List.filter (fun n => 
    let sample_number := s.first_number + (n - 1) * s.interval
    lower ≤ sample_number ∧ sample_number ≤ upper
  ) (List.range s.sample_size))

/-- Theorem: The number of people completing questionnaire C is 12 -/
theorem questionnaire_C_count : 
  count_in_range questionnaire_sample 751 1000 = 12 := by
  sorry

#eval count_in_range questionnaire_sample 751 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_questionnaire_C_count_l453_45369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l453_45372

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B : Set ℝ := {x | x^2 - 4 < 0}

-- Define the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | 1/2 ≤ x ∧ x < 2}) ∧
  (A ∪ B = {x : ℝ | -2 < x ∧ x ≤ 3}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | -2 < x ∧ x < 1/2}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l453_45372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_sequence_l453_45312

def is_valid_sequence (f : ℕ → ℕ) : Prop :=
  (0 ∈ Set.range f) ∧
  (∀ k : ℕ, k > 0 → Set.Infinite {n : ℕ | f n = k}) ∧
  (∀ n : ℕ, f (f (n + 163)) = f (f n) + f (f 361)) ∧
  (f 1 = 0) ∧ (f 361 = 1)

theorem exists_valid_sequence : ∃ f : ℕ → ℕ, is_valid_sequence f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_sequence_l453_45312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_value_for_given_x_y_l453_45387

theorem n_value_for_given_x_y : ∀ (x y n : ℤ), 
  x = 3 → y = -1 → n = x - (y : ℚ)^(x+y) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_value_for_given_x_y_l453_45387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_30_l453_45355

theorem sum_remainder_mod_30 (a b c : ℕ) 
  (ha : a % 30 = 11)
  (hb : b % 30 = 7)
  (hc : c % 30 = 18) :
  (a + b + c) % 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_30_l453_45355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_sum_l453_45371

/-- Given a line segment with one endpoint at (5, -3) and midpoint at (3, -7),
    the sum of the coordinates of the other endpoint is -10. -/
theorem endpoint_coordinate_sum :
  ∀ (endpoint1 midpoint endpoint2 : ℝ × ℝ),
    endpoint1 = (5, -3) →
    midpoint = (3, -7) →
    midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 →
    midpoint.2 = (endpoint1.2 + endpoint2.2) / 2 →
    endpoint2.1 + endpoint2.2 = -10 := by
  -- The proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_sum_l453_45371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l453_45386

noncomputable def f (x y : ℝ) : ℝ := (2 * x^2 + y^2) / (x * y)

theorem max_value_of_f :
  ∀ x y : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 3 → f x y ≤ 2 * Real.sqrt 2 :=
by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l453_45386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l453_45376

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (a b c : V)

-- State the theorem
theorem angle_between_vectors 
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : ‖a‖ = ‖b‖)
  (h5 : ‖b‖ = ‖c‖)
  (h6 : a + b = c) : 
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l453_45376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_dandelions_approx_l453_45349

/-- The average number of dandelions picked by Billy, George, and Susan -/
noncomputable def average_dandelions : ℝ :=
  let billy_initial : ℝ := 36
  let george_initial : ℝ := (2/5) * billy_initial
  let susan_initial : ℝ := 0.75 * george_initial
  let billy_additional : ℝ := 5/3
  let george_additional : ℝ := 7/2
  let susan_additional : ℝ := 0.1 * susan_initial
  let billy_total : ℝ := billy_initial + billy_additional
  let george_total : ℝ := george_initial + george_additional
  let susan_total : ℝ := susan_initial + susan_additional
  let total_dandelions : ℝ := billy_total + george_total + susan_total
  total_dandelions / 3

/-- The average number of dandelions picked is approximately 22.4822 -/
theorem average_dandelions_approx :
  ∃ ε > 0, |average_dandelions - 22.4822| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_dandelions_approx_l453_45349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_slope_l453_45399

/-- The slope of a line intersecting a hyperbola's asymptotes -/
theorem hyperbola_line_slope (k : ℝ) : 
  let C : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2 = 1
  let A : ℝ × ℝ := (1, 0)
  let l : ℝ → ℝ := λ x ↦ k * x - k
  let P : ℝ × ℝ := (k / (k - 1), k / (k - 1))
  let Q : ℝ × ℝ := (k / (k + 1), -k / (k + 1))
  C A.1 A.2 ∧ 
  (∀ x, l x = x ∨ l x = -x → (x ≠ 0 ∧ C x (l x))) ∧
  l A.1 = A.2 ∧
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = 4 * ((Q.1 - A.1)^2 + (Q.2 - A.2)^2) →
  k = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_slope_l453_45399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_configuration_impossible_l453_45345

/-- Represents a tower with a base point and height -/
structure Tower where
  base : ℝ × ℝ
  height : ℝ

/-- Configuration of two towers -/
structure TowerConfiguration where
  tower1 : Tower
  tower2 : Tower
  view_angle : ℝ
  base_angle : ℝ

/-- Checks if the given tower configuration is possible -/
def is_valid_configuration (config : TowerConfiguration) : Prop :=
  let (x1, y1) := config.tower1.base
  let (x2, y2) := config.tower2.base
  let h1 := config.tower1.height
  let h2 := config.tower2.height
  let α := config.view_angle
  let β := config.base_angle
  
  -- Conditions
  h1 = 42 ∧
  h2 = 56 ∧
  y2 - y1 = 10 ∧
  α = 16 * (Real.pi / 180) ∧ -- Convert degrees to radians
  β = 20 * (Real.pi / 180) ∧ -- Convert degrees to radians
  
  -- The angle formed at the top of the taller tower
  Real.arctan ((x2 - x1) / (h2 - h1 + y2 - y1)) = α ∧
  
  -- The angle formed by the line connecting the bases
  Real.arctan ((y2 - y1) / (x2 - x1)) = β

theorem tower_configuration_impossible :
  ¬ ∃ (config : TowerConfiguration), is_valid_configuration config := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_configuration_impossible_l453_45345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_problem_l453_45308

theorem stratified_sampling_problem :
  let total_population : ℕ := 800
  let sample_size : ℕ := 40
  let strata_sizes : List ℕ := [160, 320, 200, 120]
  let sample_proportion : ℚ := sample_size / total_population

  (∀ (stratum_size : ℕ), stratum_size ∈ strata_sizes →
    (Int.toNat (Int.floor (sample_proportion * stratum_size)) : ℕ) = 
    (if stratum_size = 160 then 8
     else if stratum_size = 320 then 16
     else if stratum_size = 200 then 10
     else 6)) ∧
  (List.sum (strata_sizes.map (λ x => (Int.toNat (Int.floor (sample_proportion * x)) : ℕ))) = sample_size) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_problem_l453_45308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_cycle_l453_45344

/-- The function g(x) = 5x - x^2 - 1 -/
def g (x : ℝ) : ℝ := 5 * x - x^2 - 1

/-- The sequence x_n defined by x_n = g(x_{n-1}) for all n ≥ 1 -/
def x_seq (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => g (x_seq x₀ n)

/-- A predicate to check if a sequence cycles through exactly three distinct values -/
def cycles_three (x₀ : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ n : ℕ, x_seq x₀ n = a ∨ x_seq x₀ n = b ∨ x_seq x₀ n = c) ∧
    (∀ n : ℕ, x_seq x₀ n = a → x_seq x₀ (n + 1) = b) ∧
    (∀ n : ℕ, x_seq x₀ n = b → x_seq x₀ (n + 1) = c) ∧
    (∀ n : ℕ, x_seq x₀ n = c → x_seq x₀ (n + 1) = a)

/-- Theorem: There are no real numbers x₀ such that the sequence cycles through exactly three distinct values perpetually -/
theorem no_three_cycle : ¬∃ x₀ : ℝ, cycles_three x₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_cycle_l453_45344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_3_proposition_4_l453_45356

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (belongs_to : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (intersection : Plane → Plane → Line)

-- Define the theorem of perpendicularity between planes
axiom perpendicularity_planes_theorem 
  (α β γ : Plane) (m : Line) :
  perpendicular_plane α β → 
  perpendicular_plane α γ → 
  intersection β γ = m → 
  perpendicular_line_plane m α

-- Define the theorem of perpendicularity and parallelism
axiom perpendicular_parallel_theorem 
  (m n : Line) (α : Plane) :
  perpendicular_line_plane m α → 
  perpendicular_lines m n → 
  parallel_plane_line α n

-- Proposition 3
theorem proposition_3 
  (α β γ : Plane) (m : Line) :
  perpendicular_plane α β → 
  perpendicular_plane α γ → 
  intersection β γ = m → 
  perpendicular_line_plane m α := by
  sorry

-- Proposition 4
theorem proposition_4 
  (m n : Line) (α β : Plane) :
  perpendicular_line_plane m α → 
  perpendicular_line_plane n β → 
  perpendicular_lines m n → 
  perpendicular_plane α β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_3_proposition_4_l453_45356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_tangent_sum_l453_45348

/-- Given an angle α in the Cartesian coordinate system with its vertex at the origin,
    initial side along the positive x-axis, and a point P(-2t, t) (t ≠ 0) on its terminal side,
    prove that tan(α + π/4) = 1/3 -/
theorem angle_tangent_sum (t : ℝ) (h : t ≠ 0) :
  let α := Real.arctan (-1/2)
  Real.tan (α + π/4) = 1/3 := by
  -- Define α
  let α := Real.arctan (-1/2)
  
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_tangent_sum_l453_45348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_approx_1_33_l453_45317

/-- Represents a rectangular park with given properties -/
structure RectangularPark where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing_cost_per_meter : ℝ
  exchange_rate : ℝ

/-- Calculates the fencing cost in USD for a given rectangular park -/
noncomputable def fencing_cost_usd (park : RectangularPark) : ℝ :=
  let perimeter := 2 * (park.length + park.width)
  let cost_inr := perimeter * park.fencing_cost_per_meter
  cost_inr / park.exchange_rate

/-- Theorem stating the fencing cost for a park with specific properties -/
theorem fencing_cost_approx_1_33 (park : RectangularPark) 
  (h1 : park.length / park.width = 3 / 2)
  (h2 : park.length * park.width = 2400)
  (h3 : park.fencing_cost_per_meter = 0.5)
  (h4 : park.exchange_rate = 75) :
  ∃ ε > 0, |fencing_cost_usd park - 1.33| < ε := by
  sorry

#check fencing_cost_approx_1_33

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_approx_1_33_l453_45317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pairs_satisfy_l453_45361

/-- The number of distinct ordered pairs of positive integers (m, n) satisfying 1/m + 1/n = 1/5 -/
def count_pairs : ℕ := 3

/-- Predicate to check if a pair of positive integers satisfies the equation -/
def satisfies_equation (m n : ℕ+) : Prop :=
  (1 : ℚ) / m.val + (1 : ℚ) / n.val = (1 : ℚ) / 5

/-- Theorem stating that there are exactly 3 distinct ordered pairs satisfying the equation -/
theorem three_pairs_satisfy :
  (∃! (pairs : Finset (ℕ+ × ℕ+)), pairs.card = count_pairs ∧
    ∀ (p : ℕ+ × ℕ+), p ∈ pairs ↔ satisfies_equation p.1 p.2) :=
by sorry

#eval count_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pairs_satisfy_l453_45361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l453_45363

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {x : ℕ | x > 1 ∧ x < 5}

theorem intersection_A_B : A ∩ B = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l453_45363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l453_45313

/-- Bitwise OR operation -/
def bitwiseOr (x y : ℕ) : ℕ := sorry

/-- The number of valid sequence pairs -/
def numValidPairs : ℕ := (2^2019 - 1)^2018

theorem count_valid_sequences :
  (∃ (a : Fin 2019 → ℕ) (b c : Fin 2018 → ℕ),
    (∀ i : Fin 2019, a i < 2^2018) ∧
    (∀ i : Fin 2018, b i = a i.val + a (i.succ)) ∧
    (∀ i : Fin 2018, c i = bitwiseOr (a i.val) (a (i.succ)))) ↔
  numValidPairs = (2^2019 - 1)^2018 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l453_45313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_three_quadratics_l453_45382

/-- Helper function to count the number of real roots of a quadratic polynomial -/
noncomputable def count_roots (f : ℝ → ℝ) : ℕ := sorry

/-- Given three positive real numbers a, b, and c, the maximum total number of real roots
    for the polynomials ax^2 + bx + c, bx^2 + cx + a, and cx^2 + ax + b is 4. -/
theorem max_roots_three_quadratics (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (n : ℕ),
    n ≤ 4 ∧
    (∀ (m : ℕ),
      (∃ (x y z : ℝ),
        m = count_roots (λ t ↦ a*t^2 + b*t + c) +
            count_roots (λ t ↦ b*t^2 + c*t + a) +
            count_roots (λ t ↦ c*t^2 + a*t + b)) →
      m ≤ n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_three_quadratics_l453_45382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l453_45383

/-- Represents an ellipse with center at origin -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  h_positive : 0 < a ∧ 0 < b
  h_major : a ≥ b

/-- Represents a line -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Theorem: Chord length of an ellipse intersected by a line through focus -/
theorem ellipse_chord_length 
  (e : Ellipse) 
  (l : Line) 
  (h_major_axis : e.a = Real.sqrt 2)
  (h_eccentricity : (Real.sqrt ((e.a^2 - e.b^2) / e.a^2)) = Real.sqrt 2 / 2)
  (h_focus : l.intercept = -1)  -- Line passes through right focus (1, 0)
  (h_slope : l.slope = 1) :
  Real.sqrt ((4/3)^2 * (1 + l.slope^2)) = 4 * Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l453_45383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_properties_l453_45329

noncomputable section

open Real

-- Define the functions
def f (x : ℝ) := sin (x + π / 3)
def g (x : ℝ) := cos (2 * x)

-- Define the propositions
def p : Prop := ∃ (T : ℝ), T > 0 ∧ T = π / 2 ∧ ∀ (x : ℝ), g x = g (x + T)
def q : Prop := ∀ (x : ℝ), f (π / 6 + x) = f (π / 6 - x)

-- State the theorem
theorem trigonometric_properties : ¬p ∧ q ∧ ¬¬q := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_properties_l453_45329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l453_45343

open BigOperators

variable {n : ℕ}
variable (a : Fin n → ℝ)

theorem inequality_proof (h1 : ∀ i, 0 ≤ a i) (h2 : ∑ i, a i = n) :
  ∑ i, (a i)^2 / (1 + (a i)^4) ≤ ∑ i, 1 / (1 + a i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l453_45343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l453_45307

noncomputable section

-- Define the circle center
def C : ℝ × ℝ := (-1/2, 3)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x + 1/2)^2 + (y - 3)^2 = 25/4

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the dot product of two vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- State the theorem
theorem circle_equation :
  ∀ P Q : ℝ × ℝ,
  line_eq P.1 P.2 ∧ line_eq Q.1 Q.2 ∧  -- P and Q are on the line
  dot_product (P.1 - O.1, P.2 - O.2) (Q.1 - O.1, Q.2 - O.2) = 0 →  -- OP · OQ = 0
  ∀ x y : ℝ,
  circle_eq x y ↔ (x - C.1)^2 + (y - C.2)^2 = ((x + 1/2)^2 + (y - 3)^2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l453_45307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_power_equality_l453_45389

theorem coprime_power_equality (a b m n : ℕ) (ha : a > 0) (hb : b > 0) (hm : m > 0) (hn : n > 0) :
  Nat.Coprime m n →
  (a^2 + b^2)^m = (a * b)^n →
  ∃ r : ℕ, r > 0 ∧ a = 2^r ∧ b = 2^r ∧ m = 2 * r ∧ n = 2 * r + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_power_equality_l453_45389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l453_45335

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, Real.sin x₀ + Real.cos x₀ ≤ Real.sqrt 2) ↔
  (∀ x : ℝ, Real.sin x + Real.cos x > Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l453_45335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_arc_length_l453_45340

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x^2)

noncomputable def a : ℝ := 0
noncomputable def b : ℝ := 1/4

noncomputable def arcLength (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, sqrt (1 + (deriv f x)^2)

theorem curve_arc_length :
  arcLength f a b = (1/2) * log (5/3) + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_arc_length_l453_45340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_characterization_l453_45346

/-- A function that returns the number of ways to represent an integer as a sum of distinct elements from a set -/
def numberOfWays (S : Set ℕ) (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem power_of_two_characterization :
  ∀ k : ℕ+, (∃ m : ℕ, ∃ S : Set ℕ, 
    ∀ n : ℕ, n > m → numberOfWays S n = k) ↔ 
  (∃ m : ℕ, k = 2^m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_characterization_l453_45346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_plus_n_zero_l453_45323

/-- Represents an arithmetic progression. -/
structure ArithmeticProgression where
  first : ℚ  -- First term
  diff : ℚ   -- Common difference

/-- Sum of the first k terms of an arithmetic progression. -/
def sum_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  (k : ℚ) / 2 * (2 * ap.first + (k - 1 : ℚ) * ap.diff)

/-- 
Given an arithmetic progression where the sum of its first m terms 
is equal to the sum of its first n terms (m ≠ n), 
prove that the sum of its first m+n terms is zero.
-/
theorem sum_m_plus_n_zero 
  (ap : ArithmeticProgression) (m n : ℕ) 
  (h_neq : m ≠ n) (h_eq : sum_k ap m = sum_k ap n) : 
  sum_k ap (m + n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_plus_n_zero_l453_45323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_break_theorem_l453_45352

def is_valid_coffee_group (n : ℕ) (k : ℕ) : Prop :=
  n = 14 ∧ 0 < k ∧ k < 7 ∧ n - 2 * k = 14 - 2 * k

theorem coffee_break_theorem :
  ∀ m : ℕ, (∃ k : ℕ, is_valid_coffee_group 14 k ∧ m = 14 - 2 * k) ↔ m ∈ ({6, 8, 10, 12} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_break_theorem_l453_45352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_15_dividing_30_factorial_l453_45374

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem largest_power_of_15_dividing_30_factorial : 
  (∃ n : ℕ, 15^n ∣ factorial 30 ∧ ∀ m : ℕ, 15^m ∣ factorial 30 → m ≤ n) ∧ 
  (∀ n : ℕ, 15^n ∣ factorial 30 → n ≤ 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_15_dividing_30_factorial_l453_45374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_property_l453_45332

-- Define the polynomial type
def MyPolynomial (R : Type*) [Ring R] := R → R

-- Define the property that P(2x) = P'(x) * P''(x) for all real x
def SatisfiesProperty (P : MyPolynomial ℝ) : Prop :=
  ∀ x : ℝ, P (2 * x) = (deriv P x) * (deriv (deriv P) x)

-- State the theorem
theorem unique_polynomial_property :
  ∃! P : MyPolynomial ℝ, SatisfiesProperty P ∧ 
  ∀ x : ℝ, P x = (4 / 9 : ℝ) * x^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_property_l453_45332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_3_l453_45326

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (Real.pi * x / 6) else 1 - 2 * x

theorem f_composition_at_3 : f (f 3) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_3_l453_45326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_less_than_critical_value_expected_males_in_sample_probability_two_out_of_three_l453_45306

-- Define the population
def population : Nat := 100

-- Define exercise time categories
inductive ExerciseTime
| LessThan10 : ExerciseTime
| From10To20 : ExerciseTime
| From20To30 : ExerciseTime
| From30To40 : ExerciseTime
| From40To50 : ExerciseTime
| From50To60 : ExerciseTime

-- Define function to count people in each category
def count_people (t : ExerciseTime) : Nat :=
  match t with
  | ExerciseTime.LessThan10 => 10
  | ExerciseTime.From10To20 => 18
  | ExerciseTime.From20To30 => 22
  | ExerciseTime.From30To40 => 25
  | ExerciseTime.From40To50 => 20
  | ExerciseTime.From50To60 => 5

-- Define function to check if exercise time meets the standard
def meets_standard (t : ExerciseTime) : Bool :=
  match t with
  | ExerciseTime.From40To50 => true
  | ExerciseTime.From50To60 => true
  | _ => false

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  !![30, 15; 45, 10]

-- Theorem 1: χ² value is less than 3.841
theorem chi_square_less_than_critical_value :
  let n := population
  let a := contingency_table 0 0
  let b := contingency_table 0 1
  let c := contingency_table 1 0
  let d := contingency_table 1 1
  let chi_square : ℚ := (n : ℚ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))
  chi_square < 3841/1000 := by sorry

-- Theorem 2: Expected number of males in sample is 6/5
theorem expected_males_in_sample :
  let total_sample : ℚ := 5
  let males_in_sample : ℚ := 2
  let selected_sample : ℚ := 3
  let expected_males := males_in_sample * selected_sample / total_sample
  expected_males = 6/5 := by sorry

-- Theorem 3: Probability of exactly 2 out of 3 meeting the standard is 9/64
theorem probability_two_out_of_three :
  let p_meets_standard : ℚ := 25 / 100
  let probability := (Nat.choose 3 2 : ℚ) * p_meets_standard^2 * (1 - p_meets_standard)
  probability = 9/64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_less_than_critical_value_expected_males_in_sample_probability_two_out_of_three_l453_45306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_escape_strategy_l453_45395

/-- Represents the state of a door (locked or unlocked) -/
inductive DoorState
| Locked
| Unlocked

/-- Represents a circular room with 6 doors -/
structure Room where
  doors : Fin 6 → DoorState
  openDoorCount : Nat
  openDoorCount_eq : openDoorCount = 1

/-- Represents Ivan's attempt to check doors -/
def Attempt (doors : Fin 3 → Fin 6) (r : Room) : Bool :=
  sorry

/-- Baba Yaga's action of locking the open door and unlocking a neighboring door -/
def babaYagaAction (r : Room) : Room :=
  sorry

/-- Theorem stating that checking even-numbered doors twice guarantees exit -/
theorem ivan_escape_strategy (r : Room) :
  ∃ (attempt1 attempt2 : Fin 3 → Fin 6),
    (∀ i : Fin 3, attempt1 i % 2 = 0) ∧
    (∀ i : Fin 3, attempt2 i % 2 = 0) ∧
    (Attempt attempt1 r ∨ Attempt attempt2 (babaYagaAction r)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_escape_strategy_l453_45395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l453_45330

-- Define the function f
variable (f : ℝ → ℝ → ℝ)

-- Define the transformations
def symmetrical_transform (f : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ :=
  λ x y ↦ f (2 - x) (-2 - y)

def reflect_about_y_eq_neg_x (f : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ :=
  λ x y ↦ f y x

-- Theorem statement
theorem function_transformation (x y : ℝ) :
  (reflect_about_y_eq_neg_x (symmetrical_transform f)) (x + 2) (y - 2) = 0 := by
  sorry

#check function_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l453_45330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_squares_area_ratio_l453_45378

/-- A square with vertices A, B, C, D -/
structure Square (α : Type*) [NormedAddCommGroup α] where
  A : α
  B : α
  C : α
  D : α

/-- The side length of a square -/
def sideLength {α : Type*} [NormedAddCommGroup α] (s : Square α) : ℝ :=
  ‖s.B - s.A‖

/-- The area of a square -/
def area {α : Type*} [NormedAddCommGroup α] (s : Square α) : ℝ :=
  (sideLength s) ^ 2

/-- Theorem: Ratio of areas of nested squares -/
theorem nested_squares_area_ratio
  {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]
  (ABCD : Square α) (IJKL : Square α) (I : α) :
  sideLength ABCD = 16 →
  ‖I - ABCD.A‖ = 3 * ‖ABCD.B - I‖ →
  (area IJKL) / (area ABCD) = 1/8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_squares_area_ratio_l453_45378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_determination_l453_45334

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem function_determination (ω φ : ℝ) :
  f ω φ (π / 9) = 2 →
  π / 3 = π / ω →
  ∃ k : ℤ, φ = π / 6 + 2 * k * π ∧ ω = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_determination_l453_45334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_intersections_exist_l453_45342

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The cosine function -/
noncomputable def cosine_function (x : ℝ) : ℝ := Real.cos x

/-- Predicate for a point being on a circle -/
def on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1) ^ 2 + (p.2 - c.center.2) ^ 2 = c.radius ^ 2

/-- Predicate for a point being on the cosine curve -/
def on_cosine_curve (p : ℝ × ℝ) : Prop :=
  p.2 = cosine_function p.1

/-- The main theorem statement -/
theorem infinite_intersections_exist :
  ∃ (c : Circle), ∀ (n : ℕ), ∃ (points : Fin n → ℝ × ℝ),
    (∀ i j, i ≠ j → points i ≠ points j) ∧
    (∀ i, on_circle c (points i) ∧ on_cosine_curve (points i)) := by
  sorry

#check infinite_intersections_exist

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_intersections_exist_l453_45342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_eaten_twice_pyramid_l453_45366

/-- The volume of a sphere with radius 1 -/
noncomputable def unit_sphere_volume : ℝ := (4 / 3) * Real.pi

/-- The number of vertices in a pentagonal prism -/
def pentagonal_prism_vertices : ℕ := 10

/-- The number of vertices in a quadrangular pyramid -/
def quadrangular_pyramid_vertices : ℕ := 5

/-- The volume of cheese eaten from a pentagonal prism -/
noncomputable def prism_eaten_volume : ℝ := pentagonal_prism_vertices * unit_sphere_volume

/-- The volume of cheese eaten from a quadrangular pyramid -/
noncomputable def pyramid_eaten_volume : ℝ := quadrangular_pyramid_vertices * unit_sphere_volume

theorem prism_eaten_twice_pyramid :
  prism_eaten_volume = 2 * pyramid_eaten_volume := by
  unfold prism_eaten_volume pyramid_eaten_volume
  unfold pentagonal_prism_vertices quadrangular_pyramid_vertices
  simp [unit_sphere_volume]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_eaten_twice_pyramid_l453_45366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_plot_area_l453_45362

-- Define the trapezoid dimensions
def bottom_base : ℝ := 20
def top_base : ℝ := 25
def trapezoid_height : ℝ := 15  -- Changed variable name to avoid conflict

-- Define the conversion factor from square miles to acres
def square_mile_to_acres : ℝ := 640

-- Theorem statement
theorem trapezoid_plot_area :
  let area_sq_miles := (bottom_base + top_base) * trapezoid_height / 2
  area_sq_miles * square_mile_to_acres = 216000 := by
  -- Unfold the definition of area_sq_miles
  unfold bottom_base top_base trapezoid_height square_mile_to_acres
  -- Simplify the expression
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- Prove the equality
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_plot_area_l453_45362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_product_l453_45368

theorem tan_half_product (α β : ℝ) 
  (h1 : Real.sin α + Real.sin β = 2 * Real.sin (α + β))
  (h2 : ∀ n : ℤ, α + β ≠ 2 * Real.pi * (n : ℝ)) :
  Real.tan (α / 2) * Real.tan (β / 2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_product_l453_45368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_fruit_cost_l453_45370

/-- Calculates the average cost per fruit given the costs and quantities of different fruits --/
theorem average_fruit_cost
  (apple_cost banana_cost orange_cost : ℚ)
  (apple_qty banana_qty orange_qty : ℕ)
  (h_apple_cost : apple_cost = 2)
  (h_banana_cost : banana_cost = 1)
  (h_orange_cost : orange_cost = 3)
  (h_apple_qty : apple_qty = 12)
  (h_banana_qty : banana_qty = 4)
  (h_orange_qty : orange_qty = 4) :
  (apple_cost * apple_qty + banana_cost * banana_qty + orange_cost * orange_qty) /
  (apple_qty + banana_qty + orange_qty : ℚ) = 2 := by
  sorry

#check average_fruit_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_fruit_cost_l453_45370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chords_theorem_l453_45347

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a structure for a chord
structure Chord where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ
  midpoint : ℝ × ℝ
  passes_through_focus : (start.1 - 1) * (endpoint.2 - 0) = (endpoint.1 - 1) * (start.2 - 0)

-- Define two perpendicular chords
def perpendicular_chords (ab cd : Chord) : Prop :=
  (ab.endpoint.1 - ab.start.1) * (cd.endpoint.1 - cd.start.1) +
  (ab.endpoint.2 - ab.start.2) * (cd.endpoint.2 - cd.start.2) = 0

-- Define the theorem
theorem parabola_chords_theorem
  (ab cd : Chord)
  (h_ab_parabola : parabola ab.start.1 ab.start.2 ∧ parabola ab.endpoint.1 ab.endpoint.2)
  (h_cd_parabola : parabola cd.start.1 cd.start.2 ∧ parabola cd.endpoint.1 cd.endpoint.2)
  (h_perpendicular : perpendicular_chords ab cd) :
  -- Part 1: Line MN passes through (3, 0)
  ∃ (k : ℝ), ab.midpoint.2 - cd.midpoint.2 = k * (ab.midpoint.1 - cd.midpoint.1) ∧
             cd.midpoint.2 = k * (cd.midpoint.1 - 3) ∧
  -- Part 2: Locus of midpoint H of common chord
  ∃ (H : ℝ × ℝ) (k : ℝ), H.2 = k * H.1 * (H.1 - 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chords_theorem_l453_45347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_fourth_quadrant_l453_45310

theorem sin_double_angle_fourth_quadrant (α : ℝ) 
  (h1 : Real.cos α = 4/5) 
  (h2 : α ∈ Set.Ioo (-π/2) 0) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_fourth_quadrant_l453_45310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersect_l453_45341

/-- A circle with a given diameter and a line at a given distance from its center -/
structure CircleAndLine where
  diameter : ℝ
  distance_to_line : ℝ

/-- The relationship between a circle and a line -/
inductive Relationship
  | Intersecting
  | NonIntersecting
  | Tangent
  | ExternallyTangent

/-- Determine the relationship between a circle and a line -/
noncomputable def determine_relationship (c : CircleAndLine) : Relationship :=
  if c.distance_to_line < c.diameter / 2 then
    Relationship.Intersecting
  else if c.distance_to_line = c.diameter / 2 then
    Relationship.Tangent
  else
    Relationship.NonIntersecting

/-- Theorem stating that for the given circle and line, they intersect -/
theorem circle_and_line_intersect (c : CircleAndLine) 
  (h1 : c.diameter = 8)
  (h2 : c.distance_to_line = 3) :
  determine_relationship c = Relationship.Intersecting := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersect_l453_45341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_tangents_l453_45304

/-- Circle represented by its equation in the form (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  r : ℝ  -- radius
  r_pos : r > 0

def circle1 : Circle := {
  h := 1
  k := 0
  r := 1
  r_pos := by norm_num
}

def circle2 : Circle := {
  h := 0
  k := -2
  r := 2
  r_pos := by norm_num
}

/-- Distance between the centers of two circles -/
noncomputable def centerDistance (c1 c2 : Circle) : ℝ :=
  Real.sqrt ((c1.h - c2.h)^2 + (c1.k - c2.k)^2)

/-- Number of common tangents between two circles -/
def number_of_common_tangents (c1 c2 : Circle) : ℕ := sorry

theorem two_common_tangents (c1 c2 : Circle) :
  c1 = circle1 → c2 = circle2 →
  ∃ (n : ℕ), n = 2 ∧ n = number_of_common_tangents c1 c2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_tangents_l453_45304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_30_l453_45360

/-- The function representing the total cost -/
noncomputable def f (x : ℝ) : ℝ := (600 / x) * 6 + 4 * x

/-- Theorem stating that 30 minimizes the total cost function -/
theorem min_cost_at_30 :
  ∀ x > 0, f x ≥ f 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_30_l453_45360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robert_travel_time_l453_45300

/-- Represents the time taken to drive from city A to B -/
def time_A_to_B : ℝ := 3

/-- The distance between city A and B in miles -/
def distance : ℝ := 180

/-- The time taken for the return trip from B to A in hours -/
def time_B_to_A : ℝ := 2.5

/-- The average speed for the round trip if 30 minutes were saved on each trip, in miles per hour -/
def avg_speed_with_savings : ℝ := 80

/-- Time saved on each trip in hours -/
def time_saved : ℝ := 0.5

theorem robert_travel_time : time_A_to_B = 3 :=
  by
    -- The proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robert_travel_time_l453_45300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_time_to_school_l453_45350

/-- Represents a person's walking parameters -/
structure WalkingParams where
  steps_per_minute : ℝ
  step_length : ℝ
  time_to_school : ℝ

/-- Calculates the distance to school based on walking parameters -/
def distance_to_school (params : WalkingParams) : ℝ :=
  params.steps_per_minute * params.step_length * params.time_to_school

/-- Theorem: Mike's time to walk to school is 19 2/7 minutes -/
theorem mike_time_to_school 
  (dave : WalkingParams) 
  (mike : WalkingParams) 
  (h1 : dave.steps_per_minute = 85)
  (h2 : dave.step_length = 70)
  (h3 : dave.time_to_school = 20)
  (h4 : mike.steps_per_minute = 95)
  (h5 : mike.step_length = 65)
  (h6 : distance_to_school dave = distance_to_school mike) :
  mike.time_to_school = 19 + 2 / 7 := by
  sorry

#check mike_time_to_school

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_time_to_school_l453_45350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_t_is_1_range_of_a_when_t_is_1_range_of_t_l453_45364

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 2

-- Part 1
theorem range_of_f_when_t_is_1 :
  ∀ y ∈ Set.Icc 1 10, ∃ x ∈ Set.Icc 0 4, f 1 x = y :=
sorry

-- Part 2
theorem range_of_a_when_t_is_1 :
  ∀ a : ℝ, (∀ x ∈ Set.Icc a (a + 2), f 1 x ≤ 5) ↔ a ∈ Set.Icc (-1) 1 :=
sorry

-- Part 3
theorem range_of_t :
  ∀ t : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 4 → x₂ ∈ Set.Icc 0 4 → |f t x₁ - f t x₂| ≤ 8) ↔
  t ∈ Set.Icc (4 - 2*Real.sqrt 2) (2*Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_t_is_1_range_of_a_when_t_is_1_range_of_t_l453_45364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_swap_and_triple_l453_45333

open Matrix

theorem row_swap_and_triple (N : Matrix (Fin 3) (Fin 3) ℝ) :
  let P : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 0, 3, 0; 1, 0, 0]
  P • N = !![N 2 0, N 2 1, N 2 2;
             3 * N 1 0, 3 * N 1 1, 3 * N 1 2;
             N 0 0, N 0 1, N 0 2] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_swap_and_triple_l453_45333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l453_45314

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x^2 - x - 3)

theorem f_monotone_increasing : 
  MonotoneOn f (Set.Ici (3/2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l453_45314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l453_45357

theorem complex_equation_solution (z : ℂ) :
  (1 + 2*I)*z = 4 + 3*I → z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l453_45357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_coordinate_l453_45381

/-- Given points A, B, and C in a 2D plane, if AB is parallel to AC, then the y-coordinate of C is 5. -/
theorem parallel_line_coordinate (A B C : ℝ × ℝ) (lambda : ℝ) : 
  A = (-1, -1) → B = (1, 3) → C = (2, lambda) → 
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) → 
  lambda = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_coordinate_l453_45381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l453_45311

/-- The midpoint of a segment after translation -/
theorem midpoint_after_translation (x₁ y₁ x₂ y₂ dx dy : ℝ) :
  let s₁_start := (x₁, y₁)
  let s₁_end := (x₂, y₂)
  let s₁_midpoint := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  let s₂_midpoint := (s₁_midpoint.1 + dx, s₁_midpoint.2 + dy)
  s₁_start = (3, -2) ∧ s₁_end = (-7, 4) ∧ dx = 3 ∧ dy = -2 →
  s₂_midpoint = (1, -1) := by
  intro h
  -- Proof steps would go here
  sorry

#check midpoint_after_translation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l453_45311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marys_average_speed_l453_45309

noncomputable def home_to_school_distance : ℝ := 1.5
noncomputable def school_to_library_distance : ℝ := 0.5
noncomputable def home_to_school_time : ℝ := 45 / 60
noncomputable def school_to_home_time : ℝ := 15 / 60
noncomputable def school_to_library_time : ℝ := 15 / 60

theorem marys_average_speed :
  let total_distance := 2 * home_to_school_distance + school_to_library_distance
  let total_time := home_to_school_time + school_to_home_time + school_to_library_time
  total_distance / total_time = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marys_average_speed_l453_45309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_and_sin_2alpha_values_l453_45322

theorem phi_and_sin_2alpha_values 
  (φ : ℝ) 
  (α : ℝ) 
  (h1 : 0 < φ ∧ φ < Real.pi) 
  (h2 : Real.sin (φ + Real.pi/4) = Real.sin (φ - Real.pi/4)) 
  (h3 : Real.pi/4 < α ∧ α < Real.pi/2) 
  (h4 : Real.sin (2*α + φ/2) = -5/13) : 
  φ = Real.pi/2 ∧ Real.sin (2*α) = 7*Real.sqrt 2/26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_and_sin_2alpha_values_l453_45322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_l453_45319

/-- Ceiling function -/
def ceil (x : ℚ) : ℤ := Int.ceil x

/-- Sequence a_n -/
def a (p : ℕ) : ℕ → ℕ
  | 0 => 2
  | n+1 => (a p n) + (ceil ((p * (a p n) : ℚ) / ((n+1) : ℚ))).toNat

theorem sequence_divisibility (p : ℕ) (h_p_prime : Nat.Prime p) (h_p_plus_2_prime : Nat.Prime (p + 2)) (h_p_gt_3 : p > 3) :
  ∀ n, 3 ≤ n → n < p → (n : ℕ) ∣ (p * a p (n-1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_l453_45319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_time_ratio_l453_45358

/-- Proves that the ratio of time taken to travel from A to B to the time taken to travel from B to C is 2:1 -/
theorem motorcycle_time_ratio (distance_AB : ℝ) (average_speed : ℝ) :
  distance_AB = 120 →
  average_speed = 25 →
  (distance_AB / average_speed) / ((distance_AB / 2) / average_speed) = 2 := by
  intro h_distance h_speed
  -- Simplify the expression
  simp [h_distance, h_speed]
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_time_ratio_l453_45358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sequence_symmetry_l453_45373

/-- A sequence of binary digits -/
def BinarySequence := List Bool

/-- Swaps two elements in a list at given indices -/
def swap (xs : List α) (i j : Nat) : List α :=
  match xs.get? i, xs.get? j with
  | some x, some y => xs.set i y |>.set j x
  | _, _ => xs

/-- Checks if a list is symmetrical about its middle element -/
def isSymmetrical (xs : List α) : Prop :=
  let n := xs.length
  ∀ i, i < n / 2 → xs.get? i = xs.get? (n - 1 - i)

/-- The main theorem stating that any binary sequence of length 1999 can be made symmetrical -/
theorem binary_sequence_symmetry (seq : BinarySequence) (h : seq.length = 1999) :
  ∃ (swaps : List (Nat × Nat)), swaps.length ≤ 999 ∧
    isSymmetrical (swaps.foldl (fun s (i, j) => swap s i j) seq) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sequence_symmetry_l453_45373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l453_45377

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given non-zero vectors OA and OB that are not collinear, and the relations
    2 * OP = x * OA + y * OB and PA = lambda * AB, prove that x + y - 2 = 0 --/
theorem vector_relation (OA OB OP : V) (x y lambda : ℝ) 
    (h₁ : OA ≠ 0)
    (h₂ : OB ≠ 0)
    (h₃ : ¬ ∃ (k : ℝ), OA = k • OB)
    (h₄ : (2 : ℝ) • OP = x • OA + y • OB)
    (h₅ : OP - OA = lambda • (OB - OA)) :
  x + y - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l453_45377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l453_45331

theorem triangle_angle_theorem (a b c : ℝ) (h : a^2 = b^2 + c^2 - b*c) :
  Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l453_45331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l453_45321

theorem teacher_age (num_students : ℕ) (avg_age_students : ℚ) (avg_increase : ℚ) : 
  num_students = 20 → 
  avg_age_students = 21 → 
  avg_increase = 1 → 
  (num_students * avg_age_students + 42) / (num_students + 1) = avg_age_students + avg_increase := by
  sorry

#check teacher_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l453_45321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_and_value_l453_45305

-- Define the power function as noncomputable
noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_through_point_and_value :
  ∃ α : ℝ, powerFunction α 2 = (1 : ℝ) / 2 → powerFunction α 4 = (1 : ℝ) / 4 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_and_value_l453_45305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l453_45380

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  abs (l.a * p.1 + l.b * p.2 + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- The circle x^2 + y^2 = 2r -/
noncomputable def problemCircle (r : ℝ) : Circle :=
  { center := (0, 0), radius := Real.sqrt (2 * r) }

/-- The line x - y = r -/
def problemLine (r : ℝ) : Line :=
  { a := 1, b := -1, c := -r }

/-- The theorem to be proved -/
theorem line_tangent_to_circle (r : ℝ) : 
  distancePointToLine (problemCircle r).center (problemLine r) = (problemCircle r).radius ↔ r = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l453_45380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l453_45398

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2 - 2*(x - 1) + 1

-- Define the domain of f(x+1)
def domain : Set ℝ := Set.Icc (-2) 6

-- Theorem statement
theorem decreasing_interval_of_f :
  ∃ (a b : ℝ), a = -1 ∧ b = 2 ∧
  (∀ (x y : ℝ), x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x > f y) ∧
  (∀ (x y : ℝ), x ∈ domain → y ∈ domain → x < y → f (x + 1) ≤ f (y + 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l453_45398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l453_45365

theorem equation_solution : ∃ x : ℝ, (4 : ℝ) ^ (Real.sqrt (16 ^ x)) = (16 : ℝ) ^ (Real.sqrt (2 ^ x)) ∧ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l453_45365
