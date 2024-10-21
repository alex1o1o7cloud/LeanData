import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l824_82476

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola with parameter p -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The area of the triangle formed by the origin and the intersection points
    of the hyperbola's asymptotes with the parabola -/
noncomputable def triangle_area (h : Hyperbola) (p : Parabola) : ℝ :=
  (p.p * h.b) / (2 * h.a)

/-- Main theorem stating the relationship between the hyperbola, parabola, and p value -/
theorem hyperbola_parabola_intersection
  (h : Hyperbola) (p : Parabola)
  (h_eccentricity : eccentricity h = 2)
  (h_area : triangle_area h p = Real.sqrt 3 / 3) :
  p.p = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l824_82476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l824_82481

-- Define the function f as noncomputable due to the use of real numbers
noncomputable def f (x a b : ℝ) : ℝ := (x^2)^(1/3) + (a - 2) / x + b

-- State the theorem
theorem even_function_implies_a_equals_two (a b : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f x a b = f (-x) a b) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l824_82481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_two_fours_up_to_243_l824_82496

def has_two_fours (n : ℕ) : Bool :=
  (n.digits 10).count 4 = 2

def count_numbers_with_two_fours (start finish : ℕ) : ℕ :=
  (List.range (finish - start + 1)).map (· + start)
    |>.filter has_two_fours
    |>.length

theorem unique_two_fours_up_to_243 :
  count_numbers_with_two_fours 10 243 = 1 ∧
  ∀ m > 243, count_numbers_with_two_fours 10 m > 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_two_fours_up_to_243_l824_82496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_county_count_l824_82418

/-- Represents the types of inhabitants in the fairy tale island. -/
inductive Inhabitant
  | Elf
  | Dwarf
  | Centaur

/-- Represents a county on the fairy tale island. -/
structure County where
  inhabitant : Inhabitant

/-- Represents the state of the fairy tale island. -/
structure IslandState where
  counties : List County

/-- Applies the division rule for a given year. -/
def applyDivisionRule (state : IslandState) (year : Nat) : IslandState :=
  match year with
  | 1 => { counties := state.counties.bind (fun c => 
      match c.inhabitant with
      | Inhabitant.Elf => [c]
      | _ => [c, c, c]) }
  | 2 => { counties := state.counties.bind (fun c => 
      match c.inhabitant with
      | Inhabitant.Dwarf => [c]
      | _ => [c, c, c, c]) }
  | 3 => { counties := state.counties.bind (fun c => 
      match c.inhabitant with
      | Inhabitant.Centaur => [c]
      | _ => [c, c, c, c, c, c]) }
  | _ => state

/-- The initial state of the island. -/
def initialState : IslandState :=
  { counties := [
    { inhabitant := Inhabitant.Elf },
    { inhabitant := Inhabitant.Dwarf },
    { inhabitant := Inhabitant.Centaur }
  ] }

/-- The final state after applying all division rules. -/
def finalState : IslandState :=
  applyDivisionRule (applyDivisionRule (applyDivisionRule initialState 1) 2) 3

/-- The theorem stating that the final number of counties is 54. -/
theorem final_county_count : finalState.counties.length = 54 := by
  sorry

#eval finalState.counties.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_county_count_l824_82418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_one_meter_apart_l824_82412

-- Define the color type
inductive Color
  | Red
  | Black

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Theorem statement
theorem same_color_one_meter_apart :
  ∃ (p1 p2 : Point), coloring p1 = coloring p2 ∧ distance p1 p2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_one_meter_apart_l824_82412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_acceleration_l824_82450

variable (k m y g : ℝ)
variable (hk : k > 0)
variable (hm : m > 0)
variable (hg : g > 0)

/-- Acceleration of the ball -/
noncomputable def acceleration (k m y g : ℝ) : ℝ := -(k/m)*y - g

/-- Theorem stating the acceleration of the ball -/
theorem ball_acceleration (k m y g : ℝ) :
  acceleration k m y g = -(k/m)*y - g :=
by
  -- Unfold the definition of acceleration
  unfold acceleration
  -- The equality holds by definition
  rfl

-- Example usage of the theorem
example (k m y g : ℝ) (hk : k > 0) (hm : m > 0) (hg : g > 0) :
  acceleration k m y g = -(k/m)*y - g :=
ball_acceleration k m y g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_acceleration_l824_82450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l824_82464

def M : Finset Nat := {1, 2}
def N : Finset Nat := {2, 3}

theorem proper_subsets_count : 
  Finset.card (Finset.powerset (M ∪ N) \ {M ∪ N}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l824_82464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_eq_one_solutions_l824_82417

theorem sin_plus_cos_eq_one_solutions (x : ℝ) :
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0 ∨ x = Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_eq_one_solutions_l824_82417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_rate_equation_l824_82419

/-- Represents the average annual growth rate of per capita disposable income -/
def x : Real := sorry

/-- Initial per capita disposable income in 2020 (in ten thousands of yuan) -/
def initial_income : Real := 3.2

/-- Final per capita disposable income in 2022 (in ten thousands of yuan) -/
def final_income : Real := 3.7

/-- Time period in years -/
def years : Nat := 2

/-- Theorem stating that the equation correctly represents the average annual growth rate -/
theorem growth_rate_equation :
  initial_income * (1 + x) ^ years = final_income := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_rate_equation_l824_82419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zeros_l824_82493

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + x + 1

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (f x)^2 - 2*(f x) - 3

-- Theorem statement
theorem g_zeros :
  (∃ a b : ℝ, a ≠ b ∧ g a = 0 ∧ g b = 0) ∧
  (∃ c : ℝ, c > 1 ∧ c < 2 ∧ g c = 0) ∧
  (∀ x y z : ℝ, g x = 0 ∧ g y = 0 ∧ g z = 0 → x = y ∨ x = z ∨ y = z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zeros_l824_82493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l824_82415

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : 2 * t.b * Real.sin t.A = Real.sqrt 3 * t.a)
  (h3 : t.a + t.c = 5)
  (h4 : t.a > t.c)
  (h5 : t.b = Real.sqrt 7) :
  t.B = Real.pi/3 ∧ Real.cos (2 * t.A + t.B) = -11/14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l824_82415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_25_terms_ap_l824_82497

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  a : ℚ  -- first term
  d : ℚ  -- common difference

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  ap.a + (n - 1 : ℚ) * ap.d

/-- The sum of the first n terms of an arithmetic progression -/
def sumFirstNTerms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

/-- Theorem: For an arithmetic progression where the sum of the 6th and 18th terms is 16,
    the sum of the first 25 terms is equal to 200 + 25d, where d is the common difference. -/
theorem sum_25_terms_ap (ap : ArithmeticProgression) 
    (h : nthTerm ap 6 + nthTerm ap 18 = 16) : 
    sumFirstNTerms ap 25 = 200 + 25 * ap.d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_25_terms_ap_l824_82497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_sold_afternoon_eq_40_l824_82465

/-- Calculates the number of oranges sold in the afternoon given the sales data --/
def oranges_sold_afternoon (apple_price : ℚ) (orange_price : ℚ)
  (morning_apples : ℕ) (morning_oranges : ℕ)
  (afternoon_apples : ℕ) (total_sales : ℚ) : ℕ :=
  let morning_sales := apple_price * morning_apples + orange_price * morning_oranges
  let afternoon_apple_sales := apple_price * afternoon_apples
  let afternoon_orange_sales := total_sales - morning_sales - afternoon_apple_sales
  (afternoon_orange_sales / orange_price).floor.toNat

/-- Theorem stating the number of oranges sold in the afternoon --/
theorem oranges_sold_afternoon_eq_40 :
  oranges_sold_afternoon (3/2) 1 40 30 50 205 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_sold_afternoon_eq_40_l824_82465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_extension_l824_82424

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then (x + 1)^2 - 1 else (x - 3)^2 - 1

-- State the theorem
theorem symmetric_function_extension :
  (∀ x : ℝ, f (2 - x) = f x) ∧  -- Symmetry about x = 1
  (∀ x : ℝ, x ≤ 1 → f x = (x + 1)^2 - 1) →  -- Definition for x ≤ 1
  (∀ x : ℝ, x > 1 → f x = (x - 3)^2 - 1) :=  -- Conclusion for x > 1
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_extension_l824_82424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_bound_l824_82413

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The product of the first n prime numbers -/
def primeProduct (n : ℕ) : ℕ := (List.range n).map nthPrime |>.prod

/-- Theorem: The product of the first k prime numbers is less than or equal to 4^(kth prime number) -/
theorem prime_product_bound (k : ℕ) : primeProduct k ≤ 4^(nthPrime k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_bound_l824_82413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rd_apl_ratio_approx_l824_82436

/-- Calculates the ratio of R&D expenses to the increase in average labor productivity -/
noncomputable def rd_apl_ratio (rd_expenses : ℝ) (apl_increase : ℝ) : ℝ :=
  rd_expenses / apl_increase

/-- Theorem stating that the ratio of R&D expenses to the increase in average labor productivity
    is approximately 3260 million rubles -/
theorem rd_apl_ratio_approx :
  let rd_expenses : ℝ := 2640.92
  let apl_increase : ℝ := 0.81
  abs (rd_apl_ratio rd_expenses apl_increase - 3260) < 1 := by
  sorry

/-- Compute an approximation of the ratio using Float -/
def rd_apl_ratio_float (rd_expenses : Float) (apl_increase : Float) : Float :=
  rd_expenses / apl_increase

#eval Float.round (rd_apl_ratio_float 2640.92 0.81)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rd_apl_ratio_approx_l824_82436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_C_height_l824_82454

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem tank_C_height (h_C : ℝ) :
  let r_C := (8 : ℝ) / (2 * Real.pi)
  let r_B := (10 : ℝ) / (2 * Real.pi)
  let h_B := (8 : ℝ)
  cylinderVolume r_C h_C = 0.8 * cylinderVolume r_B h_B →
  h_C = 10 := by
  sorry

#check tank_C_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_C_height_l824_82454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_characterization_l824_82428

/-- The quadratic equation in x with parameter m -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 2*(m+2)*x + m^2 - 1 = 0

/-- Predicate for the equation having two positive real roots -/
def has_two_positive_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m

/-- Predicate for the equation having one positive and one negative real root -/
def has_one_positive_one_negative_root (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m

theorem quadratic_roots_characterization :
  (∀ m : ℝ, has_two_positive_roots m ↔ (m ∈ Set.Icc (-5/4) (-1) ∪ Set.Ioi 1)) ∧
  (∀ m : ℝ, has_one_positive_one_negative_root m ↔ m ∈ Set.Ioo (-1) 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_characterization_l824_82428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l824_82499

theorem teacher_age (num_students : Nat) (student_avg_age : Nat) (new_avg_age : Nat) (teacher_age : Nat) :
  num_students = 40 →
  student_avg_age = 15 →
  new_avg_age = 16 →
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = new_avg_age →
  teacher_age = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l824_82499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_200_zeros_count_l824_82409

def g₀ (x : ℝ) : ℝ := x + |x - 150| - |x + 150|

def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

theorem g_200_zeros_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, g 200 x = 0) ∧ (∀ x ∉ S, g 200 x ≠ 0) ∧ S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_200_zeros_count_l824_82409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_green_apples_l824_82430

/-- The probability of choosing 2 green apples out of 9 total apples, where 4 are green, is 1/6 -/
theorem probability_two_green_apples (total : ℕ) (green : ℕ) (chosen : ℕ) :
  total = 9 →
  green = 4 →
  chosen = 2 →
  (Nat.choose green chosen : ℚ) / (Nat.choose total chosen : ℚ) = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_green_apples_l824_82430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_dividing_factorial_l824_82472

theorem largest_power_dividing_factorial : ∃ (n : ℕ), n = 7 ∧ 
  (∀ (m : ℕ), (30 ^ m : ℕ) ∣ Nat.factorial 30 → m ≤ n) ∧
  (30 ^ n : ℕ) ∣ Nat.factorial 30 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_dividing_factorial_l824_82472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_square_of_rational_b_converges_to_two_thirds_l824_82445

noncomputable def A (a b : ℝ) : ℝ := (a + b) / 2

noncomputable def G (a b : ℝ) : ℝ := Real.sqrt (a * b)

noncomputable def a : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 2) => A (A (a n) (a (n + 1))) (G (a n) (a (n + 1)))

theorem a_is_square_of_rational (n : ℕ) :
  ∃ (b : ℚ), b ≥ 0 ∧ a n = (b : ℝ) ^ 2 := by sorry

theorem b_converges_to_two_thirds (n : ℕ) (hn : n > 0) :
  ∃ (b : ℚ), b ≥ 0 ∧ a n = (b : ℝ) ^ 2 ∧ |b - 2/3| < 1 / (2 ^ n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_square_of_rational_b_converges_to_two_thirds_l824_82445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mixture_ratio_l824_82433

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol : ℚ
  water : ℚ

/-- Calculates the ratio of alcohol to water in a mixture -/
def ratio (m : Mixture) : ℚ :=
  m.alcohol / m.water

theorem new_mixture_ratio (initial : Mixture) (added_water : ℚ) :
  initial.alcohol = 10 ∧ 
  ratio initial = 4 / 3 ∧ 
  added_water = 5 →
  ratio { alcohol := initial.alcohol, water := initial.water + added_water } = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mixture_ratio_l824_82433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_f_cosine_equivalence_original_conditions_l824_82467

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem f_monotonic_increase (k : ℤ) :
  let a := k * Real.pi - Real.pi / 12
  let b := k * Real.pi + 5 * Real.pi / 12
  (∀ x ∈ Set.Icc a b, StrictMono f) ∧
  (∀ x y, x < a ∨ y > b → ¬StrictMono f) :=
by sorry

theorem f_cosine_equivalence (x : ℝ) :
  f (x / 2 + 5 * Real.pi / 12) = Real.cos x :=
by sorry

theorem original_conditions :
  (∃ ω φ, ω > 0 ∧ -Real.pi / 2 ≤ φ ∧ φ < Real.pi / 2 ∧
   ∀ x, f x = Real.sin (ω * x + φ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_f_cosine_equivalence_original_conditions_l824_82467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_shape_l824_82400

/-- The area of the enclosed shape formed by y = x^3 and y = x -/
noncomputable def enclosed_area : ℝ := 1/2

/-- The curve function y = x^3 -/
def curve (x : ℝ) : ℝ := x^3

/-- The line function y = x -/
def line (x : ℝ) : ℝ := x

/-- Theorem: The area of the enclosed shape formed by y = x^3 and y = x is 1/2 -/
theorem area_of_enclosed_shape :
  enclosed_area = ∫ x in Set.Icc 0 1, 2 * (line x - curve x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_shape_l824_82400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_lt_b_7_l824_82427

def b (α : ℕ → ℕ) : ℕ → ℚ
  | 0 => 1 + 1 / α 1  -- Add this case for Nat.zero
  | 1 => 1 + 1 / α 1
  | n + 1 => 1 + 1 / (α 1 + b α n)

theorem b_4_lt_b_7 (α : ℕ → ℕ) (h : ∀ k, α k ≥ 1) : b α 4 < b α 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_lt_b_7_l824_82427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l824_82451

-- Define the function f(x) = ln x + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 3

-- Theorem statement
theorem root_exists_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l824_82451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_sqrt_radii_equals_105_52_l824_82431

/-- Represents a circle in the upper half-plane tangent to the x-axis -/
structure Circle where
  radius : ℝ
  center_x : ℝ

/-- Constructs a new circle externally tangent to two given circles -/
noncomputable def new_circle (c1 c2 : Circle) : Circle :=
  { radius := (c1.radius * c2.radius) / ((c1.radius.sqrt + c2.radius.sqrt) ^ 2),
    center_x := 0 }  -- Center_x is not used in the calculation, so we set it to 0

/-- Generates all circles in layers L₀ to L₅ -/
noncomputable def generate_circles : List Circle :=
  sorry  -- Implementation details omitted

/-- The set S of all circles from L₀ to L₅ -/
noncomputable def S : Finset Circle :=
  sorry  -- Convert generate_circles to a Finset

/-- The sum of 1/√(r(C)) for all circles C in S -/
noncomputable def sum_inverse_sqrt_radii : ℝ :=
  S.sum (λ c => 1 / c.radius.sqrt)

theorem sum_inverse_sqrt_radii_equals_105_52 :
  sum_inverse_sqrt_radii = 105 / 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_sqrt_radii_equals_105_52_l824_82431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l824_82495

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - 1)^2 else x + 1/x

-- Theorem statement
theorem f_composition_negative_one : f (f (-1)) = 17/4 := by
  -- Evaluate f(-1)
  have h1 : f (-1) = 4 := by
    simp [f]
    norm_num
  
  -- Evaluate f(4)
  have h2 : f 4 = 17/4 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc f (f (-1))
    = f 4 := by rw [h1]
    _ = 17/4 := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l824_82495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_at_smallest_x_l824_82470

/-- The function g(x) defined as sin(x/5) + sin(x/7) -/
noncomputable def g (x : ℝ) : ℝ := Real.sin (x / 5) + Real.sin (x / 7)

/-- The smallest positive value of x in degrees where g(x) achieves its maximum -/
def smallest_max_x : ℝ := 5850

/-- Theorem stating that g(x) achieves its maximum at smallest_max_x -/
theorem g_max_at_smallest_x :
  ∀ x : ℝ, x > 0 → g x ≤ g smallest_max_x ∧
  (∀ y : ℝ, 0 < y ∧ y < smallest_max_x → g y < g smallest_max_x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_at_smallest_x_l824_82470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_speed_l824_82473

/-- Proves that the speed of the first part of the trip is 60 km/h given the conditions -/
theorem first_part_speed (total_distance : ℝ) (first_part_distance : ℝ) (second_part_speed : ℝ) (average_speed : ℝ) (v : ℝ) :
  total_distance = 60 →
  first_part_distance = 30 →
  second_part_speed = 30 →
  average_speed = 40 →
  (total_distance / ((first_part_distance / v) + (total_distance - first_part_distance) / second_part_speed) = average_speed) →
  v = 60 := by
  intros h1 h2 h3 h4 h5
  -- The proof steps would go here
  sorry

#check first_part_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_speed_l824_82473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_point_l824_82490

/-- The slope of the tangent line to the circle x^2 + y^2 = 4 at the point (1, √3) is -√3/3 -/
theorem tangent_slope_at_point (x y : ℝ) (h1 : x^2 + y^2 = 4) (h2 : x = 1) (h3 : y = Real.sqrt 3) :
  -x / y = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_point_l824_82490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loom_weaving_rate_l824_82498

/-- 
Given a loom that weaves 52 meters of cloth in 45.6140350877193 seconds,
prove that it weaves approximately 1.13953488372093 meters of cloth per second.
-/
theorem loom_weaving_rate : 
  let total_cloth : ℝ := 52
  let total_time : ℝ := 45.6140350877193
  let cloth_per_second : ℝ := total_cloth / total_time
  abs (cloth_per_second - 1.13953488372093) < 0.00000000000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loom_weaving_rate_l824_82498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_second_side_l824_82403

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapeziumArea (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem: The second parallel side of the trapezium is 18 cm -/
theorem trapezium_second_side (t : Trapezium) 
    (h1 : t.side1 = 20)
    (h2 : t.height = 25)
    (h3 : t.area = 475)
    (h4 : t.area = trapeziumArea t) : 
  t.side2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_second_side_l824_82403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implications_l824_82437

theorem set_equality_implications (a b : ℝ) 
  (h : Set.toFinset {a, b, 1} = Set.toFinset {a^2, a+b, 0}) : 
  b = 0 ∧ a = -1 ∧ a^2023 + b^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implications_l824_82437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l824_82407

-- Define the function f
noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- State the theorem
theorem function_properties :
  ∀ (A ω φ : ℝ),
    A > 0 →
    0 < ω →
    ω < 16 →
    0 < φ →
    φ < Real.pi / 2 →
    (∀ x, f A ω φ x ≤ Real.sqrt 2) →
    f A ω φ 0 = 1 →
    f A ω φ (Real.pi / 8) = Real.sqrt 2 →
    (∃ k : ℤ, ∀ x, f A ω φ x = f A ω φ (Real.pi / 4 - x + k * Real.pi)) ∧
    ω ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l824_82407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l824_82423

open BigOperators

def S : ℚ := ∑ k in Finset.range 502, (2013 : ℚ) / ((2 + 4*k) * (6 + 4*k))

theorem sum_remainder : 
  (Int.floor (2 * S + 1/2) % 2 = 0) ∧ 
  (Int.floor (2 * S + 1/2) % 5 = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l824_82423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_l824_82404

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem diagonal_length (q : Quadrilateral) :
  is_convex q →
  side_length q.A q.B + side_length q.B q.C = 2021 →
  side_length q.A q.D = side_length q.C q.D →
  angle q.A q.B q.C = π/2 →
  angle q.C q.D q.A = π/2 →
  side_length q.B q.D = 2021 / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_l824_82404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l824_82474

noncomputable def my_sequence (n : ℕ+) : ℝ := 1 / Real.sqrt n.val

theorem sequence_formula (n : ℕ+) : my_sequence n = 1 / Real.sqrt n.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l824_82474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isabellas_fifth_score_l824_82461

/-- Represents Isabella's test scores -/
def IsabellaScores := List Int

/-- Check if all elements in a list are unique -/
def all_unique (l : List Int) : Prop :=
  l.Nodup

/-- Check if all elements in a list are within a given range -/
def all_in_range (l : List Int) (lower upper : Int) : Prop :=
  ∀ x ∈ l, lower ≤ x ∧ x ≤ upper

/-- Check if the average of a list of integers is an integer -/
def integer_average (l : List Int) : Prop :=
  (l.sum % l.length) = 0

/-- Main theorem representing Isabella's test scores problem -/
theorem isabellas_fifth_score (scores : IsabellaScores) : 
  scores.length = 8 ∧ 
  all_unique scores ∧
  all_in_range scores 91 102 ∧
  (∀ k ≤ 8, integer_average (scores.take k)) ∧
  scores.get? 7 = some 97 →
  scores.get? 4 = some 95 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isabellas_fifth_score_l824_82461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_theorem_l824_82452

/-- Represents the dimensions of the rectangular hutch -/
structure HutchDimensions where
  length : ℝ
  width : ℝ

/-- Represents the tying point of the rope -/
inductive TyingPoint
  | MidpointLongSide
  | NearCornerShortSide

/-- Calculates the area accessible to the rabbit given the rope length and tying point -/
noncomputable def accessibleArea (ropeLength : ℝ) (hutch : HutchDimensions) (tyingPoint : TyingPoint) : ℝ :=
  match tyingPoint with
  | TyingPoint.MidpointLongSide => Real.pi * ropeLength^2 / 2
  | TyingPoint.NearCornerShortSide => 3 * Real.pi * ropeLength^2 / 4 - Real.pi * 3^2 / 4

/-- Theorem stating the difference in accessible areas -/
theorem area_difference_theorem (ropeLength : ℝ) (hutch : HutchDimensions) :
  accessibleArea ropeLength hutch TyingPoint.NearCornerShortSide -
  accessibleArea ropeLength hutch TyingPoint.MidpointLongSide =
  22.75 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_theorem_l824_82452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l824_82402

/-- The distance between two parallel lines -/
theorem parallel_lines_distance (m : ℝ) : 
  (∀ x y, 3 * x + y - 3 = 0 ↔ 6 * x + m * y + 4 = 0) →  -- Lines are parallel
  (abs (-6 - 4) / Real.sqrt (6^2 + m^2) = Real.sqrt 10 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l824_82402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_429_sequence_l824_82405

def has_429_sequence (m n : ℕ) : Prop :=
  ∃ k : ℕ, (1000 * (10^k * m) / n) % 1000 = 429

theorem smallest_n_with_429_sequence :
  ∃ m : ℕ, m < 43 ∧ Nat.Coprime m 43 ∧ has_429_sequence m 43 ∧
  ∀ n : ℕ, n < 43 → ¬∃ m : ℕ, m < n ∧ Nat.Coprime m n ∧ has_429_sequence m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_429_sequence_l824_82405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_solution_count_l824_82469

noncomputable def f (a x : ℝ) : ℝ := |Real.exp x - a| + |1 / Real.exp x - 1|

theorem f_composition_solution_count (a : ℝ) (h : a ≥ 4/3) :
  (∃ x : ℝ, f a (f a x) = 1/4) ↔ a = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_solution_count_l824_82469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_inequality_tangent_line_perpendicular_l824_82475

noncomputable section

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x^2 + b * (Real.log x - x)
def g (b x : ℝ) : ℝ := -1/2 * x^2 + (1 - b) * x

-- State the theorem
theorem range_of_m_for_inequality (b : ℝ) (hb : b > 1) :
  (∀ m : ℝ, m ≤ -1 → 
    ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 b ∧ x₂ ∈ Set.Icc 1 b ∧ 
      f (-1/2) b x₁ - f (-1/2) b x₂ - 1 > g b x₁ - g b x₂ + m) ∧
  (∀ m : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 b ∧ x₂ ∈ Set.Icc 1 b →
    f (-1/2) b x₁ - f (-1/2) b x₂ - 1 > g b x₁ - g b x₂ + m) → m ≤ -1) :=
by sorry

-- Additional hypothesis about the tangent line
theorem tangent_line_perpendicular (b : ℝ) :
  let f' := fun x => 2 * (-1/2) * x + b * (1/x - 1)
  f' 1 = -1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_inequality_tangent_line_perpendicular_l824_82475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l824_82410

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x : ℤ | (x : ℝ)^2 - (x : ℝ) - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {(-2 : ℤ)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l824_82410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l824_82416

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h : ∀ x : ℝ, f x φ ≤ |f (π/6) φ|) : 
  (f (11*π/12) φ = 0) ∧ 
  (|f (7*π/12) φ| < |f (π/5) φ|) ∧ 
  (¬(∀ x : ℝ, f x φ = f (-x) φ) ∧ ¬(∀ x : ℝ, f x φ = -f (-x) φ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l824_82416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_parallel_to_c_l824_82401

/-- Given vectors a and b, prove that their sum is parallel to c -/
theorem sum_parallel_to_c (x : ℝ) : 
  ∃ (k : ℝ), (![x, 1] : Fin 2 → ℝ) + (![- x, x^2] : Fin 2 → ℝ) = k • (![0, 1] : Fin 2 → ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_parallel_to_c_l824_82401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l824_82494

/-- Given a real number m > 0, define the function f(x) = x + m/x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m / x

/-- Theorem stating the properties of the function f -/
theorem f_properties (m : ℝ) (h_m : m > 0) :
  /- 1. f is an odd function -/
  (∀ x, x ≠ 0 → f m (-x) = -(f m x)) ∧
  /- 2. f is decreasing on the interval (0, √m) -/
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.sqrt m → f m x > f m y) ∧
  /- 3. If f is monotonically increasing on [2, +∞), then 0 < m ≤ 4 -/
  ((∀ x y, 2 ≤ x ∧ x < y → f m x ≤ f m y) → m ≤ 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l824_82494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_is_54_l824_82486

structure Point where
  x : Int
  y : Int

def polygon_vertices : List Point := [
  ⟨0, 0⟩, ⟨3, 0⟩, ⟨6, 0⟩, ⟨6, 3⟩, ⟨9, 3⟩, ⟨9, 6⟩,
  ⟨6, 6⟩, ⟨6, 9⟩, ⟨3, 9⟩, ⟨3, 6⟩, ⟨0, 6⟩, ⟨0, 3⟩
]

def polygon_area (vertices : List Point) : ℕ := sorry

theorem polygon_area_is_54 :
  polygon_area polygon_vertices = 54 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_is_54_l824_82486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_16_to_12th_power_l824_82422

theorem fourth_root_16_to_12th_power : (16 : ℝ) ^ ((1/4 : ℝ) * 12) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_16_to_12th_power_l824_82422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_BEDC_l824_82438

/-- Represents a parallelogram ABCD with a height AE -/
structure Parallelogram where
  area : ℝ
  cd : ℝ
  ae : ℝ
  ed : ℝ

/-- The area of region BEDC in a parallelogram -/
noncomputable def area_BEDC (p : Parallelogram) : ℝ := p.area - (1/2 * p.ae * p.ae)

/-- Theorem: In a parallelogram ABCD with area 150 and ED = 2/3 CD, 
    the area of region BEDC is 125 -/
theorem parallelogram_area_BEDC 
  (p : Parallelogram) 
  (h1 : p.area = 150) 
  (h2 : p.ed = 2/3 * p.cd) 
  (h3 : p.ae = 1/3 * p.cd) : 
  area_BEDC p = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_BEDC_l824_82438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_rational_function_l824_82432

open Filter Topology Real

/-- The limit of (3x^4 - 2x^3 + 6x^2 + x - 2) / (9x^4 + 5x^3 + 7x^2 + x + 1) as x approaches infinity is 1/3 -/
theorem limit_rational_function :
  Tendsto (fun x : ℝ => (3*x^4 - 2*x^3 + 6*x^2 + x - 2) / (9*x^4 + 5*x^3 + 7*x^2 + x + 1)) atTop (𝓝 (1/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_rational_function_l824_82432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_lunch_break_duration_l824_82488

/-- Represents the painting rate of a person or group in terms of building percentage per hour -/
structure PaintingRate where
  rate : ℝ

/-- Represents a workday with start time, end time, and percentage of building painted -/
structure Workday where
  startTime : ℝ  -- in hours from midnight
  endTime : ℝ    -- in hours from midnight
  percentPainted : ℝ

/-- The problem setup -/
structure PaintingProblem where
  anneRate : PaintingRate
  assistantsRate : PaintingRate
  monday : Workday
  tuesday : Workday
  wednesday : Workday

/-- The theorem to prove -/
theorem painting_lunch_break_duration 
  (prob : PaintingProblem)
  (h1 : prob.monday.startTime = 9)
  (h2 : prob.monday.endTime = 17)
  (h3 : prob.monday.percentPainted = 0.6)
  (h4 : prob.tuesday.startTime = 9)
  (h5 : prob.tuesday.endTime = 15)
  (h6 : prob.tuesday.percentPainted = 0.3)
  (h7 : prob.wednesday.startTime = 9)
  (h8 : prob.wednesday.endTime = 18)
  (h9 : prob.wednesday.percentPainted = 0.1)
  (h10 : prob.anneRate.rate + prob.assistantsRate.rate = 
         (prob.monday.percentPainted / (prob.monday.endTime - prob.monday.startTime)))
  (h11 : prob.assistantsRate.rate = 
         (prob.tuesday.percentPainted / (prob.tuesday.endTime - prob.tuesday.startTime)))
  (h12 : prob.anneRate.rate = 
         (prob.wednesday.percentPainted / (prob.wednesday.endTime - prob.wednesday.startTime)))
  : ∃ (lunchBreak : ℝ), lunchBreak = 213/60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_lunch_break_duration_l824_82488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_differentiation_operations_l824_82434

-- Define the derivative operation
def derivative (f : ℝ → ℝ) : ℝ → ℝ := sorry

-- State the theorem
theorem correct_differentiation_operations :
  (∀ x, derivative (λ y => Real.sin y) x = Real.cos x) ∧
  (∀ x, derivative (λ y => 2 * y^2 - 1) x = derivative (λ y => 2 * y^2) x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_differentiation_operations_l824_82434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_score_for_average_increase_l824_82456

def current_scores : List ℕ := [86, 92, 75, 68, 88, 84]

def average_increase : ℚ := 4

theorem minimum_score_for_average_increase :
  let current_sum : ℕ := current_scores.sum
  let current_count : ℕ := current_scores.length
  let current_average : ℚ := current_sum / current_count
  let target_average : ℚ := current_average + average_increase
  let min_score : ℕ := 
    (Int.ceil (target_average * (current_count + 1) - current_sum)).toNat
  min_score = 110 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_score_for_average_increase_l824_82456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l824_82458

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The sum of 46.913 and 58.27 rounded to the nearest hundredth is 105.18 -/
theorem sum_and_round :
  round_to_hundredth (46.913 + 58.27) = 105.18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l824_82458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_30_l824_82491

noncomputable def arithmeticSequence (a₁ : ℝ) (a₂ : ℝ) (n : ℕ) : ℝ := 
  a₁ + (n - 1 : ℝ) * (a₂ - a₁)

noncomputable def arithmeticSequenceSum (a₁ : ℝ) (a₂ : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a₁ + arithmeticSequence a₁ a₂ n)

theorem arithmetic_sequence_sum_30 :
  arithmeticSequenceSum 3 7 30 = 1830 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_30_l824_82491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pairs_for_profit_l824_82426

/-- Represents the purchase price of a pair of shoes in yuan -/
def purchase_price : ℕ := 150

/-- Represents the number of pairs sold in a transaction -/
def pairs_sold : ℕ := 8

/-- The purchase price is an integer not exceeding 200 yuan -/
axiom purchase_price_bound : purchase_price ≤ 200

/-- The selling price is 150% of the purchase price with a 10% discount -/
def selling_price : ℚ := (3 / 2) * purchase_price * (9 / 10)

/-- There's a 60 yuan return if the transaction amount exceeds 1000 yuan -/
def return_amount : ℚ := if selling_price * pairs_sold ≥ 1000 then 60 else 0

/-- The profit per pair is 45 yuan -/
def profit_per_pair : ℚ := 45

/-- The total profit for the transaction -/
def total_profit : ℚ := selling_price * pairs_sold - return_amount - purchase_price * pairs_sold

/-- Theorem stating that 8 is the minimum number of pairs to be sold to achieve the desired profit -/
theorem min_pairs_for_profit : 
  (∀ n : ℕ, n < 8 → total_profit / ↑n < profit_per_pair) ∧
  (total_profit / 8 = profit_per_pair) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pairs_for_profit_l824_82426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_l824_82443

-- Define the types for Plane and Line
def Plane := Type
def Line := Type

-- Define the Perpendicular relation
def Perpendicular : Type → Type → Prop := sorry

-- Define the Space structure
structure Space :=
  (α β : Plane)
  (m n : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_not_in_planes : ¬ (m = α) ∧ ¬ (m = β))
  (h_n_not_in_planes : ¬ (n = α) ∧ ¬ (n = β))

-- State the theorem
theorem perpendicular_planes (s : Space) 
  (h1 : Perpendicular s.m s.n)
  (h2 : Perpendicular s.n s.β)
  (h3 : Perpendicular s.m s.α) :
  Perpendicular s.α s.β :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_l824_82443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l824_82425

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem problem (N : ℤ) (x y : ℝ) 
  (eq1 : floor x + 2 * y = N + 2)
  (eq2 : floor y + 2 * x = 3 - N) :
  x = 3 / 2 - N := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l824_82425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l824_82477

/-- Helper function to calculate the area of a triangle given its side lengths -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given a triangle ABC with sides a, b, c, and circumradius R, 
    prove the following inequalities hold -/
theorem triangle_inequalities (a b c R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_circumradius : R = (a * b * c) / (4 * area_triangle a b c)) : 
  (a^2 + b^2 + c^2 ≤ 9 * R^2) ∧ 
  (a + b + c ≤ 3 * Real.sqrt 3 * R) ∧ 
  ((a * b * c)^(1/3 : ℝ) ≤ Real.sqrt 3 * R) ∧
  (a^2 + b^2 + c^2 = 9 * R^2 ↔ a = b ∧ b = c) ∧
  (a + b + c = 3 * Real.sqrt 3 * R ↔ a = b ∧ b = c) ∧
  ((a * b * c)^(1/3 : ℝ) = Real.sqrt 3 * R ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l824_82477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l824_82449

theorem trigonometric_problem (α β : ℝ) 
  (h1 : Real.cos (α + β) = 2 * Real.sqrt 5 / 5)
  (h2 : Real.tan β = 1 / 7)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : 0 < β ∧ β < Real.pi / 2) :
  (Real.cos (2 * β) + Real.sin (2 * β) - Real.sin β * Real.cos β = 11 / 10) ∧
  (2 * α + β = Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l824_82449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_overtake_time_l824_82480

/-- Represents a car with its initial position and speed -/
structure Car where
  position : ℚ
  speed : ℚ

/-- The time it takes for one car to overtake another -/
def overtakeTime (car1 car2 : Car) : ℚ :=
  (car2.position - car1.position) / (car1.speed - car2.speed)

/-- Theorem stating that Car A overtakes both Car B and Car C at the same time -/
theorem car_overtake_time (carA carB carC : Car)
  (hAB : carA.position = carB.position - 24)
  (hAC : carA.position = carC.position - 12)
  (hBC : carC.position = carB.position - 12)
  (hSpeedA : carA.speed = 58)
  (hSpeedB : carB.speed = 50)
  (hSpeedC : carC.speed = 54) :
  overtakeTime carA carB = overtakeTime carA carC ∧ overtakeTime carA carB = 3 := by
  sorry

#check car_overtake_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_overtake_time_l824_82480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_distance_less_than_one_l824_82455

open Real
open BigOperators
open Finset

theorem subset_sum_distance_less_than_one (n : ℕ) (x : ℝ) (xs : Fin n → ℝ) :
  ∃ (subsets : Finset (Finset (Fin n))),
    subsets.card = Nat.choose n (n / 2) ∧
    ∀ s ∈ subsets, |x - ∑ i in s, xs i| < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_distance_less_than_one_l824_82455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_complement_l824_82466

noncomputable def f (x : ℝ) := Real.sqrt (2 - x)

def M : Set ℝ := {x : ℝ | x ≤ 2}

theorem domain_complement :
  (Set.univ : Set ℝ) \ M = Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_complement_l824_82466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_80_l824_82463

/-- The speed of the car in km/h -/
noncomputable def car_speed : ℝ := 80

/-- The speed of the train in km/h -/
noncomputable def train_speed : ℝ := 1.5 * car_speed

/-- The distance between point A and point B in km -/
noncomputable def distance : ℝ := 75

/-- The time lost by the train due to stops, in hours -/
noncomputable def train_stop_time : ℝ := 12.5 / 60

/-- The time taken by the car to travel from A to B in hours -/
noncomputable def car_travel_time : ℝ := distance / car_speed

/-- The time taken by the train to travel from A to B in hours -/
noncomputable def train_travel_time : ℝ := car_travel_time - train_stop_time

theorem car_speed_is_80 :
  train_speed = 1.5 * car_speed ∧
  distance = car_speed * car_travel_time ∧
  distance = train_speed * train_travel_time ∧
  car_speed = 80 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_80_l824_82463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_5_and_10_l824_82411

/-- A normal distribution with mean μ and standard deviation σ -/
structure NormalDistribution (μ σ : ℝ) where
  mean : ℝ := μ
  std_dev : ℝ := σ
  std_dev_pos : σ > 0

variable {μ σ : ℝ}

/-- The probability that a value from a normal distribution falls within one standard deviation of the mean -/
noncomputable def prob_within_one_std_dev (X : NormalDistribution μ σ) : ℝ := 0.6826

/-- The probability that a value from a normal distribution falls within two standard deviations of the mean -/
noncomputable def prob_within_two_std_dev (X : NormalDistribution μ σ) : ℝ := 0.9544

/-- The probability that a value from the normal distribution N(0, 5²) falls between 5 and 10 -/
theorem prob_between_5_and_10 (X : NormalDistribution 0 5) :
  (prob_within_two_std_dev X - prob_within_one_std_dev X) / 2 = 0.1359 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_5_and_10_l824_82411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_21952000_l824_82429

theorem cube_root_of_21952000 : (21952000 : ℝ) ^ (1/3) = 280 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_21952000_l824_82429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_pyramid_VEFGH_l824_82457

/-- Represents a pyramid with a rectangular base -/
structure RectangularBasePyramid where
  /-- Length of one side of the rectangular base -/
  baseLength : ℝ
  /-- Width of the rectangular base -/
  baseWidth : ℝ
  /-- Height of the pyramid -/
  height : ℝ

/-- Calculates the volume of a pyramid with a rectangular base -/
noncomputable def pyramidVolume (p : RectangularBasePyramid) : ℝ :=
  (1 / 3) * p.baseLength * p.baseWidth * p.height

/-- The specific pyramid VEFGH described in the problem -/
def pyramidVEFGH : RectangularBasePyramid where
  baseLength := 10
  baseWidth := 5
  height := 8

/-- Theorem stating that the volume of pyramid VEFGH is 133 1/3 -/
theorem volume_of_pyramid_VEFGH : 
  pyramidVolume pyramidVEFGH = 400 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_pyramid_VEFGH_l824_82457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l824_82478

noncomputable def f (x : ℝ) : ℝ := (Real.cos (x + Real.pi/4))^2 - (Real.sin (x + Real.pi/4))^2

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is an odd function
  (∀ x, f (x + Real.pi) = f x) ∧  -- f has period π
  (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ Real.pi)  -- π is the least positive period
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l824_82478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_eq_a_l824_82414

/-- The sequence a_n defined recursively -/
noncomputable def a : ℕ → ℝ → ℝ
  | 0, _ => 0
  | n + 1, k => k * a n k + Real.sqrt ((k^2 - 1) * (a n k)^2 + 1)

/-- The closed form expression for a_n -/
noncomputable def a_closed (n : ℕ) (k : ℝ) : ℝ :=
  (1 / (2 * Real.sqrt (k^2 - 1))) * ((k + Real.sqrt (k^2 - 1))^n - (k - Real.sqrt (k^2 - 1))^n)

/-- Theorem stating that the closed form expression is equal to the recursive definition -/
theorem a_closed_eq_a (n : ℕ) (k : ℝ) (h : k ≥ 2) : a n k = a_closed n k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_eq_a_l824_82414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_or_six_count_divisible_by_four_or_six_l824_82487

theorem divisible_by_four_or_six (n : ℕ) : 
  (Finset.filter (λ x : ℕ => x % 4 = 0 ∨ x % 6 = 0) (Finset.range n)).card = 
    (n / 4) + (n / 6) - (n / 12) :=
sorry

theorem count_divisible_by_four_or_six : 
  (Finset.filter (λ x : ℕ => x % 4 = 0 ∨ x % 6 = 0) (Finset.range 61)).card = 20 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_or_six_count_divisible_by_four_or_six_l824_82487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterclockwise_rotation_60_degrees_l824_82439

noncomputable def initial_vector : ℝ × ℝ := (1, 1)
noncomputable def transformed_vector : ℝ × ℝ := ((1 - Real.sqrt 3) / 2, (1 + Real.sqrt 3) / 2)
noncomputable def rotation_angle : ℝ := Real.pi / 3  -- 60° in radians

theorem counterclockwise_rotation_60_degrees (v : ℝ × ℝ) :
  let rotated_v := (v.1 * Real.cos rotation_angle - v.2 * Real.sin rotation_angle,
                    v.1 * Real.sin rotation_angle + v.2 * Real.cos rotation_angle)
  rotated_v = transformed_vector := by
  sorry

#check counterclockwise_rotation_60_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterclockwise_rotation_60_degrees_l824_82439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l824_82447

open Real

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 6)

-- State the theorem
theorem function_properties (ω : ℝ) 
  (h_monotone : ∀ x₁ x₂, x₁ ∈ Set.Ioo 0 (π / 5) → x₂ ∈ Set.Ioo 0 (π / 5) → x₁ < x₂ → f ω x₁ < f ω x₂) :
  (0 < ω ∧ ω ≤ 5 / 3) ∧ 
  (f ω (3 * π / 10) > 1 / 2) ∧
  (f ω π < 0 → ∃! x, x ∈ Set.Ioo 0 (101 * π / 100) ∧ f ω x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l824_82447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unexpressible_integer_l824_82459

/-- Given pairwise coprime positive integers a, b, and c, 
    2abc - ab - bc - ca is the maximum integer that cannot be 
    expressed as xbc + yca + zab for non-negative integers x, y, z -/
theorem max_unexpressible_integer (a b c : ℕ) 
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hab : Nat.Coprime a b) (hbc : Nat.Coprime b c) (hca : Nat.Coprime c a) :
    ∀ n : ℤ, (∃ (x y z : ℕ), n = ↑(x * b * c + y * c * a + z * a * b)) ↔ 
    (↑(2 * a * b * c - a * b - b * c - c * a) < n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unexpressible_integer_l824_82459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_divisibility_l824_82442

theorem series_divisibility (n : ℕ+) : ∃ k : ℤ, (3^(5*n.val) - 1) / 2 = 121 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_divisibility_l824_82442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_neg_one_l824_82441

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | (n + 1) => (1 + sequence_a n) / (1 - sequence_a n)

theorem a_2017_equals_neg_one : sequence_a 2017 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_neg_one_l824_82441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l824_82446

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  (Complex.abs ((2 + 5*Complex.I)*z^4 - z^3)) ≤ 135 * Real.sqrt 10 ∧
  ∃ w : ℂ, Complex.abs w = 3 ∧ Complex.abs ((2 + 5*Complex.I)*w^4 - w^3) = 135 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l824_82446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l824_82482

theorem tan_double_angle (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.cos x = 3/5) : 
  Real.tan (2 * x) = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l824_82482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AP_parallel_QB_l824_82420

-- Define the circles and points
variable (α β : Set (ℝ × ℝ))
variable (X A P B Q : ℝ × ℝ)

-- Define the conditions
def circles_touch_externally (α β : Set (ℝ × ℝ)) (X : ℝ × ℝ) : Prop := 
  X ∈ α ∧ X ∈ β ∧ (∀ Y : ℝ × ℝ, Y ≠ X → Y ∉ α ∨ Y ∉ β)

def A_P_on_alpha (α : Set (ℝ × ℝ)) (X A P : ℝ × ℝ) : Prop := 
  A ∈ α ∧ P ∈ α ∧ A ≠ X ∧ P ≠ X ∧ A ≠ P

def B_on_beta (β : Set (ℝ × ℝ)) (X A B : ℝ × ℝ) : Prop := 
  B ∈ β ∧ (∃ t : ℝ, 0 < t ∧ t < 1 ∧ B = (1 - t) • X + t • A)

def Q_on_beta (β : Set (ℝ × ℝ)) (X P Q : ℝ × ℝ) : Prop := 
  Q ∈ β ∧ (∃ s : ℝ, 0 < s ∧ s < 1 ∧ Q = (1 - s) • X + s • P)

-- Define parallel lines
def parallel (AB CD : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ (CD.1 - AB.1, CD.2 - AB.2) = k • (AB.2 - CD.2, AB.1 - CD.1)

-- State the theorem
theorem AP_parallel_QB (h1 : circles_touch_externally α β X)
                       (h2 : A_P_on_alpha α X A P)
                       (h3 : B_on_beta β X A B)
                       (h4 : Q_on_beta β X P Q) :
  parallel (A - P) (Q - B) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AP_parallel_QB_l824_82420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_left_proof_l824_82462

noncomputable def initial_amount : ℝ := 480

noncomputable def food_expense : ℝ := initial_amount / 2

noncomputable def amount_after_food : ℝ := initial_amount - food_expense

noncomputable def glasses_expense : ℝ := amount_after_food / 3

noncomputable def final_amount : ℝ := amount_after_food - glasses_expense

theorem money_left_proof : final_amount = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_left_proof_l824_82462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_three_six_l824_82421

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := (a + b) / (a * b)

-- Theorem statement
theorem diamond_three_six : diamond 3 6 = 1/2 := by
  -- Unfold the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform numerical calculations
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_three_six_l824_82421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_equivalent_l824_82468

noncomputable def f (x : ℝ) : ℝ := 1 + 1 / x

noncomputable def g (y : ℝ) : ℝ := 1 + 1 / y

theorem f_g_equivalent : ∀ x : ℝ, x ≠ 0 → ∃ y : ℝ, y ≠ 0 ∧ f x = y ∧ g y = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_equivalent_l824_82468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_four_same_face_l824_82435

def number_of_favorable_outcomes (n : ℕ) : ℕ :=
  2 + 2 * (n.choose 1)

def total_number_of_outcomes (n : ℕ) : ℕ :=
  2^n

theorem probability_at_least_four_same_face (n : ℕ) (p : ℚ) : n = 5 → p = 3/8 →
  (number_of_favorable_outcomes n : ℚ) / (total_number_of_outcomes n : ℚ) = p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_four_same_face_l824_82435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_change_is_correct_l824_82453

-- Define the problem parameters
def num_sandwiches : ℕ := 3
def cost_per_sandwich : ℚ := 5
def discount_rate : ℚ := 1 / 10
def tax_rate : ℚ := 1 / 20
def payment : ℚ := 28

-- Define the function to calculate the change
def calculate_change : ℚ :=
  let total_cost := num_sandwiches * cost_per_sandwich
  let discounted_cost := total_cost * (1 - discount_rate)
  let tax_amount := discounted_cost * tax_rate
  let final_cost := discounted_cost + tax_amount
  payment - final_cost

-- Theorem to prove
theorem jack_change_is_correct :
  (calculate_change * 100).floor / 100 = 1382 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_change_is_correct_l824_82453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l824_82484

noncomputable def data_set : List ℝ := [5, 4, 4, 3, 6]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

theorem variance_of_data_set :
  mean data_set = 5 → variance data_set = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l824_82484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pa_squared_gt_pb_pc_l824_82489

/-- Triangle ABC with side lengths AB = 2√2, AC = √2, and BC = 2 -/
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)
  ab_length : dist A B = 2 * Real.sqrt 2
  ac_length : dist A C = Real.sqrt 2
  bc_length : dist B C = 2

/-- P is any point on side BC -/
def point_on_side (t : Triangle) (P : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ r : ℝ, 0 ≤ r ∧ r ≤ 1 ∧ P = (1 - r) • t.B + r • t.C

/-- The statement to be proved -/
theorem pa_squared_gt_pb_pc (t : Triangle) (P : EuclideanSpace ℝ (Fin 2)) 
    (h : point_on_side t P) : 
    (dist P t.A)^2 > (dist P t.B) * (dist P t.C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pa_squared_gt_pb_pc_l824_82489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_a_solution_l824_82440

def parallel_to_line (v : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k, 2*k)

theorem vector_a_solution :
  ∀ (a b : ℝ × ℝ),
  (‖a + 2 • b‖ = Real.sqrt 5) →
  parallel_to_line (a + 2 • b) →
  b = (2, -1) →
  (a = (-3, 4) ∨ a = (-5, 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_a_solution_l824_82440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_number_sum_theorem_l824_82448

theorem nine_number_sum_theorem (N : ℕ) (S : Finset ℝ) : 
  N ≥ 9 →
  S.card = N →
  (∀ x ∈ S, 0 ≤ x ∧ x < 1) →
  (∀ T ⊆ S, T.card = 8 → ∃ y ∈ S \ T, Int.floor (T.sum id + y) = T.sum id + y) →
  N = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_number_sum_theorem_l824_82448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l824_82479

/-- Given a circular sheet of paper with radius 8 cm cut into four congruent sectors,
    the height of the cone formed by rolling one sector until the edges meet is 2√15 cm. -/
theorem cone_height_from_circular_sector (r : ℝ) (h : r = 8) :
  let sector_arc_length := 2 * Real.pi * r / 4
  let base_radius := sector_arc_length / (2 * Real.pi)
  let slant_height := r
  Real.sqrt (slant_height ^ 2 - base_radius ^ 2) = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l824_82479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_line_l824_82485

/-- A parabola is tangent to a line if their equation has a double root -/
def is_tangent (a : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + 2 = 2 * x + 4 ∧ 
    ∀ y : ℝ, y ≠ x → a * y^2 + 2 ≠ 2 * y + 4

theorem parabola_tangent_to_line (a : ℝ) :
  is_tangent a → a = -1/2 := by
  sorry

#check parabola_tangent_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_line_l824_82485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_F2_l824_82444

/-- Definition of the ellipse E -/
noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Definition of the line that intersects the ellipse -/
noncomputable def line (k m x y : ℝ) : Prop :=
  y = k * x + m

/-- Definition of point P as the intersection of the line and ellipse -/
noncomputable def point_P (k m : ℝ) : ℝ × ℝ :=
  (-2 * k / m, 1 / m)

/-- Definition of point Q as the intersection of the line and x = 2 -/
noncomputable def point_Q (k m : ℝ) : ℝ × ℝ :=
  (2, 2 * k + m)

/-- Definition of the right focus F2 -/
def focus_F2 : ℝ × ℝ :=
  (1, 0)

/-- Theorem stating that the circle with diameter PQ passes through F2 -/
theorem circle_passes_through_F2 (k m : ℝ) :
  let P := point_P k m
  let Q := point_Q k m
  let F2 := focus_F2
  (F2.1 - P.1) * (Q.1 - F2.1) + (F2.2 - P.2) * (Q.2 - F2.2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_F2_l824_82444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_beta_delta_sum_l824_82492

-- Define the complex function f
def f (β δ : ℂ) (z : ℂ) : ℂ := (3 + 2*Complex.I)*z^2 + β*z + δ

-- State the theorem
theorem min_beta_delta_sum (β δ : ℂ) :
  (f β δ 2).im = 0 → (f β δ (-Complex.I)).im = 0 → 
  ∃ (min : ℝ), min = 14 ∧ 
  ∀ β' δ' : ℂ, (f β' δ' 2).im = 0 → (f β' δ' (-Complex.I)).im = 0 → 
  Complex.abs β' + Complex.abs δ' ≥ min :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_beta_delta_sum_l824_82492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_calculation_l824_82471

/-- Calculates the brokerage fee for a stock sale -/
def calculate_brokerage (cash_realized : ℚ) (brokerage_rate : ℚ) : ℚ :=
  cash_realized * (brokerage_rate / 100)

/-- Rounds a rational number to the nearest hundredth -/
def round_to_hundredth (x : ℚ) : ℚ :=
  ⌊x * 100 + 1/2⌋ / 100

theorem brokerage_calculation (cash_realized : ℚ) (brokerage_rate : ℚ) 
  (h1 : cash_realized = 104.25)
  (h2 : brokerage_rate = 1/4) :
  round_to_hundredth (calculate_brokerage cash_realized brokerage_rate) = 26/100 := by
  sorry

#eval round_to_hundredth (calculate_brokerage 104.25 (1/4))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_calculation_l824_82471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l824_82483

-- Define the ellipse parameters
variable (a b c : ℝ)

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci
noncomputable def F₁ (c : ℝ) : ℝ × ℝ := (-c, 0)
noncomputable def F₂ (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define point P
noncomputable def P : ℝ × ℝ := (1, 3/2)

-- State the conditions
axiom a_gt_b : a > b
axiom b_gt_zero : b > 0
axiom P_on_ellipse : ellipse_equation a b P.1 P.2
axiom PF₂_perp_x : (P.1 - (F₂ c).1) * (F₂ c).2 = (P.2 - (F₂ c).2) * (F₂ c).1

-- Define the line l
def line_equation (m : ℝ) (x y : ℝ) : Prop := y = m * (x + 1)

-- State the theorem
theorem ellipse_and_line_properties :
  ∃ m : ℝ,
    (∀ x y : ℝ, ellipse_equation a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
    (∃ A B M : ℝ × ℝ,
      A ≠ B ∧
      ellipse_equation a b A.1 A.2 ∧
      ellipse_equation a b B.1 B.2 ∧
      ellipse_equation a b M.1 M.2 ∧
      line_equation m A.1 A.2 ∧
      line_equation m B.1 B.2 ∧
      line_equation m (F₁ c).1 (F₁ c).2 ∧
      (M.1 + (F₂ c).1 = A.1 + B.1 ∧ M.2 + (F₂ c).2 = A.2 + B.2) ∧
      (m = 3 * Real.sqrt 5 / 10 ∨ m = -3 * Real.sqrt 5 / 10)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l824_82483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_cup_volume_l824_82460

/-- The volume of a cup formed by cutting and folding a square piece of paper. -/
theorem paper_cup_volume
  (square_side : ℝ)
  (cut_distance : ℝ)
  (cut_angle : ℝ)
  (h1 : square_side = 120)
  (h2 : cut_distance = 12)
  (h3 : cut_angle = 45) :
  (4 * (1/3) * Real.pi * cut_distance^2 * (cut_distance * Real.sqrt 2 * Real.cos (cut_angle / 2 * π / 180) * Real.sin (cut_angle / 4 * π / 180))) = 1091.52 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_cup_volume_l824_82460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_length_of_curve_l824_82408

/-- The focal length of the curve defined by x = 2cos(θ) and y = sin(θ) is 2√3 -/
theorem focal_length_of_curve : ∃ (θ : ℝ), 
  let x := 2 * Real.cos θ
  let y := Real.sin θ
  let a := 2  -- semi-major axis
  let b := 1  -- semi-minor axis
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 3 :=
by
  -- Proof goes here
  sorry

#check focal_length_of_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_length_of_curve_l824_82408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_sqrt_two_over_two_l824_82406

/-- An isosceles right triangle with an ellipse passing through its vertex -/
structure IsoscelesRightTriangleWithEllipse where
  /-- The side length of the isosceles right triangle -/
  c : ℝ
  /-- The side length is positive -/
  c_pos : c > 0

/-- The eccentricity of the ellipse in the IsoscelesRightTriangleWithEllipse -/
noncomputable def eccentricity (t : IsoscelesRightTriangleWithEllipse) : ℝ :=
  Real.sqrt 2 / 2

/-- Theorem: The eccentricity of the ellipse in an IsoscelesRightTriangleWithEllipse is √2/2 -/
theorem eccentricity_is_sqrt_two_over_two (t : IsoscelesRightTriangleWithEllipse) :
    eccentricity t = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_sqrt_two_over_two_l824_82406
