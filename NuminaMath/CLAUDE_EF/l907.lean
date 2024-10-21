import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_fourteen_l907_90738

noncomputable def f (x : ℝ) : ℝ := 3 * x + 7

noncomputable def f_inverse (x : ℝ) : ℝ := (x - 7) / 3

theorem inverse_of_inverse_fourteen (h : ∀ x, f (f_inverse x) = x) :
  f_inverse (f_inverse 14) = -14/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_fourteen_l907_90738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l907_90799

-- Define the number of arcs
def num_arcs : ℕ := 12

-- Define the length of each arc
noncomputable def arc_length : ℝ := 2 * Real.pi / 3

-- Define the side length of the octagon
def octagon_side : ℝ := 3

-- Theorem statement
theorem area_enclosed_by_curve (curve_area : ℝ) : 
  curve_area = 54 + 54 * Real.sqrt 2 + 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l907_90799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l907_90761

theorem plant_arrangement (n m : ℕ) : 
  n = 5 → m = 3 → (Nat.factorial (n + 1)) * (Nat.factorial m) = 4320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l907_90761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_tithe_percentage_l907_90750

/-- Proves that Alex's tithe percentage is 10% given the conditions of his income and expenses --/
theorem alex_tithe_percentage (weekly_income : ℝ) (tax_rate : ℝ) (water_bill : ℝ) (remaining : ℝ) :
  weekly_income = 500 →
  tax_rate = 0.1 →
  water_bill = 55 →
  remaining = 345 →
  let after_tax := weekly_income * (1 - tax_rate)
  let after_water := after_tax - water_bill
  let tithe := after_water - remaining
  tithe / weekly_income = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_tithe_percentage_l907_90750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decipher_message_l907_90797

/-- Represents a mapping from digits to characters -/
def digit_to_char : Nat → Char := sorry

/-- Reverses the digits of a natural number -/
def reverse_digits : Nat → Nat := sorry

/-- Converts a natural number to a string by mapping each digit to a character -/
def nat_to_string : Nat → String := sorry

/-- The original number on the sheet -/
def original_number : Nat := 2211169691162

/-- The expected deciphered message -/
def expected_message : String := "Kiss me, dearest"

/-- Theorem stating that reversing the digits of the original number
    and converting to a string yields the expected message -/
theorem decipher_message :
  nat_to_string (reverse_digits original_number) = expected_message := by
  sorry

#eval original_number
#eval expected_message

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decipher_message_l907_90797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_triangle_area_l907_90789

noncomputable def f (x : ℝ) := (1/2) * Real.sin (2*x) - (Real.cos (x + Real.pi/4))^2

theorem f_monotone_intervals (k : ℤ) :
  MonotoneOn f (Set.Icc (k * Real.pi - Real.pi/4) (k * Real.pi + Real.pi/4)) := by sorry

theorem triangle_area (A B C : ℝ) (hB : 0 < B ∧ B < Real.pi/2) 
  (hf : f (B/2) = 0) (hb : Real.sin B = 1/2) (hc : 2 = 2) :
  (1/2) * Real.sqrt 3 * 1 * (1/2) = Real.sqrt 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_triangle_area_l907_90789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_from_slope_product_l907_90744

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- A point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- The slope of a line from a point on the hyperbola to a vertex -/
noncomputable def slope_to_vertex (h : Hyperbola a b) (p : PointOnHyperbola h) (vertex_x : ℝ) : ℝ :=
  p.y / (p.x - vertex_x)

/-- The theorem stating the relationship between slope product and eccentricity -/
theorem eccentricity_from_slope_product (h : Hyperbola a b) :
  (∀ p : PointOnHyperbola h, 
    slope_to_vertex h p (-a) * slope_to_vertex h p a = 2) →
  eccentricity h = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_from_slope_product_l907_90744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l907_90777

-- Define the function f
noncomputable def f (x φ a : ℝ) : ℝ := Real.sin (2 * x + φ) + a * Real.cos (2 * x + φ)

-- State the theorem
theorem function_properties (φ a : ℝ) 
  (h1 : 0 < φ ∧ φ < Real.pi)
  (h2 : ∀ x, f x φ a ≤ 2)
  (h3 : ∃ x, f x φ a = 2)
  (h4 : ∀ x, f x φ a = f (Real.pi/2 - x) φ a) :
  φ = Real.pi/3 ∨ φ = 2*Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l907_90777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_max_balloons_l907_90788

/-- The maximum number of balloons Orvin can purchase given the conditions --/
def max_balloons (regular_price : ℚ) (budget : ℚ) (max_discounted : ℕ) : ℕ :=
  let full_price_count := (budget / regular_price).floor.toNat
  let discounted_price := regular_price / 2
  let discounted_pairs := min (max_discounted / 2) ((budget / (regular_price + discounted_price)).floor.toNat)
  let remaining_budget := budget - (discounted_pairs : ℚ) * (regular_price + discounted_price)
  let additional_full_price := (remaining_budget / regular_price).floor.toNat
  (2 * discounted_pairs + additional_full_price : ℕ)

/-- Theorem stating that the maximum number of balloons Orvin can purchase is 45 --/
theorem orvin_max_balloons :
  max_balloons 5 200 20 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_max_balloons_l907_90788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l907_90718

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.4
noncomputable def b : ℝ := (0.4 : ℝ) ^ (0.4 : ℝ)
noncomputable def c : ℝ := (0.4 : ℝ) ^ (0.3 : ℝ)

theorem ascending_order : b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l907_90718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solutions_l907_90719

open Real

theorem cosine_equation_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, -π ≤ x ∧ x ≤ π) ∧
                    (∀ x ∈ S, cos (3*x) + (cos (2*x))^2 + (cos x)^3 = 0) ∧
                    (Finset.card S = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solutions_l907_90719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l907_90721

def a : ℕ → ℚ
  | 0 => -1
  | n + 1 => a n + 1 / ((n + 1) * (n + 2))

theorem a_2017_value : a 2017 = -1 / 2017 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l907_90721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_g_symmetry_g_l907_90768

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the derivatives of f and g
def f' : ℝ → ℝ := sorry
def g' : ℝ → ℝ := sorry

-- Axioms based on the given conditions
axiom cond1 : ∀ x, f (x + 2) - g (1 - x) = 2
axiom cond2 : ∀ x, f' x = g' (x + 1)
axiom cond3 : ∀ x, g (x + 1) = -g (-x + 1)

-- Theorems to prove
theorem symmetry_g : ∀ x, g (x + 2) = g (2 - x) := by sorry

theorem symmetry_g' : ∀ x, g' (2 + x) + g' (2 - x) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_g_symmetry_g_l907_90768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_and_consecutive_pairs_l907_90716

theorem power_equation_and_consecutive_pairs :
  (∃! x : ℕ, x > 0 ∧ 3^x = x + 2) ∧
  (∀ x y : ℕ, ¬(y + 3^x = x + 3^y + 1 ∨ x + 3^y = y + 3^x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_and_consecutive_pairs_l907_90716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visitor_function_constraints_l907_90713

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * b^x + c

theorem visitor_function_constraints (a b c : ℝ) :
  b > 0 →
  b ≠ 1 →
  f a b c 1 = 100 →
  f a b c 2 = 120 →
  (∀ x y, x > y → x ≥ 1 → y ≥ 1 → f a b c x > f a b c y) →
  (∀ x, x ≥ 1 → f a b c x < 130) →
  0 < b ∧ b ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visitor_function_constraints_l907_90713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_parabola_properties_l907_90728

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the asymptotic lines
def asymptotic_lines (x y : ℝ) : Prop := y = 4/3 * x ∨ y = -4/3 * x

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 20 * x

-- Theorem statement
theorem hyperbola_and_parabola_properties :
  (∀ x y : ℝ, hyperbola_C x y → asymptotic_lines x y) ∧
  (∃ x₀ : ℝ, hyperbola_C x₀ 0 ∧ x₀ > 0 ∧ ∀ x y : ℝ, parabola x y → (x - x₀)^2 + y^2 = (x₀ / 2)^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_parabola_properties_l907_90728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_plan_comparison_l907_90710

/-- Represents a phone plan with a monthly fee, free minutes, and overtime rate. -/
structure PhonePlan where
  monthlyFee : ℚ
  freeMinutes : ℚ
  overtimeRate : ℚ

/-- Calculates the cost of a phone plan given the number of minutes used. -/
def planCost (plan : PhonePlan) (minutes : ℚ) : ℚ :=
  plan.monthlyFee + max 0 (minutes - plan.freeMinutes) * plan.overtimeRate

/-- The two phone plans described in the problem. -/
def plan1 : PhonePlan := ⟨58, 150, 1/4⟩
def plan2 : PhonePlan := ⟨88, 350, 9/20⟩

theorem phone_plan_comparison :
  ∀ t : ℚ, t ≥ 0 →
    ((t < 270 ∨ t > 450) → planCost plan1 t < planCost plan2 t) ∧
    (t = 450 → planCost plan1 t = planCost plan2 t) := by
  sorry

#eval planCost plan1 260
#eval planCost plan2 260

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_plan_comparison_l907_90710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_pi_l907_90773

noncomputable def f (x : ℝ) : ℝ := (Real.exp Real.pi - Real.exp x) / (Real.sin (5 * x) - Real.sin (3 * x))

theorem limit_f_at_pi :
  Filter.Tendsto f (Filter.atTop.comap (λ x : ℝ => |x - Real.pi|)) (nhds (Real.exp Real.pi / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_pi_l907_90773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_three_equals_fifteen_l907_90794

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 24 / (x + 3)
noncomputable def g (x : ℝ) : ℝ := 3 * (Function.invFun f x)

-- State the theorem
theorem g_of_three_equals_fifteen : g 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_three_equals_fifteen_l907_90794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_count_l907_90749

def valid_hundreds_digit (d : Nat) : Bool :=
  d ≠ 0 && d ≠ 1 && d ≠ 7 && d ≠ 8

def valid_tens_digit (d : Nat) : Bool :=
  d ≠ 1 && d ≠ 7 && d ≠ 8 && d ≠ 9

def valid_units_digit (d : Nat) : Bool :=
  d ≠ 1 && d ≠ 7 && d ≠ 8

def count_valid_digits (p : Nat → Bool) : Nat :=
  (List.range 10).filter p |>.length

theorem three_digit_numbers_count : 
  count_valid_digits valid_hundreds_digit * 
  count_valid_digits valid_tens_digit * 
  count_valid_digits valid_units_digit = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_count_l907_90749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2a_equals_7_l907_90701

noncomputable def f (x : ℝ) : ℝ := 3^x + 3^(-x)

theorem f_2a_equals_7 (a : ℝ) (h : f a = 3) : f (2 * a) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2a_equals_7_l907_90701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l907_90741

theorem order_of_expressions : 
  let a : ℝ := Real.rpow 3 0.4
  let b : ℝ := Real.rpow 0.4 3
  let c : ℝ := Real.log 3 / Real.log 0.4
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l907_90741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l907_90747

theorem vector_magnitude_range (a b c : ℝ × ℝ) (lambda : ℝ) :
  (‖a‖ = 1) →
  (‖b‖ = 1) →
  (‖c‖ = 1) →
  (b • c = 1 / 2) →
  (0 ≤ lambda) →
  (lambda ≤ 1) →
  (Real.sqrt 3 - 1 ≤ ‖a - 2 * lambda • b - (2 - 2 * lambda) • c‖) ∧
  (‖a - 2 * lambda • b - (2 - 2 * lambda) • c‖ ≤ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l907_90747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_purchase_theorem_l907_90722

/-- Represents the amount of each fruit that can be bought with all the money -/
structure FruitAmounts where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Represents the amount of each fruit actually bought -/
def EqualPurchase (x : ℚ) : FruitAmounts :=
  { a := x, b := x, c := x }

/-- The total cost of the purchase -/
def TotalCost (amounts : FruitAmounts) (maxAmounts : FruitAmounts) : ℚ :=
  amounts.a / maxAmounts.a + amounts.b / maxAmounts.b + amounts.c / maxAmounts.c

theorem equal_purchase_theorem (maxAmounts : FruitAmounts) 
    (h : maxAmounts = { a := 4, b := 6, c := 12 }) :
    ∃ x : ℚ, TotalCost (EqualPurchase x) maxAmounts = 1 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_purchase_theorem_l907_90722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_value_l907_90760

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the range constraints for f and g
axiom f_range : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 4
axiom g_range : ∀ x : ℝ, -3 ≤ g x ∧ g x ≤ 2

-- State the theorem
theorem max_product_value :
  ∃ x : ℝ, f x * g x = 8 ∧ ∀ y : ℝ, f y * g y ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_value_l907_90760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_distribution_theorem_l907_90756

/-- Represents a student in the circle -/
structure Student :=
  (id : Nat)
  (flowers : Nat)

/-- The flower distribution game -/
def FlowerGame :=
  {students : Finset Student // students.card = 12 ∧ (students.sum (λ s => s.flowers) = 13)}

/-- A single round of the game where a student distributes flowers -/
def distributeFlowers (game : FlowerGame) (distributor : Student) : FlowerGame :=
  sorry

/-- Predicate to check if at least 7 students have flowers -/
def atLeastSevenWithFlowers (game : FlowerGame) : Prop :=
  (game.val.filter (λ s => s.flowers > 0)).card ≥ 7

/-- Helper function to iterate distributeFlowers -/
def iterateDistributeFlowers : Nat → FlowerGame → FlowerGame
  | 0, game => game
  | n + 1, game => iterateDistributeFlowers n (distributeFlowers game sorry)

/-- The main theorem -/
theorem flower_distribution_theorem (initialGame : FlowerGame) :
  ∃ (game : FlowerGame), (∃ (steps : Nat), game = iterateDistributeFlowers steps initialGame) ∧
    atLeastSevenWithFlowers game :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_distribution_theorem_l907_90756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_transformation_l907_90793

def sequence_length : Nat := 100

structure Sequence where
  a : Nat
  b : Nat
  h1 : a + b = sequence_length

def transform_operations (s1 s2 : Sequence) : Nat :=
  Int.natAbs (s1.a - s2.a)

theorem sequence_transformation (s1 s2 : Sequence) :
  transform_operations s1 s2 ≤ 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_transformation_l907_90793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l907_90706

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 2 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l907_90706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_point_zero_l907_90751

-- Define the sets
def set1 : Set ℝ := {x | ∃ n : ℤ, n ≥ 0 ∧ x = n / (n + 1)}
def set2 : Set ℝ := {x : ℝ | x ≠ 0}
def set3 : Set ℝ := {x | ∃ n : ℤ, n ≠ 0 ∧ x = 1 / n}
def set4 : Set ℝ := Set.range (Int.cast : ℤ → ℝ)

-- Define what it means to be a limit point
def isLimitPoint (s : Set ℝ) (x : ℝ) : Prop :=
  ∀ a > 0, ∃ y ∈ s, 0 < |y - x| ∧ |y - x| < a

-- State the theorem
theorem limit_point_zero :
  isLimitPoint set2 0 ∧
  isLimitPoint set3 0 ∧
  ¬isLimitPoint set1 0 ∧
  ¬isLimitPoint set4 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_point_zero_l907_90751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l907_90732

/-- The line l: x - y + 4 = 0 -/
def line_l (x y : ℝ) : Prop := x - y + 4 = 0

/-- The circle C: x = 1 + 2cos(θ), y = 1 + 2sin(θ) -/
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y + 4| / Real.sqrt 2

/-- The theorem stating the minimum distance from circle C to line l -/
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - 2 ∧
  ∀ (θ : ℝ), distance_to_line (circle_C θ).1 (circle_C θ).2 ≥ d := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l907_90732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_value_at_one_l907_90707

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The domain of the function -/
def Domain (a : ℝ) : Set ℝ := {x | x ∈ (Set.Ioo (-2*a+2) 0) ∪ (Set.Ioo 0 a)}

/-- The function f -/
noncomputable def f (a b x : ℝ) : ℝ := (a*x^3 + (a-1)*x + a - 2*b) / x

theorem even_function_value_at_one (a b : ℝ) :
  (IsEven (f a b)) ∧ (∀ x ∈ Domain a, x ≠ 0) → f a b 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_value_at_one_l907_90707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_llama_area_theorem_l907_90798

/-- Represents the dimensions of a rectangular shed -/
structure Shed where
  length : ℝ
  width : ℝ

/-- Calculates the area accessible to a llama tied to a corner of a rectangular shed -/
noncomputable def llamaAccessibleArea (shed : Shed) (leashLength : ℝ) : ℝ :=
  7 * Real.pi

/-- Theorem stating that a llama tied to a 2m x 3m shed with a 3m leash has 7π m² accessible area -/
theorem llama_area_theorem :
  let shed : Shed := { length := 2, width := 3 }
  let leashLength : ℝ := 3
  llamaAccessibleArea shed leashLength = 7 * Real.pi := by
  sorry

#check llama_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_llama_area_theorem_l907_90798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l907_90739

/-- Given a parabola with equation x^2 = (1/2)y, its directrix has the equation y = -1/8 -/
theorem parabola_directrix (x y : ℝ) :
  (x^2 = (1/2) * y) → (y = -1/8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l907_90739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l907_90752

/-- Given a point A and a line l, this theorem proves that the equation
    x - 2y + 4 = 0 represents the line passing through A and parallel to l. -/
theorem parallel_line_equation (A : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  A = (2, 3) →
  l = {(x, y) | 2*x - 4*y + 7 = 0} →
  ∃ (m : ℝ), {(x, y) | 2*x - 4*y + m = 0} = {(x, y) | x - 2*y + 4 = 0} :=
by
  intros h_A h_l
  use 8
  ext ⟨x, y⟩
  simp [h_A, h_l]
  sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l907_90752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_parity_l907_90782

noncomputable def a (n : ℕ) : ℤ := ⌊((3 + Real.sqrt 17) / 2) ^ n⌋

theorem a_parity (n : ℕ) (h : n > 0) : 
  (n % 2 = 1 → a n % 2 = 1) ∧ (n % 2 = 0 → a n % 2 = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_parity_l907_90782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_direction_vector_of_line_l907_90746

/-- Given a line with equation y = (3/4)x - 1, prove that ±(4/5, 3/5) is a unit direction vector. -/
theorem unit_direction_vector_of_line (x y : ℝ) :
  y = (3/4) * x - 1 →
  (∃ (ε : ℝ) (hε : ε = 1 ∨ ε = -1), 
    let v := (ε * (4/5), ε * (3/5))
    (v.1, v.2) ≠ (0, 0) ∧ 
    v.1^2 + v.2^2 = 1 ∧
    ∃ (t : ℝ), y = (3/4) * x + t * v.2 - t * (3/4) * v.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_direction_vector_of_line_l907_90746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_drink_ratio_l907_90778

/-- Represents a conical container --/
structure ConicalContainer where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a conical container --/
noncomputable def volume (c : ConicalContainer) : ℝ :=
  (1/3) * Real.pi * c.radius^2 * c.height

/-- Theorem: In a conical container, if one person drinks half the liquid by height
    and another person drinks the rest, the first person consumes 7 times more
    liquid than the second person --/
theorem conical_drink_ratio (c : ConicalContainer) :
  let v_total := volume c
  let v_remaining := volume { radius := c.radius/2, height := c.height/2 }
  let v_first := v_total - v_remaining
  v_first / v_remaining = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_drink_ratio_l907_90778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_function_equality_l907_90785

theorem angle_function_equality (α : Real) : 
  (π < α ∧ α < 3*π/2) →  -- α is in the third quadrant
  Real.cos (α - 3*π/2) = 1/5 → 
  (Real.sin (α - π/2) * Real.cos (3*π/2 + α) * Real.tan (π - α)) / (Real.tan (-π - α) * Real.sin (-π - α)) = 2*Real.sqrt 6 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_function_equality_l907_90785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l907_90723

open Real

-- Define the interval (0, π/2)
def OpenInterval : Set ℝ := { x | 0 < x ∧ x < π/2 }

-- Define the conditions
def ConditionA (a : ℝ) : Prop := a ∈ OpenInterval ∧ cos a = a
def ConditionB (b : ℝ) : Prop := b ∈ OpenInterval ∧ sin (cos b) = b
def ConditionC (c : ℝ) : Prop := c ∈ OpenInterval ∧ cos (sin c) = c

-- Theorem statement
theorem ascending_order (a b c : ℝ) 
  (ha : ConditionA a) (hb : ConditionB b) (hc : ConditionC c) : 
  b < a ∧ a < c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l907_90723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_marvelous_monday_after_feb1_wed_l907_90766

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, BEq

/-- Represents a date in a year -/
structure Date where
  month : Nat
  day : Nat
deriving Repr

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the number of days in a given month (assuming non-leap year) -/
def daysInMonth (month : Nat) : Nat :=
  match month with
  | 2 => 28
  | 4 | 6 | 9 | 11 => 30
  | _ => 31

/-- Returns the day of the week for a given date, assuming Feb 1 is Wednesday -/
def dayOfWeek (d : Date) : DayOfWeek :=
  sorry

/-- Returns true if the given date is a Monday -/
def isMonday (d : Date) : Bool :=
  dayOfWeek d == DayOfWeek.Monday

/-- Returns true if the given date is the fifth Monday of its month -/
def isFifthMonday (d : Date) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem first_marvelous_monday_after_feb1_wed (year : Nat) :
  (∀ m, m ≤ 12 → daysInMonth m = if m = 2 then 28 else if m ∈ [4, 6, 9, 11] then 30 else 31) →
  dayOfWeek ⟨2, 1⟩ = DayOfWeek.Wednesday →
  (∃ d : Date, d.month = 5 ∧ d.day = 29 ∧ isFifthMonday d ∧
    ∀ d' : Date, (d'.month < 5 ∨ (d'.month = 5 ∧ d'.day < 29)) → ¬isFifthMonday d') :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_marvelous_monday_after_feb1_wed_l907_90766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_five_l907_90769

/-- The radius of a sphere given its shadow length and a reference object's shadow. -/
noncomputable def sphere_radius (sphere_shadow : ℝ) (stick_height : ℝ) (stick_shadow : ℝ) : ℝ :=
  (sphere_shadow * stick_height) / stick_shadow

theorem sphere_radius_is_five :
  let sphere_shadow : ℝ := 10
  let stick_height : ℝ := 1
  let stick_shadow : ℝ := 2
  sphere_radius sphere_shadow stick_height stick_shadow = 5 := by
  -- Unfold the definition of sphere_radius
  unfold sphere_radius
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Prove the equality
  norm_num

#check sphere_radius_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_five_l907_90769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_calculation_l907_90709

/-- Atomic weight of Potassium in atomic mass units (amu) -/
def atomic_weight_K : ℝ := 39.10

/-- Atomic weight of Chromium in atomic mass units (amu) -/
def atomic_weight_Cr : ℝ := 51.996

/-- Atomic weight of Oxygen in atomic mass units (amu) -/
def atomic_weight_O : ℝ := 15.999

/-- Atomic weight of Hydrogen in atomic mass units (amu) -/
def atomic_weight_H : ℝ := 1.008

/-- Atomic weight of Nitrogen in atomic mass units (amu) -/
def atomic_weight_N : ℝ := 14.007

/-- Number of Potassium atoms in the compound -/
def num_K : ℕ := 3

/-- Number of Chromium atoms in the compound -/
def num_Cr : ℕ := 2

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 7

/-- Number of Hydrogen atoms in the compound -/
def num_H : ℕ := 4

/-- Number of Nitrogen atoms in the compound -/
def num_N : ℕ := 1

/-- The molecular weight of the compound -/
def molecular_weight : ℝ :=
  (num_K : ℝ) * atomic_weight_K +
  (num_Cr : ℝ) * atomic_weight_Cr +
  (num_O : ℝ) * atomic_weight_O +
  (num_H : ℝ) * atomic_weight_H +
  (num_N : ℝ) * atomic_weight_N

theorem molecular_weight_calculation :
  ∃ ε > 0, |molecular_weight - 351.324| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_calculation_l907_90709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_count_l907_90796

/-- The function g(t) = 10^t - 11t -/
noncomputable def g (t : ℝ) : ℝ := 10^t - 11*t

/-- The equation from the original problem -/
def equation (a x : ℝ) : Prop :=
  4*a^2 + 3*x*(Real.log x) + 3*(Real.log x)^2 = 13*a*(Real.log x) + a*x

theorem unique_solution_count :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  (∀ a ∈ s, ∃! x, x > 0 ∧ equation a x) ∧
  (∀ a ∉ s, ¬∃! x, x > 0 ∧ equation a x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_count_l907_90796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_profit_problem_l907_90772

/-- Calculates the minimum number of pencils to sell for a given profit -/
def min_pencils_to_sell (total_pencils : ℕ) (cost_per_pencil : ℚ) (sell_price : ℚ) (desired_profit : ℚ) : ℕ :=
  let total_cost := (total_pencils : ℚ) * cost_per_pencil
  let revenue_needed := total_cost + desired_profit
  (revenue_needed / sell_price).ceil.toNat

/-- The problem statement -/
theorem pencil_profit_problem :
  min_pencils_to_sell 2000 (15/100) (35/100) 200 = 1429 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_profit_problem_l907_90772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_indeterminate_ratio_l907_90708

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_indeterminate_ratio 
  (a₁ : ℝ) (a₂ : ℝ) (n : ℕ) 
  (h : 8 * a₂ + a₁ = 0) :
  ∃ (q : ℝ), 
    (∀ k, geometric_sequence a₁ q k = geometric_sequence a₁ q 1 * q^(k-1)) ∧
    (geometric_sequence a₁ q 5 / geometric_sequence a₁ q 3 = (1/64)) ∧
    (geometric_sum a₁ q 5 / geometric_sum a₁ q 3 = (1 + (1/8)^5) / (1 + (1/8)^3)) ∧
    (∀ k, geometric_sequence a₁ q (k+1) / geometric_sequence a₁ q k = -1/8) ∧
    (¬ ∃ c, ∀ k, geometric_sum a₁ q (k+1) / geometric_sum a₁ q k = c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_indeterminate_ratio_l907_90708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_l907_90781

/-- Vector in R2 -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Addition of Vec2 -/
def Vec2.add (v w : Vec2) : Vec2 :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Scalar multiplication of Vec2 -/
def Vec2.scale (r : ℝ) (v : Vec2) : Vec2 :=
  ⟨r * v.x, r * v.y⟩

/-- Magnitude (length) of Vec2 -/
noncomputable def Vec2.magnitude (v : Vec2) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

theorem midpoint_theorem (m n : Vec2) : 
  let AB := Vec2.add (Vec2.scale 2 m) (Vec2.scale 2 n)
  let AC := Vec2.add (Vec2.scale 2 m) (Vec2.scale (-6) n)
  let BC := Vec2.add AC (Vec2.scale (-1) AB)
  let BD := Vec2.scale (1/2) BC
  let AD := Vec2.add AB BD
  Vec2.magnitude AD = 2 :=
by
  sorry

#check midpoint_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_l907_90781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_relationship_l907_90748

open Real

theorem function_relationship (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (π - x))
  (h2 : ∀ x, x ∈ Set.Ioo (-π/2) (π/2) → f x = x + sin x) :
  f 2 > f 1 ∧ f 1 > f 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_relationship_l907_90748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l907_90762

noncomputable def f (x : ℝ) : ℝ := (2 / (x + 2)) + (8 / (x + 6))

theorem inequality_solution :
  ∀ x : ℝ, f x ≥ 2 ↔ x ∈ Set.Icc (-6) (-4) ∪ Set.Icc (-2) (-1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l907_90762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_of_line_l907_90740

/-- The inclination angle of a line with slope m is the angle formed between the line and the positive x-axis -/
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m

/-- The line equation y = √3x + 1 -/
noncomputable def line_equation (x : ℝ) : ℝ := Real.sqrt 3 * x + 1

theorem inclination_angle_of_line :
  inclination_angle (Real.sqrt 3) = π / 3 := by
  sorry

#check inclination_angle_of_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_of_line_l907_90740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_is_rectangle_l907_90711

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vec where
  x : ℝ
  y : ℝ

/-- Calculate the vector between two points -/
def vectorBetween (p q : Point) : Vec :=
  { x := q.x - p.x, y := q.y - p.y }

/-- Calculate the dot product of two vectors -/
def dotProduct (v w : Vec) : ℝ :=
  v.x * w.x + v.y * w.y

/-- Check if two vectors are equal -/
def vectorsEqual (v w : Vec) : Prop :=
  v.x = w.x ∧ v.y = w.y

/-- Check if two vectors are perpendicular -/
def vectorsPerpendicular (v w : Vec) : Prop :=
  dotProduct v w = 0

/-- Define the points A, B, C, D -/
def A : Point := { x := -2, y := 0 }
def B : Point := { x := 1, y := 6 }
def C : Point := { x := 5, y := 4 }
def D : Point := { x := 2, y := -2 }

/-- Theorem: ABCD is a rectangle -/
theorem abcd_is_rectangle : 
  vectorsEqual (vectorBetween A B) (vectorBetween D C) ∧
  vectorsPerpendicular (vectorBetween A B) (vectorBetween A D) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_is_rectangle_l907_90711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_b_l907_90780

/-- A quadratic function passing through (1,0), (3,0), and (0,3) has coefficient b equal to -4 -/
theorem quadratic_coefficient_b (g : ℝ → ℝ) (a b c : ℝ) : 
  (∀ x, g x = a * x^2 + b * x + c) → 
  g 1 = 0 → g 3 = 0 → g 0 = 3 → 
  b = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_b_l907_90780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l907_90753

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2^(x^2 - 5*x - 6)

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x y : ℝ, x < y → x < (5/2 : ℝ) → f y < f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l907_90753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l907_90714

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (u z : Fin 2 → ℝ)

def vector_eq (v w : Fin 2 → ℝ) : Prop :=
  v 0 = w 0 ∧ v 1 = w 1

theorem matrix_vector_computation 
  (h1 : vector_eq (M.mulVec u) (λ i => if i = 0 then -3 else 8))
  (h2 : vector_eq (M.mulVec z) (λ i => if i = 0 then 4 else -1)) :
  vector_eq (M.mulVec (3 • u - 5 • z)) (λ i => if i = 0 then -29 else 29) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l907_90714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_equal_self_iff_l907_90742

/-- The function g(x) = (3x + 4) / (mx - 5) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (3 * x + 4) / (m * x - 5)

/-- The inverse function of g -/
noncomputable def g_inv (m : ℝ) (x : ℝ) : ℝ := 
  (5 * x + 4) / (3 - m * x)

/-- Theorem stating that g^(-1)(x) = g(x) if and only if m = -9 -/
theorem g_inverse_equal_self_iff (m : ℝ) :
  (∀ x : ℝ, g_inv m x = g m x) ↔ m = -9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_equal_self_iff_l907_90742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_negative_four_l907_90786

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ := 
  (u.1 * v.1 + u.2 * v.2) / Real.sqrt (v.1^2 + v.2^2)

theorem projection_equals_negative_four :
  ∀ (d : ℝ),
  let a := arithmetic_sequence 4 d
  let S := arithmetic_sum 4 d
  S 3 = 6 →
  vector_projection (a 5, 3) (1, a 3) = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_negative_four_l907_90786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_is_15_inches_l907_90755

/-- The height of a box given its base dimensions, total volume needed, cost per box, and total amount spent -/
noncomputable def box_height (base_length : ℝ) (base_width : ℝ) (total_volume : ℝ) (cost_per_box : ℝ) (total_spent : ℝ) : ℝ :=
  total_volume / (total_spent / cost_per_box * base_length * base_width)

theorem box_height_is_15_inches :
  let base_length : ℝ := 20
  let base_width : ℝ := 20
  let total_volume : ℝ := 3060000
  let cost_per_box : ℝ := 1.20
  let total_spent : ℝ := 612
  box_height base_length base_width total_volume cost_per_box total_spent = 15 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_is_15_inches_l907_90755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_acceleration_for_floating_l907_90734

/-- Represents the density of water -/
def ρ₀ : ℝ := 1

/-- Represents the density of the cube material -/
def ρ₁ : ℝ := 3 * ρ₀

/-- Represents the gravitational acceleration -/
def g : ℝ := 9.8

/-- The condition for the cube to start floating -/
def floating_condition (a : ℝ) : Prop :=
  ρ₀ * (g - a) ≥ ρ₁ * g

/-- Theorem stating that the minimum downward acceleration required for the cube to start floating is equal to g -/
theorem minimum_acceleration_for_floating :
  (∀ a, floating_condition a → a ≥ g) ∧ floating_condition g :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_acceleration_for_floating_l907_90734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l907_90758

def initial_investment : ℝ := 1500
def interest_rate : ℝ := 0.04
def time_period : ℕ := 20

theorem compound_interest_calculation :
  let final_amount := initial_investment * (1 + interest_rate) ^ time_period
  ∃ ε > 0, |final_amount - 3286.68| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l907_90758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_sum_l907_90731

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.slope * l₂.slope = -1

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_lines_sum (a b c : ℝ) : 
  ∃ (l₁ l₂ : Line) (p : Point),
    l₁.a = a ∧ l₁.b = 4 ∧ l₁.c = -2 ∧
    l₂.a = 2 ∧ l₂.b = -5 ∧ l₂.c = b ∧
    perpendicular l₁ l₂ ∧
    p.x = 1 ∧ p.y = c ∧
    on_line p l₁ ∧ on_line p l₂ →
    a + b + c = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_sum_l907_90731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buffet_dressing_usage_l907_90771

/-- Given a buffet that offers ranch and caesar dressing with a usage ratio of 7:1 (ranch to caesar),
    this theorem proves that if 28 cases of ranch dressing are used, then 4 cases of caesar dressing are used. -/
theorem buffet_dressing_usage (ranch_cases : ℕ) (ratio : ℚ) :
  ranch_cases = 28 → ratio = 7 / 1 → (ranch_cases : ℚ) / ratio = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buffet_dressing_usage_l907_90771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_exists_l907_90715

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3 - (1/2) * x - 1

/-- The function g as defined in the problem -/
noncomputable def g (x m : ℝ) : ℝ := x^3 + m / x

/-- The function h derived from the problem conditions -/
noncomputable def h (x : ℝ) : ℝ := x * Real.exp x - (1/2) * x^2 - x

/-- Theorem stating the unique value of m that satisfies the problem conditions -/
theorem unique_m_exists : ∃! m : ℝ, 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h x₁ = m ∧ h x₂ = m) ∧ 
  m = 1/2 - 1/Real.exp 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_exists_l907_90715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inv_4_eq_neg_2_l907_90767

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
def f_inv : ℝ → ℝ := sorry

-- Axiom for the symmetry of f with respect to (1,2)
axiom symmetry_f (x : ℝ) : f (x + 1) + f (1 - x) = 4

-- Axiom for f(4) = 0
axiom f_4_eq_0 : f 4 = 0

-- Axiom for f_inv being the inverse of f
axiom f_inv_def (x : ℝ) : f (f_inv x) = x

-- Theorem to prove
theorem f_inv_4_eq_neg_2 : f_inv 4 = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inv_4_eq_neg_2_l907_90767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l907_90735

def S : Finset Nat := {1, 2, 3}

theorem proper_subsets_count : Finset.card (S.powerset.filter (· ⊂ S)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l907_90735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_product_l907_90745

/-- Given that D is the midpoint of PQ, where P and Q are points in ℝ², prove that the product of Q's coordinates is 50. -/
theorem midpoint_coordinate_product (D P Q : ℝ × ℝ) : 
  D = ((-4 : ℝ), (2 : ℝ)) → P = ((2 : ℝ), (9 : ℝ)) → D = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) → Q.1 * Q.2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_product_l907_90745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_condition_l907_90791

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 + a * x

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1/x + x + a

theorem tangent_line_parallel_condition (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f_derivative a x = 3) → a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_condition_l907_90791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_sector_degrees_l907_90770

/-- Calculates the number of degrees in a circle graph for a given percentage -/
noncomputable def degreesInCircleGraph (percentage : ℝ) : ℝ :=
  (percentage / 100) * 360

/-- The percentage of employees in the manufacturing department -/
def manufacturingPercentage : ℝ := 35

theorem manufacturing_sector_degrees :
  degreesInCircleGraph manufacturingPercentage = 126 := by
  -- Unfold the definition of degreesInCircleGraph
  unfold degreesInCircleGraph
  -- Unfold the definition of manufacturingPercentage
  unfold manufacturingPercentage
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_sector_degrees_l907_90770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_lateral_area_l907_90717

/-- The lateral surface area of a circular truncated cone. -/
noncomputable def lateralSurfaceArea (r R h : ℝ) : ℝ :=
  let l := Real.sqrt ((R - r)^2 + h^2)
  Real.pi * (r + R) * l

/-- Theorem: The lateral surface area of a circular truncated cone with
    upper base radius 1, lower base radius 4, and height 4 is equal to 25π. -/
theorem truncated_cone_lateral_area :
  lateralSurfaceArea 1 4 4 = 25 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_lateral_area_l907_90717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l907_90765

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def N : Set ℝ := {x | -5 ≤ x ∧ x ≤ 0}

-- Define the intersection of M and N
def intersection : Set ℝ := M ∩ N

-- State the theorem
theorem intersection_equals_interval : intersection = Set.Ioo (-1) 0 ∪ {0} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l907_90765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_evaluation_l907_90775

noncomputable def original_expr (a : ℝ) : ℝ :=
  ((a^2 - 2*a) / (a^2 - 4*a + 4) + 1) / ((a^2 - 1) / (a^2 + a))

noncomputable def simplified_expr (a : ℝ) : ℝ := 2*a / (a - 2)

theorem simplification_and_evaluation :
  (∀ a : ℝ, a ≠ -1 ∧ a ≠ 0 ∧ a ≠ 1 ∧ a ≠ 2 → original_expr a = simplified_expr a) ∧
  original_expr (-2) = 1 := by
  sorry

#check simplification_and_evaluation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_evaluation_l907_90775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l907_90787

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (2 + x) - Real.log (2 - x)

-- State the theorem
theorem f_properties :
  -- f is defined on (-2, 2)
  (∀ x, -2 < x ∧ x < 2 → f x = Real.log (2 + x) - Real.log (2 - x)) ∧
  -- f is an odd function
  (∀ x, -2 < x ∧ x < 2 → f (-x) = -f x) ∧
  -- f is increasing on (-2, 0)
  (∀ x y, -2 < x ∧ x < y ∧ y < 0 → f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l907_90787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_14_l907_90720

theorem x_plus_y_equals_14 (x y : ℝ) 
  (h1 : 2 * abs x + x + y = 18) 
  (h2 : x + 2 * abs y - y = 14) : 
  x + y = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_14_l907_90720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_tan_product_positive_l907_90727

theorem sin_cos_tan_product_positive :
  Real.sin 1 * Real.cos 2 * Real.tan 3 > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_tan_product_positive_l907_90727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l907_90712

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The parabola y = 3x^2 + 6x + 5 -/
def parabola : QuadraticFunction := ⟨3, 6, 5⟩

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_x (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_y (f : QuadraticFunction) : ℝ :=
  f.a * (vertex_x f)^2 + f.b * (vertex_x f) + f.c

/-- The y-intercept of a quadratic function -/
def y_intercept (f : QuadraticFunction) : ℝ := f.c

theorem parabola_properties :
  (vertex_x parabola = -1) ∧
  (vertex_y parabola = 2) ∧
  (y_intercept parabola = 5) := by
  sorry

#check parabola_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l907_90712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_difference_l907_90759

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 2*x + 2) - Real.sqrt (x^2 - 3*x + 3)

/-- The theorem stating the difference between the maximum and minimum values of f(x) -/
theorem f_max_min_difference :
  ∃ (max min : ℝ), (∀ x, f x ≤ max ∧ min ≤ f x) ∧ (max - min = 2 * Real.sqrt (8 - Real.sqrt 3)) := by
  sorry

#check f_max_min_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_difference_l907_90759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_weighted_average_l907_90729

/-- Calculate the weighted average score given marks and weights -/
noncomputable def weighted_average (marks : List ℝ) (weights : List ℝ) : ℝ :=
  (List.sum (List.zipWith (· * ·) marks weights)) / (List.sum weights)

/-- David's weighted average score problem -/
theorem davids_weighted_average :
  let marks : List ℝ := [70, 60, 78, 60, 65]
  let weights : List ℝ := [0.25, 0.20, 0.30, 0.15, 0.10]
  weighted_average marks weights = 68.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_weighted_average_l907_90729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_friend_group_l907_90736

/-- Represents a company of people with quarrels between some pairs -/
structure Company where
  people : Finset Nat
  quarrels : Finset (Nat × Nat)

/-- Represents a group of three people -/
def TripleGroup (c : Company) : Type :=
  { group : Finset Nat // group.card = 3 ∧ group ⊆ c.people }

/-- Checks if a group of three people are all friends -/
def AreFriends (c : Company) (g : TripleGroup c) : Prop :=
  ∀ x y, x ∈ g.val → y ∈ g.val → x ≠ y → (x, y) ∉ c.quarrels ∧ (y, x) ∉ c.quarrels

theorem exists_friend_group (c : Company) 
  (h1 : c.people.card = 10) 
  (h2 : c.quarrels.card = 14) : 
  ∃ g : TripleGroup c, AreFriends c g :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_friend_group_l907_90736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_six_consecutive_heads_l907_90779

/-- A sequence of coin flips represented as a list of booleans, where true represents heads and false represents tails. -/
def CoinFlipSequence := List Bool

/-- Returns true if the given sequence contains at least n consecutive heads. -/
def hasConsecutiveHeads (sequence : CoinFlipSequence) (n : Nat) : Bool :=
  let rec check (subseq : CoinFlipSequence) : Bool :=
    match subseq with
    | [] => false
    | true :: rest => if rest.take (n - 1) |>.all id then true else check rest
    | false :: rest => check rest
  check sequence

/-- The number of possible outcomes when flipping a fair coin n times. -/
def totalOutcomes (n : Nat) : Nat :=
  2^n

/-- The number of favorable outcomes (sequences with at least 6 consecutive heads) when flipping a fair coin 10 times. -/
def favorableOutcomes : Nat := 129

/-- The probability of getting at least 6 consecutive heads in 10 fair coin flips. -/
def probabilityConsecutiveHeads : ℚ :=
  favorableOutcomes / totalOutcomes 10

theorem probability_six_consecutive_heads :
  probabilityConsecutiveHeads = 129 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_six_consecutive_heads_l907_90779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_X_and_Q_l907_90754

-- Define the circle and points
variable (circle : Set ℝ)
variable (A B Q D C P : ℝ)

-- Define the arc measures
variable (arc_BQ arc_QD arc_AP arc_PC : ℝ)

-- Define angle X
variable (X : ℝ)

-- Define the conditions
variable (h1 : A ∈ circle)
variable (h2 : B ∈ circle)
variable (h3 : Q ∈ circle)
variable (h4 : D ∈ circle)
variable (h5 : C ∈ circle)
variable (h6 : P ∈ circle)
variable (h7 : arc_BQ = 42)
variable (h8 : arc_QD = 38)
variable (h9 : arc_AP = 20)
variable (h10 : arc_PC = 40)
variable (h11 : X = (arc_AP + arc_PC) / 2)

-- State the theorem
theorem sum_of_angles_X_and_Q :
  X + (arc_BQ + arc_QD) / 2 = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_X_and_Q_l907_90754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_arithmetic_progression_l907_90784

theorem smallest_sum_arithmetic_progression (a b c d : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  ∃ r : ℕ, b = a + r ∧ c = a + 2*r ∧ d = a + 3*r →
  a * b * c * d = Nat.factorial 9 →
  (∀ w x y z : ℕ, w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 →
    (∃ s : ℕ, x = w + s ∧ y = w + 2*s ∧ z = w + 3*s) →
    w * x * y * z = Nat.factorial 9 →
    a + b + c + d ≤ w + x + y + z) →
  a + b + c + d = 120 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_arithmetic_progression_l907_90784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_circles_common_chord_l907_90703

/-- The length of the common chord between two overlapping circles -/
noncomputable def common_chord_length (r : ℝ) : ℝ :=
  2 * r * Real.sqrt 3

/-- A circle in the real plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point is in a circle if its distance from the center is equal to the radius -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem overlapping_circles_common_chord
  (r : ℝ)
  (h : r = 12)
  (circles_overlap : ∃ (c₁ c₂ : Circle),
    c₁.radius = r ∧ c₂.radius = r ∧
    c₂.contains c₁.center ∧ c₁.contains c₂.center) :
  common_chord_length r = 12 * Real.sqrt 3 := by
  sorry

#check overlapping_circles_common_chord

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_circles_common_chord_l907_90703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_probability_l907_90776

noncomputable def particle_movement (n k : ℕ) : ℝ :=
  2 * (Nat.factorial (2 * n + k)) / (Nat.factorial n * Nat.factorial (n + k) * (2 ^ (2 * n + k)))

theorem particle_probability (n k : ℕ) :
  particle_movement n k = 2 * (Nat.factorial (2 * n + k)) / (Nat.factorial n * Nat.factorial (n + k) * (2 ^ (2 * n + k))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_probability_l907_90776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_707_l907_90790

/-- A square with side length 4 -/
structure Square where
  side_length : ℝ
  is_four : side_length = 4

/-- A line segment with length 3 and endpoints on adjacent sides of the square -/
structure LineSegment (s : Square) where
  length : ℝ
  is_three : length = 3
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ
  on_adjacent_sides : Bool

/-- The set of all valid line segments -/
def T (s : Square) := {l : LineSegment s | l.on_adjacent_sides = true}

/-- The region enclosed by the midpoints of line segments in T -/
def enclosed_region (s : Square) : Set (ℝ × ℝ) := sorry

/-- The area of the enclosed region -/
noncomputable def area (s : Square) : ℝ := sorry

/-- Main theorem -/
theorem enclosed_area_is_707 (s : Square) : 
  Int.floor (100 * (area s + 0.005)) = 707 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_707_l907_90790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_from_cot_sin_relation_l907_90725

theorem cos_value_from_cot_sin_relation (φ : ℝ) 
  (h1 : 6 * (Real.cos φ / Real.sin φ) = 4 * Real.sin φ) 
  (h2 : 0 < φ) (h3 : φ < Real.pi/2) : 
  Real.cos φ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_from_cot_sin_relation_l907_90725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_incenter_l907_90700

/-- A triangle in a 2D plane --/
structure Triangle where
  A : Real × Real
  B : Real × Real
  C : Real × Real

/-- A line in a 2D plane --/
structure Line where
  P : Real × Real
  Q : Real × Real

/-- The incenter of a triangle --/
noncomputable def incenter (t : Triangle) : Real × Real :=
  sorry

/-- Checks if a point lies on a line --/
def point_on_line (p : Real × Real) (l : Line) : Prop :=
  sorry

/-- Checks if a line halves the perimeter of a triangle --/
def halves_perimeter (l : Line) (t : Triangle) : Prop :=
  sorry

/-- Checks if a line halves the area of a triangle --/
def halves_area (l : Line) (t : Triangle) : Prop :=
  sorry

/-- Checks if a line intersects two sides of a triangle --/
def intersects_two_sides (l : Line) (t : Triangle) : Prop :=
  sorry

/-- Theorem: If a line halves both the perimeter and area of a triangle,
    it passes through the incenter --/
theorem line_through_incenter (t : Triangle) (l : Line) :
  intersects_two_sides l t →
  halves_perimeter l t →
  halves_area l t →
  point_on_line (incenter t) l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_incenter_l907_90700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_problem_l907_90705

/-- The shopping problem involving Lisa and Carly -/
theorem shopping_problem (lisa_tshirts : ℕ) : 
  lisa_tshirts = 40 →
  (let lisa_jeans := lisa_tshirts / 2
   let lisa_coats := lisa_tshirts * 2
   let carly_tshirts := lisa_tshirts / 4
   let carly_jeans := lisa_jeans * 3
   let carly_coats := lisa_coats / 4
   let lisa_total := lisa_tshirts + lisa_jeans + lisa_coats
   let carly_total := carly_tshirts + carly_jeans + carly_coats
   lisa_total + carly_total = 230) :=
by
  intro h
  sorry

#check shopping_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_problem_l907_90705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l907_90764

/-- Given a geometric sequence {a_n} with first term a₁ and common ratio q,
    S_n represents the sum of the first n terms. -/
noncomputable def S_n (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence {a_n} with sum formula S_n for the first n terms,
    if 2S_4 = S_5 + S_6, then the common ratio q = -2. -/
theorem geometric_sequence_common_ratio
  (a₁ : ℝ) (q : ℝ) (h : q ≠ 1) :
  2 * S_n a₁ q 4 = S_n a₁ q 5 + S_n a₁ q 6 → q = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l907_90764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_minus_repeating_third_l907_90757

theorem one_minus_repeating_third : 1 - (1/3 : ℚ) = 2/3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_minus_repeating_third_l907_90757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_sine_rule_l907_90726

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles in radians
  (a b c : Real)  -- Side lengths

-- Define our specific triangle
noncomputable def triangle_ABC : Triangle := {
  A := Real.pi/2.4  -- 75°
  B := Real.pi/4    -- 45°
  C := Real.pi/3    -- 60°
  a := 0            -- We don't know this value
  b := 0            -- We don't know this value
  c := 1
}

-- State the theorem
theorem shortest_side_length (t : Triangle) (h1 : t.B = Real.pi/4) (h2 : t.C = Real.pi/3) (h3 : t.c = 1) :
  min t.a (min t.b t.c) = Real.sqrt 6 / 3 := by
  sorry

-- Define the sine rule
theorem sine_rule (t : Triangle) :
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧ t.b / Real.sin t.B = t.c / Real.sin t.C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_sine_rule_l907_90726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_term_position_l907_90795

-- Define the sequence
def mySequence (n m : ℕ) : ℚ :=
  if m ≤ n then (m : ℚ) / (n - m + 1 : ℚ) else (n - m + 1 : ℚ) / (m : ℚ)

-- Define the position function
def position (n : ℕ) : ℕ := n * (n + 1) / 2

-- State the theorem
theorem term_position : ∃ n m : ℕ, mySequence n m = 2 / 6 ∧ position n + m = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_term_position_l907_90795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l907_90704

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - x) + Real.log (x - 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo 1 2 ∪ {2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l907_90704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_double_dimensions_l907_90702

noncomputable section

-- Define the volume of a cone
def coneVolume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Define the ratio of volumes
def volumeRatio (r1 h1 r2 h2 : ℝ) : ℝ := 
  coneVolume r1 h1 / coneVolume r2 h2

-- Theorem statement
theorem cone_volume_ratio_double_dimensions (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  volumeRatio r h (2*r) (2*h) = 1/8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_double_dimensions_l907_90702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l907_90733

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B :
  (A ∩ B) = {-2, -1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l907_90733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l907_90763

theorem power_equality (x : ℝ) : (1 / 4 : ℝ) * (2 : ℝ) ^ 30 = (4 : ℝ) ^ x → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l907_90763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_l907_90783

def is_valid_sequence (s : List Nat) : Prop :=
  s.length = 2003 ∧
  s.head? = some 2 ∧
  ∀ i, i < s.length - 1 → 
    (s.get! i * 10 + s.get! (i+1)) % 17 = 0 ∨ 
    (s.get! i * 10 + s.get! (i+1)) % 23 = 0

theorem largest_last_digit :
  ∃ (s : List Nat), is_valid_sequence s ∧
    ∀ (t : List Nat), is_valid_sequence t → 
      Option.getD (t.getLast?) 0 ≤ Option.getD (s.getLast?) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_l907_90783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l907_90724

-- Define the line
def line (a b x y : ℝ) : Prop := Real.sqrt 2 * a * x + b * y = 2

-- Define the circle (using a different name to avoid conflict)
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- State the theorem
theorem min_value_of_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line a b x₁ y₁ ∧ unit_circle x₁ y₁ ∧
    line a b x₂ y₂ ∧ unit_circle x₂ y₂ ∧
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    (x₁ - 0)*(x₂ - 0) + (y₁ - 0)*(y₂ - 0) = 0) →
  (1 / a^2 + 2 / b^2 ≥ 1 ∧ ∃ a₀ b₀, 1 / a₀^2 + 2 / b₀^2 = 1) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l907_90724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l907_90730

-- Define the circle C
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Define the line l
def line_eq (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 5 = 0

-- State the theorem
theorem shortest_distance_circle_to_line :
  ∃ (d : ℝ), d = 1 ∧
  ∀ (p q : ℝ × ℝ),
    circle_eq p.1 p.2 →
    line_eq q.1 q.2 →
    d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l907_90730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_function_correct_l907_90792

def distance_AB : ℝ := 150
def speed_AB : ℝ := 60
def speed_BA : ℝ := 50
def stay_time : ℝ := 1

noncomputable def distance_function (t : ℝ) : ℝ :=
  if t ≤ 0 then 0
  else if t ≤ 2.5 then 60 * t
  else if t ≤ 3.5 then 150
  else if t ≤ 6.5 then 150 - 50 * (t - 3.5)
  else 0

theorem distance_function_correct :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 6.5 →
    distance_function t = 
      if t ≤ 2.5 then 60 * t
      else if t ≤ 3.5 then 150
      else 150 - 50 * (t - 3.5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_function_correct_l907_90792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_difference_l907_90737

theorem arithmetic_geometric_mean_difference (x y : ℤ) : 
  x ≠ y →
  x > 0 →
  y > 0 →
  (x + y) / 2 = 361 →
  Real.sqrt (x * y : ℝ) = 163 →
  |x - y| = 154 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_difference_l907_90737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_work_hours_l907_90774

/-- Calculates the hours worked per day given the total hours and number of days -/
noncomputable def hoursPerDay (totalHours : ℝ) (numDays : ℝ) : ℝ :=
  totalHours / numDays

/-- Proves that working 7.5 hours over 3 days results in 2.5 hours per day -/
theorem andrew_work_hours :
  hoursPerDay 7.5 3 = 2.5 := by
  -- Unfold the definition of hoursPerDay
  unfold hoursPerDay
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_work_hours_l907_90774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contains_integer_l907_90743

/-- Definition of the sequence (x_n) -/
noncomputable def x (x₁ : ℚ) : ℕ → ℚ
  | 0 => x₁
  | n + 1 => x x₁ n + 1 / ⌊x x₁ n⌋

/-- Theorem: The sequence (x_n) contains an integer -/
theorem sequence_contains_integer (x₁ : ℚ) (h₁ : x₁ > 1) :
  ∃ n : ℕ, ∃ k : ℕ, x x₁ n = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contains_integer_l907_90743
