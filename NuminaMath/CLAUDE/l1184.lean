import Mathlib

namespace sixth_power_sum_l1184_118430

theorem sixth_power_sum (x : ℝ) (hx : x ≠ 0) : x + 1/x = 1 → x^6 + 1/x^6 = 2 := by
  sorry

end sixth_power_sum_l1184_118430


namespace sum_of_two_squares_l1184_118467

theorem sum_of_two_squares (a b : ℝ) : 2 * a^2 + 2 * b^2 = (a + b)^2 + (a - b)^2 := by
  sorry

end sum_of_two_squares_l1184_118467


namespace smallest_sum_of_squares_l1184_118419

theorem smallest_sum_of_squares (x y : ℝ) :
  (∀ a b : ℝ, x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 9 :=
by
  sorry

end smallest_sum_of_squares_l1184_118419


namespace horner_method_for_f_at_3_l1184_118402

/-- Horner's method for a polynomial of degree 4 -/
def horner_method (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₄ * x + a₃) * x + a₂) * x + a₁) * x + a₀)

/-- The polynomial f(x) = 2x⁴ - x³ + 3x² + 7 -/
def f (x : ℝ) : ℝ := 2 * x^4 - x^3 + 3 * x^2 + 7

theorem horner_method_for_f_at_3 :
  horner_method 2 (-1) 3 0 7 3 = f 3 ∧ horner_method 2 (-1) 3 0 7 3 = 54 := by
  sorry

end horner_method_for_f_at_3_l1184_118402


namespace abc_product_is_one_l1184_118452

theorem abc_product_is_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h1 : a + 1 / b^2 = b + 1 / c^2) (h2 : b + 1 / c^2 = c + 1 / a^2) :
  |a * b * c| = 1 := by
sorry

end abc_product_is_one_l1184_118452


namespace monotonic_subsequence_exists_l1184_118420

theorem monotonic_subsequence_exists (a : Fin 10 → ℝ) (h : Function.Injective a) :
  ∃ (i j k l : Fin 10), i < j ∧ j < k ∧ k < l ∧
    ((a i ≤ a j ∧ a j ≤ a k ∧ a k ≤ a l) ∨
     (a i ≥ a j ∧ a j ≥ a k ∧ a k ≥ a l)) := by
  sorry

end monotonic_subsequence_exists_l1184_118420


namespace unique_four_digit_square_l1184_118489

/-- A four-digit number -/
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- Each digit is less than 7 -/
def DigitsLessThan7 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d < 7

/-- The number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

/-- The main theorem -/
theorem unique_four_digit_square (N : ℕ) : 
  FourDigitNumber N ∧ 
  DigitsLessThan7 N ∧ 
  IsPerfectSquare N ∧ 
  IsPerfectSquare (N + 3333) → 
  N = 1156 := by
  sorry

end unique_four_digit_square_l1184_118489


namespace instantaneous_velocity_at_2_l1184_118414

/-- The distance function of a particle moving in a straight line -/
def s (t : ℝ) : ℝ := 3 * t^2 + t

/-- The instantaneous velocity of the particle at time t -/
def v (t : ℝ) : ℝ := 6 * t + 1

theorem instantaneous_velocity_at_2 : v 2 = 13 := by
  sorry

end instantaneous_velocity_at_2_l1184_118414


namespace max_area_rectangle_l1184_118438

/-- The parabola function y = 4 - x^2 --/
def parabola (x : ℝ) : ℝ := 4 - x^2

/-- The area function of the rectangle --/
def area (x : ℝ) : ℝ := 2 * x * (4 - x^2)

/-- The theorem stating the maximum area of the rectangle --/
theorem max_area_rectangle :
  ∃ (x : ℝ), x > 0 ∧ x < 2 ∧
  (∀ (y : ℝ), y > 0 ∧ y < 2 → area x ≥ area y) ∧
  (2 * x = (4 / 3) * Real.sqrt 3) := by
  sorry

#check max_area_rectangle

end max_area_rectangle_l1184_118438


namespace specific_kite_area_l1184_118485

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a kite shape -/
structure Kite where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a square -/
structure Square where
  center : Point
  sideLength : ℝ

/-- Calculates the area of a kite with an internal square -/
def kiteArea (k : Kite) (s : Square) : ℝ :=
  sorry

/-- The theorem stating the area of the specific kite -/
theorem specific_kite_area :
  let k : Kite := {
    v1 := {x := 1, y := 6},
    v2 := {x := 4, y := 7},
    v3 := {x := 7, y := 6},
    v4 := {x := 4, y := 0}
  }
  let s : Square := {
    center := {x := 4, y := 3},
    sideLength := 2
  }
  kiteArea k s = 10 := by
  sorry

end specific_kite_area_l1184_118485


namespace surface_area_of_slice_theorem_l1184_118428

/-- Represents a right prism with isosceles triangular bases -/
structure IsoscelesPrism where
  height : ℝ
  base_length : ℝ
  side_length : ℝ

/-- Calculates the surface area of the sliced off portion of the prism -/
def surface_area_of_slice (prism : IsoscelesPrism) : ℝ :=
  sorry

/-- Theorem stating the surface area of the sliced portion -/
theorem surface_area_of_slice_theorem (prism : IsoscelesPrism) 
  (h1 : prism.height = 10)
  (h2 : prism.base_length = 10)
  (h3 : prism.side_length = 12) :
  surface_area_of_slice prism = 52.25 := by
  sorry

end surface_area_of_slice_theorem_l1184_118428


namespace lawnmower_initial_price_l1184_118497

/-- Proves that the initial price of a lawnmower was $100 given specific depreciation rates and final value -/
theorem lawnmower_initial_price (initial_price : ℝ) : 
  let price_after_six_months := initial_price * 0.75
  let final_price := price_after_six_months * 0.8
  final_price = 60 →
  initial_price = 100 := by
  sorry

end lawnmower_initial_price_l1184_118497


namespace largest_integer_less_than_80_remainder_3_mod_5_l1184_118450

theorem largest_integer_less_than_80_remainder_3_mod_5 : ∃ n : ℕ, 
  (n < 80 ∧ n % 5 = 3 ∧ ∀ m : ℕ, m < 80 ∧ m % 5 = 3 → m ≤ n) ∧ n = 78 := by
  sorry

end largest_integer_less_than_80_remainder_3_mod_5_l1184_118450


namespace unique_number_with_triple_property_l1184_118433

/-- A six-digit number with 1 as its leftmost digit -/
def sixDigitNumberStartingWith1 (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 1

/-- Function to move the leftmost digit to the rightmost position -/
def moveFirstDigitToEnd (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

/-- The main theorem statement -/
theorem unique_number_with_triple_property :
  ∃! n : ℕ, sixDigitNumberStartingWith1 n ∧ moveFirstDigitToEnd n = 3 * n :=
by
  -- The proof would go here
  sorry

end unique_number_with_triple_property_l1184_118433


namespace square_difference_l1184_118437

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 80) (h2 : x * y = 12) : 
  (x - y)^2 = 32 := by
  sorry

end square_difference_l1184_118437


namespace solution_set_inequality_1_solution_set_inequality_2_l1184_118453

-- Problem 1
theorem solution_set_inequality_1 (x : ℝ) :
  (x + 2) / (x - 3) ≤ 0 ↔ -2 ≤ x ∧ x < 3 :=
sorry

-- Problem 2
theorem solution_set_inequality_2 (x a : ℝ) :
  (x + a) * (x - 1) > 0 ↔
    (a > -1 ∧ (x < -a ∨ x > 1)) ∨
    (a = -1 ∧ x ≠ 1) ∨
    (a < -1 ∧ (x < 1 ∨ x > -a)) :=
sorry

end solution_set_inequality_1_solution_set_inequality_2_l1184_118453


namespace sin_derivative_bound_and_inequality_range_l1184_118476

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem sin_derivative_bound_and_inequality_range :
  (∀ x > 0, (deriv f) x > 1 - x^2 / 2) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Ioo 0 (Real.pi / 2), f x + f x / (deriv f) x > a * x) → a ≤ 2) := by
  sorry

end sin_derivative_bound_and_inequality_range_l1184_118476


namespace shaded_cubes_count_l1184_118477

/-- Represents a 4x4x4 cube with a specific shading pattern -/
structure ShadedCube where
  /-- Total number of smaller cubes -/
  total_cubes : Nat
  /-- Number of cubes per face -/
  cubes_per_face : Nat
  /-- Number of shaded cubes on one face -/
  shaded_per_face : Nat
  /-- Number of corner cubes -/
  corner_cubes : Nat
  /-- Number of edge cubes -/
  edge_cubes : Nat
  /-- Condition: The cube is 4x4x4 -/
  is_4x4x4 : total_cubes = 64 ∧ cubes_per_face = 16
  /-- Condition: Shading pattern on one face -/
  shading_pattern : shaded_per_face = 9
  /-- Condition: Number of corners and edges -/
  cube_structure : corner_cubes = 8 ∧ edge_cubes = 12

/-- Theorem: The number of shaded cubes in the given 4x4x4 cube is 33 -/
theorem shaded_cubes_count (c : ShadedCube) : 
  c.corner_cubes + c.edge_cubes + (3 * c.shaded_per_face - c.corner_cubes - c.edge_cubes) = 33 := by
  sorry

end shaded_cubes_count_l1184_118477


namespace range_of_a_l1184_118492

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def g (x : ℝ) : ℝ := -x^2 + 2*x + 1

theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 (Real.exp 1), ∃ x₂ ∈ Set.Icc 0 3, f a x₁ = g x₂) →
  a ∈ Set.Icc (-1 / Real.exp 1) (3 / Real.exp 1) :=
sorry

end range_of_a_l1184_118492


namespace min_value_of_f_l1184_118427

theorem min_value_of_f (x : ℝ) (hx : x > 0) : x + 1/x - 2 ≥ 0 ∧ (x + 1/x - 2 = 0 ↔ x = 1) :=
sorry

end min_value_of_f_l1184_118427


namespace expression_positivity_l1184_118472

theorem expression_positivity (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end expression_positivity_l1184_118472


namespace largest_value_u3_plus_v3_l1184_118424

theorem largest_value_u3_plus_v3 (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 3)
  (h2 : Complex.abs (u^2 + v^2) = 10) :
  Complex.abs (u^3 + v^3) = 31.5 := by
  sorry

end largest_value_u3_plus_v3_l1184_118424


namespace dani_pants_after_five_years_l1184_118412

/-- The number of pants Dani will have after a given number of years -/
def total_pants (initial_pants : ℕ) (pairs_per_year : ℕ) (years : ℕ) : ℕ :=
  initial_pants + pairs_per_year * 2 * years

/-- Theorem stating that Dani will have 90 pants after 5 years -/
theorem dani_pants_after_five_years :
  total_pants 50 4 5 = 90 := by
  sorry

end dani_pants_after_five_years_l1184_118412


namespace geometric_sequence_sum_l1184_118440

/-- Given a geometric sequence {a_n} with the specified conditions, 
    prove that the sum of the 6th, 7th, and 8th terms is 32. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 1 + a 2 + a 3 = 1 →                             -- first given condition
  a 2 + a 3 + a 4 = 2 →                             -- second given condition
  a 6 + a 7 + a 8 = 32 :=                           -- conclusion to prove
by sorry

end geometric_sequence_sum_l1184_118440


namespace company_employees_l1184_118491

/-- 
If a company had 15% more employees in December than in January,
and it had 450 employees in December, then it had 391 employees in January.
-/
theorem company_employees (december_employees : ℕ) (january_employees : ℕ) : 
  december_employees = 450 → 
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 391 := by
sorry

end company_employees_l1184_118491


namespace stating_max_areas_formula_l1184_118447

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii_count : ℕ := 3 * n
  secant_lines : ℕ := 2
  h_positive : n > 0

/-- 
Calculates the maximum number of non-overlapping areas in a divided disk 
-/
def max_areas (disk : DividedDisk) : ℕ := 4 * disk.n + 1

/-- 
Theorem stating that the maximum number of non-overlapping areas 
in a divided disk is 4n + 1 
-/
theorem max_areas_formula (disk : DividedDisk) : 
  max_areas disk = 4 * disk.n + 1 := by sorry

end stating_max_areas_formula_l1184_118447


namespace absolute_value_equation_unique_solution_l1184_118473

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 4| := by
sorry

end absolute_value_equation_unique_solution_l1184_118473


namespace certain_number_solution_l1184_118404

theorem certain_number_solution (x : ℝ) : 
  8 * 5.4 - (x * 10) / 1.2 = 31.000000000000004 → x = 1.464 := by
  sorry

end certain_number_solution_l1184_118404


namespace subset_count_l1184_118441

theorem subset_count : ℕ := by
  -- Define the universal set U
  let U : Finset ℕ := {1, 2, 3, 4, 5, 6}
  
  -- Define the required subset A
  let A : Finset ℕ := {1, 2, 3}
  
  -- Define the count of subsets X such that A ⊆ X ⊆ U
  let count := Finset.filter (fun X => A ⊆ X) U.powerset |>.card
  
  -- Assert that this count is equal to 8
  have h : count = 8 := by sorry
  
  -- Return the result
  exact 8

end subset_count_l1184_118441


namespace show_end_time_l1184_118418

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents a TV show -/
structure TVShow where
  start_time : Time
  end_time : Time
  weekday_only : Bool

def total_watch_time (s : TVShow) (days_watched : Nat) : Nat :=
  days_watched * (s.end_time.hour * 60 + s.end_time.minute - s.start_time.hour * 60 - s.start_time.minute)

theorem show_end_time (s : TVShow) 
  (h1 : s.start_time = ⟨14, 0, by norm_num, by norm_num⟩)
  (h2 : s.weekday_only = true)
  (h3 : total_watch_time s 4 = 120) :
  s.end_time = ⟨14, 30, by norm_num, by norm_num⟩ := by
  sorry

end show_end_time_l1184_118418


namespace derricks_yard_length_l1184_118496

theorem derricks_yard_length :
  ∀ (derrick_length alex_length brianne_length : ℝ),
    brianne_length = 30 →
    alex_length = derrick_length / 2 →
    brianne_length = 6 * alex_length →
    derrick_length = 10 := by
  sorry

end derricks_yard_length_l1184_118496


namespace q_satisfies_conditions_l1184_118417

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (6/5) * x^2 - (4/5) * x + 8/5

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-2) = 8 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

end q_satisfies_conditions_l1184_118417


namespace penguin_colony_growth_l1184_118449

/-- Represents the penguin colony growth over three years -/
structure PenguinColony where
  initial_size : ℕ
  first_year_growth : ℕ → ℕ
  second_year_growth : ℕ → ℕ
  third_year_gain : ℕ
  current_size : ℕ
  fish_per_penguin : ℚ
  initial_fish_caught : ℕ

/-- Theorem stating the number of penguins gained in the third year -/
theorem penguin_colony_growth (colony : PenguinColony) : colony.third_year_gain = 129 :=
  by
  have h1 : colony.initial_size = 158 := by sorry
  have h2 : colony.first_year_growth colony.initial_size = 2 * colony.initial_size := by sorry
  have h3 : colony.second_year_growth (colony.first_year_growth colony.initial_size) = 
            3 * (colony.first_year_growth colony.initial_size) := by sorry
  have h4 : colony.current_size = 1077 := by sorry
  have h5 : colony.fish_per_penguin = 3/2 := by sorry
  have h6 : colony.initial_fish_caught = 237 := by sorry
  have h7 : colony.initial_size * colony.fish_per_penguin = colony.initial_fish_caught := by sorry
  sorry

end penguin_colony_growth_l1184_118449


namespace f_geq_4_iff_valid_a_range_f_3_geq_4_l1184_118405

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 3/a| + |x - a|

def valid_a_range (a : ℝ) : Prop :=
  a ∈ Set.Iic (-3) ∪ Set.Icc (-1) 0 ∪ Set.Ioc 0 1 ∪ Set.Ici 3

theorem f_geq_4_iff_valid_a_range (a : ℝ) (h : a ≠ 0) :
  (∀ x, f a x ≥ 4) ↔ valid_a_range a := by sorry

theorem f_3_geq_4 (a : ℝ) (h : a ≠ 0) : f a 3 ≥ 4 := by sorry

end f_geq_4_iff_valid_a_range_f_3_geq_4_l1184_118405


namespace simplify_expression_l1184_118463

theorem simplify_expression (a : ℝ) : ((4 * a + 6) - 7 * a) / 3 = -a + 2 := by
  sorry

end simplify_expression_l1184_118463


namespace polynomial_divisibility_l1184_118439

theorem polynomial_divisibility (m : ℚ) :
  (∀ x, (x^4 - 5*x^2 + 4*x - m) % (2*x + 1) = 0) → m = -51/16 := by
  sorry

end polynomial_divisibility_l1184_118439


namespace no_real_roots_iff_m_gt_one_l1184_118432

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Theorem statement
theorem no_real_roots_iff_m_gt_one (m : ℝ) :
  (∀ x : ℝ, quadratic x m ≠ 0) ↔ m > 1 :=
by sorry

end no_real_roots_iff_m_gt_one_l1184_118432


namespace large_painting_area_is_150_l1184_118413

/-- Represents Davonte's art collection --/
structure ArtCollection where
  square_paintings : Nat
  small_paintings : Nat
  large_painting : Nat
  square_side : Nat
  small_width : Nat
  small_height : Nat
  total_area : Nat

/-- Calculates the area of the large painting in Davonte's collection --/
def large_painting_area (collection : ArtCollection) : Nat :=
  collection.total_area -
  (collection.square_paintings * collection.square_side * collection.square_side +
   collection.small_paintings * collection.small_width * collection.small_height)

/-- Theorem stating that the area of the large painting is 150 square feet --/
theorem large_painting_area_is_150 (collection : ArtCollection)
  (h1 : collection.square_paintings = 3)
  (h2 : collection.small_paintings = 4)
  (h3 : collection.square_side = 6)
  (h4 : collection.small_width = 2)
  (h5 : collection.small_height = 3)
  (h6 : collection.total_area = 282) :
  large_painting_area collection = 150 := by
  sorry

#eval large_painting_area { square_paintings := 3, small_paintings := 4, large_painting := 1,
                            square_side := 6, small_width := 2, small_height := 3, total_area := 282 }

end large_painting_area_is_150_l1184_118413


namespace rational_function_property_l1184_118484

theorem rational_function_property (f : ℚ → ℝ) 
  (add_prop : ∀ x y : ℚ, f (x + y) = f x + f y)
  (mul_prop : ∀ x y : ℚ, f (x * y) = f x * f y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = 0) := by
sorry

end rational_function_property_l1184_118484


namespace sqrt_sum_squares_eq_sum_l1184_118495

theorem sqrt_sum_squares_eq_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a*b + a*c + b*c = 0 ∧ a + b + c ≥ 0 := by
  sorry

end sqrt_sum_squares_eq_sum_l1184_118495


namespace banana_production_theorem_l1184_118460

/-- The total banana production from two islands -/
def total_banana_production (jakies_production : ℕ) (nearby_production : ℕ) : ℕ :=
  jakies_production + nearby_production

/-- Theorem stating the total banana production from Jakies Island and a nearby island -/
theorem banana_production_theorem (nearby_production : ℕ) 
  (h1 : nearby_production = 9000)
  (h2 : ∃ (jakies_production : ℕ), jakies_production = 10 * nearby_production) :
  ∃ (total_production : ℕ), total_production = 99000 ∧ 
  total_production = total_banana_production (10 * nearby_production) nearby_production :=
by
  sorry


end banana_production_theorem_l1184_118460


namespace max_attendance_days_l1184_118409

structure Day where
  name : String
  dan_available : Bool
  eve_available : Bool
  frank_available : Bool
  grace_available : Bool

def schedule : List Day := [
  { name := "Monday",    dan_available := false, eve_available := true,  frank_available := false, grace_available := true  },
  { name := "Tuesday",   dan_available := true,  eve_available := false, frank_available := false, grace_available := true  },
  { name := "Wednesday", dan_available := false, eve_available := true,  frank_available := true,  grace_available := false },
  { name := "Thursday",  dan_available := true,  eve_available := false, frank_available := true,  grace_available := false },
  { name := "Friday",    dan_available := false, eve_available := false, frank_available := false, grace_available := true  }
]

def count_available (day : Day) : Nat :=
  (if day.dan_available then 1 else 0) +
  (if day.eve_available then 1 else 0) +
  (if day.frank_available then 1 else 0) +
  (if day.grace_available then 1 else 0)

def max_available (schedule : List Day) : Nat :=
  schedule.map count_available |>.maximum?
    |>.getD 0

theorem max_attendance_days (schedule : List Day) :
  max_available schedule = 2 ∧
  schedule.filter (fun day => count_available day = 2) =
    schedule.filter (fun day => day.name ∈ ["Monday", "Tuesday", "Wednesday", "Thursday"]) :=
by sorry

end max_attendance_days_l1184_118409


namespace cantor_is_founder_l1184_118479

/-- Represents a mathematician -/
inductive Mathematician
  | Gauss
  | Dedekind
  | Weierstrass
  | Cantor

/-- Represents the founder of modern set theory -/
def founder_of_modern_set_theory : Mathematician := Mathematician.Cantor

/-- Theorem stating that Cantor is the founder of modern set theory -/
theorem cantor_is_founder : 
  founder_of_modern_set_theory = Mathematician.Cantor := by sorry

end cantor_is_founder_l1184_118479


namespace five_balls_three_boxes_l1184_118443

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 243 := by
  sorry

end five_balls_three_boxes_l1184_118443


namespace gcd_840_1764_l1184_118444

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l1184_118444


namespace percentage_both_languages_l1184_118415

/-- Represents the number of diplomats speaking both French and Russian -/
def both_languages (total french not_russian neither : ℕ) : ℕ :=
  french + (total - not_russian) - (total - neither)

/-- Theorem stating the percentage of diplomats speaking both French and Russian -/
theorem percentage_both_languages :
  let total := 100
  let french := 22
  let not_russian := 32
  let neither := 20
  (both_languages total french not_russian neither : ℚ) / total * 100 = 10 := by
  sorry

end percentage_both_languages_l1184_118415


namespace max_ratio_squared_l1184_118469

theorem max_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  (∃ (ρ : ℝ), ∀ (x y : ℝ), 
    (0 ≤ x ∧ x < a) → 
    (0 ≤ y ∧ y < b) → 
    (a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a + x)^2 + (b - y)^2) →
    (a / b)^2 ≤ ρ^2 ∧
    ρ^2 = 4/3) :=
sorry

end max_ratio_squared_l1184_118469


namespace find_a_l1184_118490

def U (a : ℤ) : Set ℤ := {2, 4, a^2 - a + 1}

def A (a : ℤ) : Set ℤ := {a+4, 4}

def complement_A (a : ℤ) : Set ℤ := {7}

theorem find_a : ∃ a : ℤ, 
  (U a = {2, 4, a^2 - a + 1}) ∧ 
  (A a = {a+4, 4}) ∧ 
  (complement_A a = {7}) ∧
  (Set.inter (A a) (complement_A a) = ∅) ∧
  (Set.union (A a) (complement_A a) = U a) ∧
  (a = -2) := by sorry

end find_a_l1184_118490


namespace valid_rectangles_count_l1184_118478

/-- Represents a square array of dots -/
structure DotArray where
  size : ℕ

/-- Represents a rectangle in the dot array -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Returns true if the rectangle has an area greater than 1 -/
def Rectangle.areaGreaterThanOne (r : Rectangle) : Prop :=
  r.width * r.height > 1

/-- Returns the number of valid rectangles in the dot array -/
def countValidRectangles (arr : DotArray) : ℕ :=
  sorry

theorem valid_rectangles_count (arr : DotArray) :
  arr.size = 5 → countValidRectangles arr = 84 := by
  sorry

end valid_rectangles_count_l1184_118478


namespace find_X_l1184_118446

theorem find_X : ∃ X : ℕ, X = 555 * 465 * (3 * (555 - 465)) + (555 - 465)^2 ∧ X = 69688350 := by
  sorry

end find_X_l1184_118446


namespace smallest_undefined_value_l1184_118466

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y < (1/4 : ℝ), (10*y^2 - 90*y + 20) ≠ 0) ∧ 
  (10*(1/4 : ℝ)^2 - 90*(1/4 : ℝ) + 20 = 0) := by
  sorry

end smallest_undefined_value_l1184_118466


namespace opposite_of_negative_three_l1184_118421

theorem opposite_of_negative_three : 
  ∃ y : ℤ, ((-3 : ℤ) + y = 0) ∧ y = 3 := by
  sorry

end opposite_of_negative_three_l1184_118421


namespace cubic_system_solution_l1184_118416

theorem cubic_system_solution :
  ∃ (x y z : ℝ), 
    x + y + z = 1 ∧
    x^2 + y^2 + z^2 = 1 ∧
    x^3 + y^3 + z^3 = 89/125 ∧
    ((x = 2/5 ∧ y = (3 + Real.sqrt 33)/10 ∧ z = (3 - Real.sqrt 33)/10) ∨
     (x = 2/5 ∧ y = (3 - Real.sqrt 33)/10 ∧ z = (3 + Real.sqrt 33)/10) ∨
     (x = (3 + Real.sqrt 33)/10 ∧ y = 2/5 ∧ z = (3 - Real.sqrt 33)/10) ∨
     (x = (3 + Real.sqrt 33)/10 ∧ y = (3 - Real.sqrt 33)/10 ∧ z = 2/5) ∨
     (x = (3 - Real.sqrt 33)/10 ∧ y = 2/5 ∧ z = (3 + Real.sqrt 33)/10) ∨
     (x = (3 - Real.sqrt 33)/10 ∧ y = (3 + Real.sqrt 33)/10 ∧ z = 2/5)) :=
by
  sorry


end cubic_system_solution_l1184_118416


namespace number_and_square_sum_l1184_118483

theorem number_and_square_sum (x : ℝ) : x + x^2 = 306 → x = 17 := by
  sorry

end number_and_square_sum_l1184_118483


namespace other_color_counts_l1184_118401

def total_students : ℕ := 800

def blue_shirt_percent : ℚ := 45/100
def red_shirt_percent : ℚ := 23/100
def green_shirt_percent : ℚ := 15/100

def black_pants_percent : ℚ := 30/100
def khaki_pants_percent : ℚ := 25/100
def jeans_percent : ℚ := 10/100

def white_shoes_percent : ℚ := 40/100
def black_shoes_percent : ℚ := 20/100
def brown_shoes_percent : ℚ := 15/100

theorem other_color_counts :
  let other_shirt_count := total_students - (blue_shirt_percent + red_shirt_percent + green_shirt_percent) * total_students
  let other_pants_count := total_students - (black_pants_percent + khaki_pants_percent + jeans_percent) * total_students
  let other_shoes_count := total_students - (white_shoes_percent + black_shoes_percent + brown_shoes_percent) * total_students
  (other_shirt_count : ℚ) = 136 ∧ (other_pants_count : ℚ) = 280 ∧ (other_shoes_count : ℚ) = 200 :=
by sorry

end other_color_counts_l1184_118401


namespace binomial_coefficient_19_10_l1184_118411

theorem binomial_coefficient_19_10 : 
  (Nat.choose 17 7 = 19448) → (Nat.choose 17 9 = 24310) → (Nat.choose 19 10 = 87516) := by
  sorry

end binomial_coefficient_19_10_l1184_118411


namespace exists_valid_grid_l1184_118499

-- Define a 5x5 grid of integers (0 or 1)
def Grid := Fin 5 → Fin 5 → Fin 2

-- Function to check if a number is divisible by 3
def divisible_by_three (n : ℕ) : Prop := ∃ k, n = 3 * k

-- Function to sum a 2x2 subgrid
def sum_subgrid (g : Grid) (i j : Fin 4) : ℕ :=
  (g i j).val + (g i (j + 1)).val + (g (i + 1) j).val + (g (i + 1) (j + 1)).val

-- Theorem statement
theorem exists_valid_grid : ∃ (g : Grid),
  (∀ i j : Fin 4, divisible_by_three (sum_subgrid g i j)) ∧
  (∃ i j : Fin 5, g i j = 0) ∧
  (∃ i j : Fin 5, g i j = 1) :=
sorry

end exists_valid_grid_l1184_118499


namespace acute_angle_inequality_l1184_118436

theorem acute_angle_inequality (α : Real) (h : 0 < α ∧ α < π / 2) :
  α < (Real.sin α + Real.tan α) / 2 := by
  sorry

end acute_angle_inequality_l1184_118436


namespace hagrid_divisible_by_three_l1184_118448

def HAGRID (H A G R I D : ℕ) : ℕ := 100000*H + 10000*A + 1000*G + 100*R + 10*I + D

theorem hagrid_divisible_by_three 
  (H A G R I D : ℕ) 
  (h_distinct : H ≠ A ∧ H ≠ G ∧ H ≠ R ∧ H ≠ I ∧ H ≠ D ∧ 
                A ≠ G ∧ A ≠ R ∧ A ≠ I ∧ A ≠ D ∧ 
                G ≠ R ∧ G ≠ I ∧ G ≠ D ∧ 
                R ≠ I ∧ R ≠ D ∧ 
                I ≠ D)
  (h_range : H < 10 ∧ A < 10 ∧ G < 10 ∧ R < 10 ∧ I < 10 ∧ D < 10) : 
  3 ∣ (HAGRID H A G R I D * H * A * G * R * I * D) :=
sorry

end hagrid_divisible_by_three_l1184_118448


namespace pascal_triangle_interior_sum_l1184_118481

/-- Sum of interior numbers in the n-th row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The problem statement -/
theorem pascal_triangle_interior_sum :
  interior_sum 6 = 30 →
  interior_sum 8 = 126 := by
sorry

end pascal_triangle_interior_sum_l1184_118481


namespace no_intersection_point_l1184_118494

theorem no_intersection_point :
  ¬ ∃ (x y : ℝ), 
    (3 * x + 4 * y - 12 = 0) ∧ 
    (5 * x - 4 * y - 10 = 0) ∧ 
    (x = 3) ∧ 
    (y = -1/2) := by
  sorry

end no_intersection_point_l1184_118494


namespace total_bikes_l1184_118451

theorem total_bikes (jungkook_bikes : ℕ) (yoongi_bikes : ℕ) 
  (h1 : jungkook_bikes = 3) (h2 : yoongi_bikes = 4) : 
  jungkook_bikes + yoongi_bikes = 7 := by
  sorry

end total_bikes_l1184_118451


namespace female_grade_one_jiu_is_set_l1184_118498

-- Define the universe of students
def Student : Type := sorry

-- Define the property of being female
def is_female : Student → Prop := sorry

-- Define the property of being in grade one of Jiu Middle School
def is_grade_one_jiu : Student → Prop := sorry

-- Define our set
def female_grade_one_jiu : Set Student :=
  {s : Student | is_female s ∧ is_grade_one_jiu s}

-- Theorem stating that female_grade_one_jiu is a well-defined set
theorem female_grade_one_jiu_is_set :
  ∀ (s : Student), Decidable (s ∈ female_grade_one_jiu) :=
sorry

end female_grade_one_jiu_is_set_l1184_118498


namespace no_real_solutions_l1184_118410

theorem no_real_solutions :
  (¬ ∃ x : ℝ, Real.sqrt (x + 1) - Real.sqrt (x - 1) = 0) ∧
  (¬ ∃ x : ℝ, Real.sqrt x - Real.sqrt (x - Real.sqrt (1 - x)) = 1) := by
  sorry

end no_real_solutions_l1184_118410


namespace set_operations_l1184_118482

open Set

-- Define the universe set U
def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}

-- Define sets A, B, and C
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ A) ∩ (U \ B) = {3}) ∧
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 5, 6, 7, 8, 9, 10}) ∧
  ((A ∩ B) ∩ C = ∅) ∧
  ((A ∪ B) ∩ C = {5, 7}) := by
sorry


end set_operations_l1184_118482


namespace price_adjustment_l1184_118470

theorem price_adjustment (original_price : ℝ) (original_price_pos : 0 < original_price) : 
  let increased_price := original_price * (1 + 0.25)
  let decrease_percentage := (increased_price - original_price) / increased_price
  decrease_percentage = 0.20 := by
  sorry

end price_adjustment_l1184_118470


namespace geometric_sequence_ratio_l1184_118456

/-- Given a geometric sequence with common ratio 2, prove that (2a₁ + a₂) / (2a₃ + a₄) = 1/4 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →
  (2 * a 1 + a 2) / (2 * a 3 + a 4) = 1 / 4 := by
  sorry

end geometric_sequence_ratio_l1184_118456


namespace find_X_l1184_118458

theorem find_X : ∃ X : ℝ, 
  1.5 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1200.0000000000002 ∧ 
  X = 0.3 := by
  sorry

end find_X_l1184_118458


namespace woodwind_to_brass_ratio_l1184_118407

/-- Represents the composition of a marching band -/
structure MarchingBand where
  total : ℕ
  percussion : ℕ
  woodwind : ℕ
  brass : ℕ

/-- Checks if the marching band satisfies the given conditions -/
def validBand (band : MarchingBand) : Prop :=
  band.total = 110 ∧
  band.percussion = 4 * band.woodwind ∧
  band.brass = 10 ∧
  band.total = band.percussion + band.woodwind + band.brass

/-- Theorem stating the ratio of woodwind to brass players -/
theorem woodwind_to_brass_ratio (band : MarchingBand) 
  (h : validBand band) : 
  band.woodwind = 2 * band.brass :=
sorry

end woodwind_to_brass_ratio_l1184_118407


namespace calculate_total_profit_total_profit_is_150000_l1184_118462

/-- Calculates the total profit given investment ratios and B's profit -/
theorem calculate_total_profit (a_c_ratio : Rat) (a_b_ratio : Rat) (b_profit : ℕ) : ℕ :=
  let a_c_ratio := 2/1
  let a_b_ratio := 2/3
  let b_profit := 75000
  2 * b_profit

theorem total_profit_is_150000 : 
  calculate_total_profit (2/1) (2/3) 75000 = 150000 := by
  sorry

end calculate_total_profit_total_profit_is_150000_l1184_118462


namespace no_meetings_before_return_l1184_118403

/-- The number of times two boys meet on a circular track before returning to their starting point -/
def number_of_meetings (circumference : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℕ :=
  sorry

theorem no_meetings_before_return :
  let circumference : ℝ := 120
  let speed1 : ℝ := 6
  let speed2 : ℝ := 10
  number_of_meetings circumference speed1 speed2 = 0 :=
by sorry

end no_meetings_before_return_l1184_118403


namespace kyle_paper_delivery_l1184_118455

/-- The number of papers Kyle delivers each week -/
def weekly_papers (
  regular_houses : ℕ
  ) (sunday_skip : ℕ) (sunday_extra : ℕ) : ℕ :=
  (6 * regular_houses) + (regular_houses - sunday_skip + sunday_extra)

theorem kyle_paper_delivery :
  weekly_papers 100 10 30 = 720 := by
  sorry

end kyle_paper_delivery_l1184_118455


namespace green_minus_blue_equals_twenty_l1184_118426

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green
  | Red

/-- Represents the distribution of disks in the bag -/
structure DiskDistribution where
  blue : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ

/-- The total number of disks in the bag -/
def totalDisks : ℕ := 108

/-- The ratio of blue:yellow:green:red disks -/
def colorRatio : DiskDistribution :=
  { blue := 3, yellow := 7, green := 8, red := 9 }

/-- The sum of all parts in the ratio -/
def totalRatioParts : ℕ :=
  colorRatio.blue + colorRatio.yellow + colorRatio.green + colorRatio.red

/-- Calculates the actual distribution of disks based on the ratio and total number of disks -/
def actualDistribution : DiskDistribution :=
  let disksPerPart := totalDisks / totalRatioParts
  { blue := colorRatio.blue * disksPerPart,
    yellow := colorRatio.yellow * disksPerPart,
    green := colorRatio.green * disksPerPart,
    red := colorRatio.red * disksPerPart }

/-- Theorem: There are 20 more green disks than blue disks in the bag -/
theorem green_minus_blue_equals_twenty :
  actualDistribution.green - actualDistribution.blue = 20 := by
  sorry

end green_minus_blue_equals_twenty_l1184_118426


namespace regular_polygon_sides_l1184_118464

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) :
  n > 2 →
  exterior_angle = 40 * Real.pi / 180 →
  exterior_angle = (2 * Real.pi) / n →
  n = 9 := by
sorry

end regular_polygon_sides_l1184_118464


namespace min_sum_squares_on_parabola_l1184_118431

/-- The parabola equation y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line passing through P(4, 0) and (x, y) -/
def line_through_P (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x - 4)

/-- Theorem: The minimum value of y₁² + y₂² is 32 for points on the parabola
    intersected by a line through P(4, 0) -/
theorem min_sum_squares_on_parabola (x₁ y₁ x₂ y₂ : ℝ) :
  parabola x₁ y₁ →
  parabola x₂ y₂ →
  line_through_P x₁ y₁ →
  line_through_P x₂ y₂ →
  y₁^2 + y₂^2 ≥ 32 :=
by sorry

end min_sum_squares_on_parabola_l1184_118431


namespace exponent_multiplication_l1184_118475

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l1184_118475


namespace probability_of_double_l1184_118461

/-- Represents a domino with two ends --/
structure Domino :=
  (end1 : Nat)
  (end2 : Nat)

/-- A standard set of dominoes with numbers from 0 to 6 --/
def StandardDominoSet : Set Domino :=
  {d : Domino | d.end1 ≤ 6 ∧ d.end2 ≤ 6}

/-- Predicate for a double domino --/
def IsDouble (d : Domino) : Prop :=
  d.end1 = d.end2

/-- The total number of dominoes in a standard set --/
def TotalDominoes : Nat := 28

/-- The number of doubles in a standard set --/
def NumberOfDoubles : Nat := 7

theorem probability_of_double :
  (NumberOfDoubles : ℚ) / (TotalDominoes : ℚ) = 1 / 4 := by
  sorry


end probability_of_double_l1184_118461


namespace min_roots_count_l1184_118488

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem min_roots_count
  (f : ℝ → ℝ)
  (h1 : is_symmetric_about f 2)
  (h2 : is_symmetric_about f 7)
  (h3 : f 0 = 0) :
  ∃ N : ℕ, N ≥ 401 ∧
  (∀ m : ℕ, (∃ S : Finset ℝ, S.card = m ∧
    (∀ x ∈ S, -1000 ≤ x ∧ x ≤ 1000 ∧ f x = 0)) →
    m ≤ N) :=
  sorry

end min_roots_count_l1184_118488


namespace set_operations_l1184_118435

def A (x : ℝ) : Set ℝ := {0, |x|}
def B : Set ℝ := {1, 0, -1}

theorem set_operations (x : ℝ) (h : A x ⊆ B) :
  (A x ∩ B = {0, 1}) ∧
  (A x ∪ B = {-1, 0, 1}) ∧
  (B \ A x = {-1}) := by
sorry

end set_operations_l1184_118435


namespace class_average_theorem_l1184_118474

theorem class_average_theorem (total_students : ℕ) (students_without_two : ℕ) 
  (avg_without_two : ℚ) (score1 : ℕ) (score2 : ℕ) :
  total_students = students_without_two + 2 →
  (students_without_two : ℚ) * avg_without_two + score1 + score2 = total_students * 80 :=
by
  sorry

#check class_average_theorem 40 38 79 98 100

end class_average_theorem_l1184_118474


namespace exponent_multiplication_l1184_118429

theorem exponent_multiplication (a b : ℝ) : -a^2 * 2*a^4*b = -2*a^6*b := by
  sorry

end exponent_multiplication_l1184_118429


namespace trigonometric_identity_l1184_118400

theorem trigonometric_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end trigonometric_identity_l1184_118400


namespace power_equation_solution_l1184_118493

theorem power_equation_solution (m : ℕ) : 5^m = 5 * 25^2 * 125^3 → m = 14 := by
  sorry

end power_equation_solution_l1184_118493


namespace complex_real_part_l1184_118442

theorem complex_real_part (z : ℂ) (h : (z^2 + z).im = 0) : z.re = -1/2 := by
  sorry

end complex_real_part_l1184_118442


namespace green_face_box_dimensions_l1184_118423

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given dimensions satisfy the green face condition -/
def satisfiesGreenFaceCondition (dim : BoxDimensions) : Prop :=
  3 * ((dim.a - 2) * (dim.b - 2) * (dim.c - 2)) = dim.a * dim.b * dim.c

/-- List of valid box dimensions -/
def validDimensions : List BoxDimensions := [
  ⟨7, 30, 4⟩, ⟨8, 18, 4⟩, ⟨9, 14, 4⟩, ⟨10, 12, 4⟩,
  ⟨5, 27, 5⟩, ⟨6, 12, 5⟩, ⟨7, 9, 5⟩, ⟨6, 8, 6⟩
]

theorem green_face_box_dimensions :
  ∀ dim : BoxDimensions,
    satisfiesGreenFaceCondition dim ↔ dim ∈ validDimensions :=
by sorry

end green_face_box_dimensions_l1184_118423


namespace floor_sqrt_50_l1184_118471

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end floor_sqrt_50_l1184_118471


namespace inscribed_square_area_l1184_118422

theorem inscribed_square_area (x y : ℝ) (h1 : x = 18) (h2 : y = 30) :
  let s := Real.sqrt ((x * y) / (x + y))
  s ^ 2 = 540 := by sorry

end inscribed_square_area_l1184_118422


namespace line_passes_through_fixed_point_line_equation_with_equal_intercepts_l1184_118445

-- Define the line equation
def line_equation (m x y : ℝ) : Prop :=
  (m + 2) * x - (m + 1) * y - 3 * m - 7 = 0

-- Theorem 1: The line passes through the point (4, 1) for all real m
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m 4 1 :=
sorry

-- Theorem 2: When x and y intercepts are equal, the line equation becomes x+y-5=0
theorem line_equation_with_equal_intercepts :
  ∀ m : ℝ, 
    (∃ k : ℝ, k ≠ 0 ∧ line_equation m k 0 ∧ line_equation m 0 (-k)) →
    ∃ c : ℝ, ∀ x y : ℝ, line_equation m x y ↔ x + y - 5 = 0 :=
sorry

end line_passes_through_fixed_point_line_equation_with_equal_intercepts_l1184_118445


namespace free_fall_time_l1184_118406

-- Define the relationship between height and time
def height_time_relation (t : ℝ) : ℝ := 4.9 * t^2

-- Define the initial height
def initial_height : ℝ := 490

-- Theorem statement
theorem free_fall_time : 
  ∃ (t : ℝ), t > 0 ∧ height_time_relation t = initial_height ∧ t = 10 := by
  sorry

end free_fall_time_l1184_118406


namespace height_order_l1184_118468

-- Define the set of children
inductive Child : Type
  | A : Child
  | B : Child
  | C : Child
  | D : Child

-- Define the height relation
def taller_than (x y : Child) : Prop := sorry

-- Define the conditions
axiom A_taller_than_B : taller_than Child.A Child.B
axiom B_shorter_than_C : taller_than Child.C Child.B
axiom D_shorter_than_A : taller_than Child.A Child.D
axiom A_not_tallest : ∃ x : Child, taller_than x Child.A
axiom D_not_shortest : ∃ x : Child, taller_than Child.D x

-- Define the order relation
def in_order (w x y z : Child) : Prop :=
  taller_than w x ∧ taller_than x y ∧ taller_than y z

-- State the theorem
theorem height_order : in_order Child.C Child.A Child.D Child.B := by sorry

end height_order_l1184_118468


namespace long_jump_difference_l1184_118434

/-- Represents the long jump event results for Ricciana and Margarita -/
structure LongJumpEvent where
  ricciana_total : ℕ
  ricciana_run : ℕ
  ricciana_jump : ℕ
  margarita_run : ℕ
  h_ricciana_total : ricciana_total = ricciana_run + ricciana_jump
  h_margarita_jump : ℕ

/-- The difference in total distance between Margarita and Ricciana is 1 foot -/
theorem long_jump_difference (event : LongJumpEvent)
  (h_ricciana_total : event.ricciana_total = 24)
  (h_ricciana_run : event.ricciana_run = 20)
  (h_ricciana_jump : event.ricciana_jump = 4)
  (h_margarita_run : event.margarita_run = 18)
  (h_margarita_jump : event.h_margarita_jump = 2 * event.ricciana_jump - 1) :
  event.margarita_run + event.h_margarita_jump - event.ricciana_total = 1 := by
  sorry

end long_jump_difference_l1184_118434


namespace property_necessary_not_sufficient_l1184_118459

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define the property that f(x+1) > f(x) for all x ∈ ℝ
def property_f (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) > f x

-- Define what it means for a function to be increasing on ℝ
def increasing_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Theorem stating that the property is necessary but not sufficient
-- for the function to be increasing on ℝ
theorem property_necessary_not_sufficient :
  (∀ f : ℝ → ℝ, increasing_on_reals f → property_f f) ∧
  ¬(∀ f : ℝ → ℝ, property_f f → increasing_on_reals f) :=
sorry

end property_necessary_not_sufficient_l1184_118459


namespace profit_maximized_at_optimal_production_l1184_118480

/-- Sales revenue as a function of production volume -/
def sales_revenue (x : ℝ) : ℝ := 17 * x^2

/-- Total production cost as a function of production volume -/
def total_cost (x : ℝ) : ℝ := 2 * x^3 - x^2

/-- Profit as a function of production volume -/
def profit (x : ℝ) : ℝ := sales_revenue x - total_cost x

/-- The production volume that maximizes profit -/
def optimal_production : ℝ := 6

theorem profit_maximized_at_optimal_production :
  ∀ x > 0, profit x ≤ profit optimal_production :=
by sorry

end profit_maximized_at_optimal_production_l1184_118480


namespace root_square_minus_two_plus_2023_l1184_118465

theorem root_square_minus_two_plus_2023 (m : ℝ) :
  m^2 - 2*m - 3 = 0 → m^2 - 2*m + 2023 = 2026 := by
  sorry

end root_square_minus_two_plus_2023_l1184_118465


namespace binary_1100111_to_decimal_l1184_118408

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1100111₂ -/
def binary_1100111 : List Bool := [true, true, false, false, true, true, true]

/-- Theorem stating that the decimal equivalent of 1100111₂ is 103 -/
theorem binary_1100111_to_decimal :
  binary_to_decimal binary_1100111 = 103 := by
  sorry

end binary_1100111_to_decimal_l1184_118408


namespace sues_waiting_time_l1184_118425

/-- Proves that Sue's waiting time in New York is 16 hours given the travel conditions -/
theorem sues_waiting_time (total_time : ℝ) (ny_to_sf_time : ℝ) (no_to_ny_ratio : ℝ) 
  (h1 : total_time = 58)
  (h2 : ny_to_sf_time = 24)
  (h3 : no_to_ny_ratio = 3/4)
  : total_time - (no_to_ny_ratio * ny_to_sf_time) - ny_to_sf_time = 16 := by
  sorry

#check sues_waiting_time

end sues_waiting_time_l1184_118425


namespace marble_196_is_green_l1184_118486

/-- Represents the color of a marble -/
inductive MarbleColor
  | Red
  | Green
  | Blue

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 12 with
  | 0 | 1 | 2 => MarbleColor.Red
  | 3 | 4 | 5 | 6 | 7 => MarbleColor.Green
  | _ => MarbleColor.Blue

/-- Theorem stating that the 196th marble is green -/
theorem marble_196_is_green : marbleColor 196 = MarbleColor.Green := by
  sorry


end marble_196_is_green_l1184_118486


namespace table_movement_l1184_118454

theorem table_movement (table_length table_width : ℝ) 
  (hl : table_length = 12) (hw : table_width = 9) : 
  let diagonal := Real.sqrt (table_length^2 + table_width^2)
  ∀ L W : ℕ, 
    (L ≥ diagonal ∧ W ≥ diagonal ∧ L ≥ table_length) → 
    (∀ L' W' : ℕ, (L' < L ∨ W' < W) → 
      ¬(L' ≥ diagonal ∧ W' ≥ diagonal ∧ L' ≥ table_length)) → 
    L = 15 ∧ W = 15 :=
by sorry

end table_movement_l1184_118454


namespace xiaolis_estimate_l1184_118487

theorem xiaolis_estimate (x y z w : ℝ) (hx : x > y) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  (x + z) - (y - w) > x - y := by
  sorry

end xiaolis_estimate_l1184_118487


namespace andy_solves_56_problems_l1184_118457

/-- The number of problems Andy solves -/
def problems_solved (first last : ℕ) : ℕ := last - first + 1

/-- Theorem stating that Andy solves 56 problems -/
theorem andy_solves_56_problems : 
  problems_solved 70 125 = 56 := by sorry

end andy_solves_56_problems_l1184_118457
