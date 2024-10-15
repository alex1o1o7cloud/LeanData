import Mathlib

namespace NUMINAMATH_CALUDE_half_angle_quadrants_l3886_388612

/-- An angle is in the 4th quadrant if it's between 3π/2 and 2π (exclusive) -/
def in_fourth_quadrant (α : ℝ) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

/-- An angle is in the 2nd quadrant if it's between π/2 and π (exclusive) -/
def in_second_quadrant (α : ℝ) : Prop :=
  Real.pi / 2 < α ∧ α < Real.pi

/-- An angle is in the 4th quadrant if it's between 3π/2 and 2π (exclusive) -/
def in_fourth_quadrant_half (α : ℝ) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

theorem half_angle_quadrants (α : ℝ) :
  in_fourth_quadrant α →
  (in_second_quadrant (α/2) ∨ in_fourth_quadrant_half (α/2)) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrants_l3886_388612


namespace NUMINAMATH_CALUDE_inequality_proof_l3886_388652

theorem inequality_proof (m n : ℕ) (h : m < Real.sqrt 2 * n) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * n^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3886_388652


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3886_388647

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = 3 * a^2 * b - a * b^2 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) :
  7 * x + 2 * (x^2 - 2) - 4 * (1/2 * x^2 - x + 3) = 11 * x - 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3886_388647


namespace NUMINAMATH_CALUDE_parabola_focus_l3886_388694

/-- The parabola defined by y = 2x² -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The theorem stating that (0, -1/8) is the focus of the parabola y = 2x² -/
theorem parabola_focus :
  ∃ (f : Focus), f.x = 0 ∧ f.y = -1/8 ∧
  ∀ (x y : ℝ), parabola x y →
    (x - f.x)^2 + (y - f.y)^2 = (y + 1/8)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l3886_388694


namespace NUMINAMATH_CALUDE_f_extrema_on_3_to_5_f_extrema_on_neg1_to_3_l3886_388618

-- Define the function f
def f (x : ℝ) := x^2 - 4*x + 3

-- Theorem for the first interval [3, 5]
theorem f_extrema_on_3_to_5 :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 3 5, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 3 5, f x = max) ∧
    (∀ x ∈ Set.Icc 3 5, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 3 5, f x = min) ∧
    max = 8 ∧ min = 0 :=
sorry

-- Theorem for the second interval [-1, 3]
theorem f_extrema_on_neg1_to_3 :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-1) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-1) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = min) ∧
    max = 8 ∧ min = -1 :=
sorry

end NUMINAMATH_CALUDE_f_extrema_on_3_to_5_f_extrema_on_neg1_to_3_l3886_388618


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3886_388683

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h1 : r = 1 / 4)
  (h2 : S = 50)
  (h3 : S = a / (1 - r)) :
  a = 75 / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3886_388683


namespace NUMINAMATH_CALUDE_square_of_85_l3886_388685

theorem square_of_85 : (85 : ℕ)^2 = 7225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_85_l3886_388685


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3886_388682

theorem triangle_third_side_length (a b c : ℕ) (h1 : a = 5) (h2 : b = 2) (h3 : Odd c) : 
  (a + b > c ∧ a + c > b ∧ b + c > a) → c = 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3886_388682


namespace NUMINAMATH_CALUDE_farm_field_theorem_l3886_388602

/-- Represents the farm field ploughing problem -/
structure FarmField where
  planned_hectares_per_day : ℕ
  actual_hectares_per_day : ℕ
  technical_delay_days : ℕ
  weather_delay_days : ℕ
  remaining_hectares : ℕ

/-- The solution to the farm field problem -/
def farm_field_solution (f : FarmField) : ℕ × ℕ :=
  let total_area := 1560
  let planned_days := 13
  (total_area, planned_days)

/-- Theorem stating the correctness of the farm field solution -/
theorem farm_field_theorem (f : FarmField)
    (h1 : f.planned_hectares_per_day = 120)
    (h2 : f.actual_hectares_per_day = 85)
    (h3 : f.technical_delay_days = 3)
    (h4 : f.weather_delay_days = 2)
    (h5 : f.remaining_hectares = 40) :
    farm_field_solution f = (1560, 13) := by
  sorry

#check farm_field_theorem

end NUMINAMATH_CALUDE_farm_field_theorem_l3886_388602


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l3886_388660

/-- Represents a pentagon with vertices F, G, H, I, J -/
structure Pentagon :=
  (F G H I J : Point)

/-- The area of the pentagon -/
def area (p : Pentagon) : ℝ := sorry

/-- Condition that the pentagon is constructed from 10 line segments of length 3 -/
def is_valid_pentagon (p : Pentagon) : Prop := sorry

theorem pentagon_area_sum (p : Pentagon) (a b : ℕ) :
  is_valid_pentagon p →
  area p = Real.sqrt a + Real.sqrt b →
  a + b = 29 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l3886_388660


namespace NUMINAMATH_CALUDE_triangle_ABC_is_obtuse_angled_l3886_388696

/-- Triangle ABC is obtuse-angled given the specified angle conditions -/
theorem triangle_ABC_is_obtuse_angled (A B C : ℝ) 
  (h1 : A + B = 141)
  (h2 : C + B = 165)
  (h3 : A + B + C = 180) : 
  ∃ (angle : ℝ), angle > 90 ∧ (angle = A ∨ angle = B ∨ angle = C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_is_obtuse_angled_l3886_388696


namespace NUMINAMATH_CALUDE_tangerines_remaining_l3886_388661

/-- The number of tangerines remaining in Yuna's house after Yoo-jung ate some. -/
def remaining_tangerines (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem stating that the number of remaining tangerines is 9. -/
theorem tangerines_remaining :
  remaining_tangerines 12 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_remaining_l3886_388661


namespace NUMINAMATH_CALUDE_general_term_formula_l3886_388631

/-- The sequence defined by the problem -/
def a (n : ℕ+) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 3/4
  else if n = 3 then 5/9
  else if n = 4 then 7/16
  else (2*n - 1) / (n^2)

/-- The theorem stating that the general term formula is correct -/
theorem general_term_formula (n : ℕ+) : a n = (2*n - 1) / (n^2) := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_l3886_388631


namespace NUMINAMATH_CALUDE_central_angle_from_arc_length_l3886_388690

/-- Given a circle with radius 2 and arc length 4, prove that the central angle is 2 radians -/
theorem central_angle_from_arc_length (r : ℝ) (l : ℝ) (h1 : r = 2) (h2 : l = 4) :
  l / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_from_arc_length_l3886_388690


namespace NUMINAMATH_CALUDE_nth_prime_upper_bound_l3886_388666

def nth_prime (n : ℕ) : ℕ := sorry

theorem nth_prime_upper_bound (n : ℕ) : nth_prime n ≤ 2^(2^(n-1)) := by sorry

end NUMINAMATH_CALUDE_nth_prime_upper_bound_l3886_388666


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l3886_388654

/-- Cost calculation for Option 1 -/
def option1Cost (n : ℕ) : ℕ :=
  n.digits 10
    |> List.foldl (fun acc d => acc + if d % 2 = 0 then 2 * d else d) 0

/-- Cost calculation for Option 2 -/
def option2Cost (n : ℕ) : ℕ :=
  n.digits 2
    |> List.foldl (fun acc d => acc + if d = 1 then 2 else 1) 0

/-- Theorem stating that 237 is the largest number less than 500 with equal costs -/
theorem largest_equal_cost_number :
  ∀ n : ℕ, n < 500 → n > 237 →
    option1Cost n ≠ option2Cost n :=
by
  sorry

#eval option1Cost 237
#eval option2Cost 237

end NUMINAMATH_CALUDE_largest_equal_cost_number_l3886_388654


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l3886_388676

theorem polar_to_cartesian (θ : Real) (x y : Real) :
  x = (2 * Real.sin θ + 4 * Real.cos θ) * Real.cos θ ∧
  y = (2 * Real.sin θ + 4 * Real.cos θ) * Real.sin θ →
  (x - 2)^2 + (y - 1)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l3886_388676


namespace NUMINAMATH_CALUDE_function_properties_l3886_388606

/-- The function f(x) -/
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem function_properties (a b : ℝ) :
  (∀ x : ℝ, f' a b x = f' a b (-1 - x)) →  -- f'(x) is symmetric about x = -1/2
  f' a b 1 = 0 →                           -- f'(1) = 0
  a = 3 ∧ b = -12 ∧                        -- values of a and b
  f a b (-2) = 21 ∧ f a b 1 = -6           -- extreme values
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l3886_388606


namespace NUMINAMATH_CALUDE_lcm_prime_sum_l3886_388622

theorem lcm_prime_sum (x y z : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hz : Nat.Prime z)
  (hlcm : Nat.lcm x (Nat.lcm y z) = 210) (hord : x > y ∧ y > z) : 2 * x + y + z = 22 := by
  sorry

end NUMINAMATH_CALUDE_lcm_prime_sum_l3886_388622


namespace NUMINAMATH_CALUDE_julio_lime_cost_l3886_388672

/-- Represents the number of days Julio makes mocktails -/
def days : ℕ := 30

/-- Represents the amount of lime juice used per mocktail in tablespoons -/
def juice_per_mocktail : ℚ := 1

/-- Represents the amount of lime juice that can be squeezed from one lime in tablespoons -/
def juice_per_lime : ℚ := 2

/-- Represents the number of limes sold for $1.00 -/
def limes_per_dollar : ℚ := 3

/-- Calculates the total cost of limes for Julio's mocktails over the given number of days -/
def lime_cost (d : ℕ) (j_mocktail j_lime l_dollar : ℚ) : ℚ :=
  (d * j_mocktail / j_lime) / l_dollar

/-- Theorem stating that Julio will spend $5.00 on limes after 30 days -/
theorem julio_lime_cost : 
  lime_cost days juice_per_mocktail juice_per_lime limes_per_dollar = 5 := by
  sorry

end NUMINAMATH_CALUDE_julio_lime_cost_l3886_388672


namespace NUMINAMATH_CALUDE_tom_search_cost_l3886_388680

/-- Calculates the total cost for Tom's search service given the number of days -/
def total_cost (days : ℕ) : ℕ :=
  if days ≤ 5 then
    100 * days
  else
    500 + 60 * (days - 5)

/-- The problem statement -/
theorem tom_search_cost : total_cost 10 = 800 := by
  sorry

end NUMINAMATH_CALUDE_tom_search_cost_l3886_388680


namespace NUMINAMATH_CALUDE_money_distribution_l3886_388677

theorem money_distribution (p q r : ℚ) : 
  p + q + r = 5000 →
  r = (2/3) * (p + q) →
  r = 2000 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3886_388677


namespace NUMINAMATH_CALUDE_unique_quadruple_existence_l3886_388691

theorem unique_quadruple_existence : 
  ∃! (a b c d : ℝ), 
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a^2 + b^2 + c^2 + d^2 = 4 ∧
    (a + b + c + d) * (a^2 + b^2 + c^2 + d^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadruple_existence_l3886_388691


namespace NUMINAMATH_CALUDE_wage_decrease_theorem_l3886_388670

theorem wage_decrease_theorem (x : ℝ) : 
  (100 - x) * 1.5 = 75 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_wage_decrease_theorem_l3886_388670


namespace NUMINAMATH_CALUDE_female_teachers_count_l3886_388627

/-- The number of teachers in the group -/
def total_teachers : ℕ := 5

/-- The probability of selecting a female teacher -/
def prob_female : ℚ := 7/10

/-- Calculates the number of combinations of k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the probability of selecting two male teachers given x female teachers -/
def prob_two_male (x : ℕ) : ℚ :=
  1 - (choose (total_teachers - x) 2 : ℚ) / (choose total_teachers 2 : ℚ)

theorem female_teachers_count :
  ∃ x : ℕ, x ≤ total_teachers ∧ prob_two_male x = 1 - prob_female :=
sorry

end NUMINAMATH_CALUDE_female_teachers_count_l3886_388627


namespace NUMINAMATH_CALUDE_max_intersection_points_l3886_388637

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 20

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 10

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := 8550

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points :
  (num_x_points.choose 2) * (num_y_points.choose 2) = max_intersections :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_l3886_388637


namespace NUMINAMATH_CALUDE_problem_solution_l3886_388613

open Real

noncomputable def f (x : ℝ) : ℝ := (log (1 + x)) / x

theorem problem_solution :
  (∀ x y, 0 < x ∧ x < y → f y < f x) ∧
  (∀ a : ℝ, (∀ x : ℝ, 0 < x → log (1 + x) < a * x) ↔ 1 ≤ a) ∧
  (∀ n : ℕ, 0 < n → (1 + 1 / n : ℝ) ^ n < exp 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3886_388613


namespace NUMINAMATH_CALUDE_victors_total_money_l3886_388616

-- Define Victor's initial amount
def initial_amount : ℕ := 10

-- Define Victor's allowance
def allowance : ℕ := 8

-- Theorem stating Victor's total money
theorem victors_total_money : initial_amount + allowance = 18 := by
  sorry

end NUMINAMATH_CALUDE_victors_total_money_l3886_388616


namespace NUMINAMATH_CALUDE_smallest_linear_combination_2023_54321_l3886_388620

theorem smallest_linear_combination_2023_54321 :
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 2023 * m + 54321 * n) ∧
  ∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 2023 * x + 54321 * y) → k ≤ j :=
by sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_2023_54321_l3886_388620


namespace NUMINAMATH_CALUDE_possible_distances_l3886_388625

/-- Represents the position of a house on a street. -/
structure House where
  position : ℝ

/-- Represents a street with four houses. -/
structure Street where
  andrey : House
  boris : House
  vova : House
  gleb : House

/-- The distance between two houses. -/
def distance (h1 h2 : House) : ℝ :=
  |h1.position - h2.position|

/-- A street is valid if it satisfies the given conditions. -/
def validStreet (s : Street) : Prop :=
  distance s.andrey s.boris = 600 ∧
  distance s.vova s.gleb = 600 ∧
  distance s.andrey s.gleb = 3 * distance s.boris s.vova

/-- The theorem stating the possible distances between Andrey's and Gleb's houses. -/
theorem possible_distances (s : Street) (h : validStreet s) :
  distance s.andrey s.gleb = 900 ∨ distance s.andrey s.gleb = 1800 :=
sorry

end NUMINAMATH_CALUDE_possible_distances_l3886_388625


namespace NUMINAMATH_CALUDE_unique_abc_solution_l3886_388655

/-- Represents a base-5 number with two digits -/
def BaseFiveNumber (a b : Nat) : Nat := 5 * a + b

/-- Represents a three-digit number in base 10 -/
def ThreeDigitNumber (a b c : Nat) : Nat := 100 * a + 10 * b + c

theorem unique_abc_solution :
  ∀ (A B C : Nat),
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A < 5 → B < 5 → C < 5 →
    A ≠ B → B ≠ C → A ≠ C →
    BaseFiveNumber A B + C = BaseFiveNumber C 0 →
    BaseFiveNumber A B + BaseFiveNumber B A = BaseFiveNumber C C →
    ThreeDigitNumber A B C = 323 := by
  sorry

end NUMINAMATH_CALUDE_unique_abc_solution_l3886_388655


namespace NUMINAMATH_CALUDE_cube_root_nine_inequality_false_l3886_388662

theorem cube_root_nine_inequality_false : 
  ¬(∀ n : ℤ, (n : ℝ) < (9 : ℝ)^(1/3) ∧ (9 : ℝ)^(1/3) < (n : ℝ) + 1 → n = 3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_nine_inequality_false_l3886_388662


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3886_388656

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 5 * i) / (2 + 7 * i) = Complex.mk (-29/53) (-31/53) :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3886_388656


namespace NUMINAMATH_CALUDE_min_focal_chord_length_is_2p_l3886_388639

/-- Represents a parabola defined by the equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- The minimum length of focal chords for a given parabola -/
def min_focal_chord_length (par : Parabola) : ℝ := 2 * par.p

/-- Theorem stating that the minimum length of focal chords is 2p -/
theorem min_focal_chord_length_is_2p (par : Parabola) :
  min_focal_chord_length par = 2 * par.p := by sorry

end NUMINAMATH_CALUDE_min_focal_chord_length_is_2p_l3886_388639


namespace NUMINAMATH_CALUDE_sum_of_s_coordinates_l3886_388686

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle defined by four points -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Given a rectangle PQRS with P and R as diagonally opposite corners,
    proves that the sum of coordinates of S is 8 -/
theorem sum_of_s_coordinates (rect : Rectangle) : 
  rect.P = Point.mk (-3) (-2) →
  rect.R = Point.mk 9 1 →
  rect.Q = Point.mk 2 (-5) →
  rect.S.x + rect.S.y = 8 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_s_coordinates_l3886_388686


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l3886_388663

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  (Complex.exp (Complex.I * θ) = (4 : ℝ) / 5 + (3 : ℝ) / 5 * Complex.I) →
  (Complex.exp (Complex.I * φ) = -(5 : ℝ) / 13 + (12 : ℝ) / 13 * Complex.I) →
  Real.sin (θ + φ) = (33 : ℝ) / 65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l3886_388663


namespace NUMINAMATH_CALUDE_eleven_row_triangle_pieces_l3886_388624

/-- Calculates the total number of pieces in a triangle with given number of rows -/
def totalPieces (rows : ℕ) : ℕ :=
  let rodSum := (rows * (rows + 1) * 3) / 2
  let connectorSum := (rows + 1) * (rows + 2) / 2
  rodSum + connectorSum

/-- Theorem stating that an eleven-row triangle has 276 pieces -/
theorem eleven_row_triangle_pieces :
  totalPieces 11 = 276 := by sorry

end NUMINAMATH_CALUDE_eleven_row_triangle_pieces_l3886_388624


namespace NUMINAMATH_CALUDE_triangle_existence_l3886_388679

theorem triangle_existence (y : ℕ+) : 
  (∃ (a b c : ℝ), a = 8 ∧ b = 12 ∧ c = y.val^2 ∧ 
   a + b > c ∧ a + c > b ∧ b + c > a) ↔ (y = 3 ∨ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_l3886_388679


namespace NUMINAMATH_CALUDE_tan_arccos_three_fifths_l3886_388692

theorem tan_arccos_three_fifths : Real.tan (Real.arccos (3/5)) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_arccos_three_fifths_l3886_388692


namespace NUMINAMATH_CALUDE_kelly_travel_days_l3886_388673

/-- Kelly's vacation details -/
structure VacationSchedule where
  total_days : ℕ
  initial_travel : ℕ
  grandparents : ℕ
  brother : ℕ
  to_sister_travel : ℕ
  sister : ℕ
  final_travel : ℕ

/-- The number of days Kelly spent traveling between her Grandparents' house and her brother's house -/
def days_between_grandparents_and_brother (schedule : VacationSchedule) : ℕ :=
  schedule.total_days - (schedule.initial_travel + schedule.grandparents + 
    schedule.brother + schedule.to_sister_travel + schedule.sister + schedule.final_travel)

/-- Theorem stating that Kelly spent 1 day traveling between her Grandparents' and brother's houses -/
theorem kelly_travel_days (schedule : VacationSchedule) 
  (h1 : schedule.total_days = 21)  -- Three weeks
  (h2 : schedule.initial_travel = 1)
  (h3 : schedule.grandparents = 5)
  (h4 : schedule.brother = 5)
  (h5 : schedule.to_sister_travel = 2)
  (h6 : schedule.sister = 5)
  (h7 : schedule.final_travel = 2) :
  days_between_grandparents_and_brother schedule = 1 := by
  sorry

end NUMINAMATH_CALUDE_kelly_travel_days_l3886_388673


namespace NUMINAMATH_CALUDE_chessboard_impossible_l3886_388699

/-- Represents a 6x6 chessboard filled with numbers -/
def Chessboard := Fin 6 → Fin 6 → Fin 36

/-- The sum of numbers from 1 to 36 -/
def total_sum : Nat := (36 * 37) / 2

/-- The required sum for each row, column, and diagonal -/
def required_sum : Nat := total_sum / 6

/-- Checks if a number appears exactly once on the chessboard -/
def appears_once (board : Chessboard) (n : Fin 36) : Prop :=
  ∃! (i j : Fin 6), board i j = n

/-- Checks if a row has the required sum -/
def row_sum_correct (board : Chessboard) (i : Fin 6) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun j => (board i j).val + 1) = required_sum

/-- Checks if a column has the required sum -/
def col_sum_correct (board : Chessboard) (j : Fin 6) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun i => (board i j).val + 1) = required_sum

/-- Checks if a northeast diagonal has the required sum -/
def diag_sum_correct (board : Chessboard) (k : Fin 6) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun i =>
    (board i ((i.val - k.val + 6) % 6 : Fin 6)).val + 1) = required_sum

/-- The main theorem stating that it's impossible to fill the chessboard with the given conditions -/
theorem chessboard_impossible : ¬∃ (board : Chessboard),
  (∀ n : Fin 36, appears_once board n) ∧
  (∀ i : Fin 6, row_sum_correct board i) ∧
  (∀ j : Fin 6, col_sum_correct board j) ∧
  (∀ k : Fin 6, diag_sum_correct board k) :=
sorry

end NUMINAMATH_CALUDE_chessboard_impossible_l3886_388699


namespace NUMINAMATH_CALUDE_trig_identity_l3886_388609

theorem trig_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin x + Real.cos (x + y) * Real.cos x = Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3886_388609


namespace NUMINAMATH_CALUDE_circle_area_difference_l3886_388630

theorem circle_area_difference (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 24) (h₂ : r₂ = 36) :
  r₃ ^ 2 * π = (r₂ ^ 2 - r₁ ^ 2) * π → r₃ = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l3886_388630


namespace NUMINAMATH_CALUDE_sin_30_degrees_l3886_388617

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l3886_388617


namespace NUMINAMATH_CALUDE_professor_seating_count_l3886_388604

/-- The number of chairs in a row -/
def total_chairs : ℕ := 12

/-- The number of professors -/
def num_professors : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 8

/-- The number of chairs available for professors (excluding first and last) -/
def available_chairs : ℕ := total_chairs - 2

/-- The number of effective chairs after considering spacing requirements -/
def effective_chairs : ℕ := available_chairs - (num_professors - 1)

/-- The number of ways to arrange professors' seating -/
def professor_seating_arrangements : ℕ := (effective_chairs.choose num_professors) * num_professors.factorial

theorem professor_seating_count :
  professor_seating_arrangements = 1680 :=
sorry

end NUMINAMATH_CALUDE_professor_seating_count_l3886_388604


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_14_l3886_388638

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the vertex of the original parabola
def vertex : ℝ × ℝ := (3, 4)

-- Define the transformed parabola after rotation and translation
def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 1

-- Define the zeros of the transformed parabola
def p : ℝ := 6
def q : ℝ := 8

theorem sum_of_zeros_is_14 : p + q = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_14_l3886_388638


namespace NUMINAMATH_CALUDE_specific_value_calculation_l3886_388681

theorem specific_value_calculation : ∀ (x : ℕ), x = 11 → x + 3 + 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_specific_value_calculation_l3886_388681


namespace NUMINAMATH_CALUDE_tires_cost_calculation_l3886_388608

def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def total_spent : ℚ := 387.85

theorem tires_cost_calculation :
  total_spent - (speakers_cost + cd_player_cost) = 112.46 :=
by sorry

end NUMINAMATH_CALUDE_tires_cost_calculation_l3886_388608


namespace NUMINAMATH_CALUDE_stratified_sample_is_proportional_l3886_388695

/-- Represents the number of students in each grade and the sample size -/
structure School :=
  (total : ℕ)
  (freshmen : ℕ)
  (sophomores : ℕ)
  (seniors : ℕ)
  (sample_size : ℕ)

/-- Represents the number of students sampled from each grade -/
structure Sample :=
  (freshmen : ℕ)
  (sophomores : ℕ)
  (seniors : ℕ)

/-- Calculates the proportional sample size for a given grade -/
def proportional_sample (grade_size : ℕ) (school : School) : ℕ :=
  (grade_size * school.sample_size) / school.total

/-- Checks if a sample is proportionally correct -/
def is_proportional_sample (school : School) (sample : Sample) : Prop :=
  sample.freshmen = proportional_sample school.freshmen school ∧
  sample.sophomores = proportional_sample school.sophomores school ∧
  sample.seniors = proportional_sample school.seniors school

/-- Theorem: The stratified sample is proportional for the given school -/
theorem stratified_sample_is_proportional (school : School)
  (h1 : school.total = 900)
  (h2 : school.freshmen = 300)
  (h3 : school.sophomores = 200)
  (h4 : school.seniors = 400)
  (h5 : school.sample_size = 45)
  (h6 : school.total = school.freshmen + school.sophomores + school.seniors) :
  is_proportional_sample school ⟨15, 10, 20⟩ := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_is_proportional_l3886_388695


namespace NUMINAMATH_CALUDE_hyperbolic_to_linear_transformation_l3886_388688

theorem hyperbolic_to_linear_transformation (x y a b : ℝ) (h : 1 / y = a + b / x) :
  1 / y = a + b * (1 / x) := by sorry

end NUMINAMATH_CALUDE_hyperbolic_to_linear_transformation_l3886_388688


namespace NUMINAMATH_CALUDE_sixth_grade_forgot_homework_percentage_l3886_388669

/-- Represents the percentage of students who forgot their homework in a group -/
def forgot_homework_percentage (total : ℕ) (forgot : ℕ) : ℚ :=
  (forgot : ℚ) / (total : ℚ) * 100

/-- Calculates the total number of students who forgot their homework -/
def total_forgot (group_a_total : ℕ) (group_b_total : ℕ) 
  (group_a_forgot_percent : ℚ) (group_b_forgot_percent : ℚ) : ℕ :=
  (group_a_total * group_a_forgot_percent.num / group_a_forgot_percent.den).toNat +
  (group_b_total * group_b_forgot_percent.num / group_b_forgot_percent.den).toNat

theorem sixth_grade_forgot_homework_percentage :
  let group_a_total : ℕ := 20
  let group_b_total : ℕ := 80
  let group_a_forgot_percent : ℚ := 20 / 100
  let group_b_forgot_percent : ℚ := 15 / 100
  let total_students : ℕ := group_a_total + group_b_total
  let total_forgot : ℕ := total_forgot group_a_total group_b_total group_a_forgot_percent group_b_forgot_percent
  forgot_homework_percentage total_students total_forgot = 16 := by
sorry

end NUMINAMATH_CALUDE_sixth_grade_forgot_homework_percentage_l3886_388669


namespace NUMINAMATH_CALUDE_unique_point_perpendicular_segments_l3886_388659

/-- Given a non-zero real number α, there exists a unique point P in the coordinate plane
    such that for every line through P intersecting the parabola y = αx² in two distinct points A and B,
    the segments OA and OB are perpendicular (where O is the origin). -/
theorem unique_point_perpendicular_segments (α : ℝ) (h : α ≠ 0) :
  ∃! P : ℝ × ℝ, ∀ (A B : ℝ × ℝ),
    (A.2 = α * A.1^2) →
    (B.2 = α * B.1^2) →
    (∃ t : ℝ, A.1 + t * (P.1 - A.1) = B.1 ∧ A.2 + t * (P.2 - A.2) = B.2) →
    (A ≠ B) →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    P = (0, 1 / α) :=
sorry

end NUMINAMATH_CALUDE_unique_point_perpendicular_segments_l3886_388659


namespace NUMINAMATH_CALUDE_trapezoid_area_is_42_5_l3886_388600

/-- A trapezoid bounded by the lines y = x + 2, y = 12, y = 7, and the y-axis -/
structure Trapezoid where
  -- Line equations
  line1 : ℝ → ℝ := λ x => x + 2
  line2 : ℝ → ℝ := λ _ => 12
  line3 : ℝ → ℝ := λ _ => 7
  y_axis : ℝ → ℝ := λ _ => 0

/-- The area of the trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that the area of the trapezoid is 42.5 square units -/
theorem trapezoid_area_is_42_5 (t : Trapezoid) : trapezoid_area t = 42.5 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_42_5_l3886_388600


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l3886_388619

theorem opposite_of_negative_two_thirds :
  -(-(2/3)) = 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l3886_388619


namespace NUMINAMATH_CALUDE_roots_on_circle_l3886_388626

theorem roots_on_circle : ∃ (r : ℝ), r = 2 / Real.sqrt 3 ∧
  ∀ (z : ℂ), (z + 2)^4 = 16 * z^4 →
  ∃ (c : ℂ), Complex.abs (z - c) = r :=
sorry

end NUMINAMATH_CALUDE_roots_on_circle_l3886_388626


namespace NUMINAMATH_CALUDE_bridge_building_time_l3886_388657

/-- If 60 workers can build a bridge in 8 days, then 40 workers can build the same bridge in 12 days, given that all workers work at the same rate. -/
theorem bridge_building_time 
  (work : ℝ) -- Total amount of work required to build the bridge
  (rate : ℝ) -- Rate of work per worker per day
  (h1 : work = 60 * rate * 8) -- 60 workers complete the bridge in 8 days
  (h2 : rate > 0) -- Workers have a positive work rate
  : work = 40 * rate * 12 := by
  sorry

end NUMINAMATH_CALUDE_bridge_building_time_l3886_388657


namespace NUMINAMATH_CALUDE_fourth_selected_id_is_16_l3886_388623

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  totalStudents : Nat
  numTickets : Nat
  selectedIDs : Fin 3 → Nat

/-- Calculates the sampling interval for a given systematic sampling -/
def samplingInterval (s : SystematicSampling) : Nat :=
  s.totalStudents / s.numTickets

/-- Checks if a given ID is part of the systematic sampling -/
def isSelectedID (s : SystematicSampling) (id : Nat) : Prop :=
  ∃ k : Fin s.numTickets, id = (s.selectedIDs 0) + k * samplingInterval s

/-- Theorem: Given the conditions, the fourth selected ID is 16 -/
theorem fourth_selected_id_is_16 (s : SystematicSampling) 
  (h1 : s.totalStudents = 54)
  (h2 : s.numTickets = 4)
  (h3 : s.selectedIDs 0 = 3)
  (h4 : s.selectedIDs 1 = 29)
  (h5 : s.selectedIDs 2 = 42)
  : ∃ id : Nat, id = 16 ∧ isSelectedID s id :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_selected_id_is_16_l3886_388623


namespace NUMINAMATH_CALUDE_stream_speed_l3886_388689

/-- Proves that the speed of a stream is 4 km/hr given the conditions of the boat problem -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : boat_speed = 24)
  (h2 : distance = 168) (h3 : time = 6)
  (h4 : distance = (boat_speed + (distance / time - boat_speed)) * time) : 
  distance / time - boat_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3886_388689


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3886_388697

theorem algebraic_expression_equality (x y : ℝ) : 
  x - 2*y + 8 = 18 → 3*x - 6*y + 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3886_388697


namespace NUMINAMATH_CALUDE_expand_and_simplify_powers_of_two_one_more_than_cube_l3886_388615

-- Part (i)
theorem expand_and_simplify (x : ℝ) : (x + 1) * (x^2 - x + 1) = x^3 + 1 := by sorry

-- Part (ii)
theorem powers_of_two_one_more_than_cube : 
  {n : ℕ | ∃ k : ℕ, 2^n = k^3 + 1} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_powers_of_two_one_more_than_cube_l3886_388615


namespace NUMINAMATH_CALUDE_tan_theta_plus_pi_fourth_l3886_388632

theorem tan_theta_plus_pi_fourth (θ : Real) 
  (h : (Real.cos (2 * θ) + 1) / (1 + 2 * Real.sin (2 * θ)) = -2/3) : 
  Real.tan (θ + π/4) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_plus_pi_fourth_l3886_388632


namespace NUMINAMATH_CALUDE_power_of_seven_expansion_l3886_388636

theorem power_of_seven_expansion : 7^3 - 3*(7^2) + 3*7 - 1 = 216 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_expansion_l3886_388636


namespace NUMINAMATH_CALUDE_complex_square_of_one_plus_i_l3886_388674

theorem complex_square_of_one_plus_i :
  ∀ z : ℂ, (z.re = 1 ∧ z.im = 1) → z^2 = 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_of_one_plus_i_l3886_388674


namespace NUMINAMATH_CALUDE_line_difference_l3886_388628

/-- Represents a character in the script --/
structure Character where
  lines : ℕ

/-- Represents the script with three characters --/
structure Script where
  char1 : Character
  char2 : Character
  char3 : Character

/-- Theorem: The difference in lines between the first and second character is 8 --/
theorem line_difference (s : Script) : 
  s.char3.lines = 2 →
  s.char2.lines = 3 * s.char3.lines + 6 →
  s.char1.lines = 20 →
  s.char1.lines - s.char2.lines = 8 := by
  sorry


end NUMINAMATH_CALUDE_line_difference_l3886_388628


namespace NUMINAMATH_CALUDE_inequality_preservation_l3886_388675

theorem inequality_preservation (m n : ℝ) (h : m > n) : m / 4 > n / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3886_388675


namespace NUMINAMATH_CALUDE_work_completion_time_l3886_388607

theorem work_completion_time (a b c : ℝ) 
  (h1 : a + b + c = 1/4)  -- Combined work rate
  (h2 : a = 1/12)         -- a's work rate
  (h3 : b = 1/18)         -- b's work rate
  : c = 1/9 :=            -- c's work rate (to be proved)
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3886_388607


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3886_388678

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x + 1/2 ≥ 0) → 
  (k > 0 ∧ k ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3886_388678


namespace NUMINAMATH_CALUDE_bank_transfer_problem_l3886_388603

theorem bank_transfer_problem (X : ℝ) :
  (0.8 * X = 30000) → X = 37500 := by
  sorry

end NUMINAMATH_CALUDE_bank_transfer_problem_l3886_388603


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_75_4410_l3886_388611

theorem gcd_lcm_sum_75_4410 : Nat.gcd 75 4410 + Nat.lcm 75 4410 = 22065 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_75_4410_l3886_388611


namespace NUMINAMATH_CALUDE_test_score_calculation_l3886_388642

/-- Calculates the total score for a test given the total number of problems,
    points for correct answers, points deducted for wrong answers,
    and the number of wrong answers. -/
def calculateScore (totalProblems : ℕ) (pointsPerCorrect : ℕ) (pointsPerWrong : ℕ) (wrongAnswers : ℕ) : ℤ :=
  (totalProblems - wrongAnswers : ℤ) * pointsPerCorrect - wrongAnswers * pointsPerWrong

/-- Theorem stating that for a test with 25 problems, 4 points for each correct answer,
    1 point deducted for each wrong answer, and 3 wrong answers, the total score is 85. -/
theorem test_score_calculation :
  calculateScore 25 4 1 3 = 85 := by
  sorry

end NUMINAMATH_CALUDE_test_score_calculation_l3886_388642


namespace NUMINAMATH_CALUDE_sin_negative_300_degrees_l3886_388610

theorem sin_negative_300_degrees : Real.sin (-(300 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_300_degrees_l3886_388610


namespace NUMINAMATH_CALUDE_f_strictly_increasing_iff_a_in_range_l3886_388633

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a^2 - 1) * Real.exp (a * x)

-- State the theorem
theorem f_strictly_increasing_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, (deriv (f a)) x > 0) ↔ (1 < a ∧ a ≤ Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_iff_a_in_range_l3886_388633


namespace NUMINAMATH_CALUDE_arithmetic_operations_l3886_388629

theorem arithmetic_operations : 
  (100 - 54 - 46 = 0) ∧ 
  (234 - (134 + 45) = 55) ∧ 
  (125 * 7 * 8 = 7000) ∧ 
  (15 * (61 - 45) = 240) ∧ 
  (318 / 6 + 165 = 218) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l3886_388629


namespace NUMINAMATH_CALUDE_penny_remaining_money_l3886_388698

/-- Calculates the remaining money after Penny's shopping trip --/
def remaining_money (initial_amount : ℚ) (sock_price : ℚ) (sock_quantity : ℕ)
  (hat_price : ℚ) (hat_quantity : ℕ) (scarf_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_cost := sock_price * sock_quantity + hat_price * hat_quantity + scarf_price
  let discounted_cost := total_cost * (1 - discount_rate)
  initial_amount - discounted_cost

/-- Theorem stating that Penny has $14 left after her purchases --/
theorem penny_remaining_money :
  remaining_money 50 4 3 10 2 8 (1/10) = 14 := by
  sorry

end NUMINAMATH_CALUDE_penny_remaining_money_l3886_388698


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3886_388644

/-- Proves that the rate of interest is 8% given the problem conditions --/
theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) (rate : ℝ) :
  principal = 1100 →
  interest_paid = 704 →
  interest_paid = principal * rate * rate / 100 →
  rate = 8 :=
by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l3886_388644


namespace NUMINAMATH_CALUDE_no_non_divisor_exists_l3886_388601

theorem no_non_divisor_exists (a : ℕ+) : ∃ (b n : ℕ+), a.val ∣ (b.val ^ n.val - n.val) := by
  sorry

end NUMINAMATH_CALUDE_no_non_divisor_exists_l3886_388601


namespace NUMINAMATH_CALUDE_cubic_equation_product_l3886_388635

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016)
  (h₄ : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃))
  (h₅ : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -671/336 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_product_l3886_388635


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_of_sqrt5_to_4th_l3886_388665

theorem sqrt_fifth_power_of_sqrt5_to_4th : (((5 : ℝ) ^ (1/2)) ^ 5) ^ (1/2) ^ 4 = 9765625 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_of_sqrt5_to_4th_l3886_388665


namespace NUMINAMATH_CALUDE_small_triangle_perimeter_l3886_388653

/-- Given a triangle with perimeter 11 and three trapezoids formed by cuts parallel to its sides
    with perimeters 5, 7, and 9, the perimeter of the small triangle formed after the cuts is 10. -/
theorem small_triangle_perimeter (original_perimeter : ℝ) (trapezoid1_perimeter trapezoid2_perimeter trapezoid3_perimeter : ℝ)
    (h1 : original_perimeter = 11)
    (h2 : trapezoid1_perimeter = 5)
    (h3 : trapezoid2_perimeter = 7)
    (h4 : trapezoid3_perimeter = 9) :
    trapezoid1_perimeter + trapezoid2_perimeter + trapezoid3_perimeter = original_perimeter + 10 := by
  sorry

end NUMINAMATH_CALUDE_small_triangle_perimeter_l3886_388653


namespace NUMINAMATH_CALUDE_solution_set_theorem_l3886_388605

def f (a b x : ℝ) := (a * x - 1) * (x + b)

theorem solution_set_theorem (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x, f a b (-2 * x) < 0 ↔ x < -3/2 ∨ 1/2 < x) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l3886_388605


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l3886_388614

theorem factor_difference_of_squares (t : ℝ) : 4 * t^2 - 81 = (2*t - 9) * (2*t + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l3886_388614


namespace NUMINAMATH_CALUDE_brendan_lawn_cutting_l3886_388687

/-- The number of yards Brendan could cut per day before buying the lawnmower -/
def initial_yards : ℝ := 8

/-- The increase in cutting capacity after buying the lawnmower -/
def capacity_increase : ℝ := 0.5

/-- The number of days Brendan worked with the new lawnmower -/
def days_worked : ℕ := 7

/-- The total number of yards cut with the new lawnmower -/
def total_yards_cut : ℕ := 84

theorem brendan_lawn_cutting :
  initial_yards * (1 + capacity_increase) * days_worked = total_yards_cut :=
by sorry

end NUMINAMATH_CALUDE_brendan_lawn_cutting_l3886_388687


namespace NUMINAMATH_CALUDE_twins_age_product_difference_l3886_388667

theorem twins_age_product_difference (current_age : ℕ) : 
  current_age = 2 → (current_age + 1)^2 - current_age^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_product_difference_l3886_388667


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3886_388646

theorem fraction_multiplication (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hx : x ≠ 0) :
  (3 * a * b) / x * (2 * x^2) / (9 * a * b^2) = (2 * x) / (3 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3886_388646


namespace NUMINAMATH_CALUDE_complex_power_difference_l3886_388684

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^12 - (1 - i)^12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l3886_388684


namespace NUMINAMATH_CALUDE_novel_pages_l3886_388640

theorem novel_pages (planned_days : ℕ) (actual_days : ℕ) (extra_pages_per_day : ℕ) 
  (h1 : planned_days = 20)
  (h2 : actual_days = 15)
  (h3 : extra_pages_per_day = 20) : 
  (planned_days * ((actual_days * extra_pages_per_day) / (planned_days - actual_days))) = 1200 :=
by sorry

end NUMINAMATH_CALUDE_novel_pages_l3886_388640


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3886_388664

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := -1 + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3886_388664


namespace NUMINAMATH_CALUDE_exists_perpendicular_angles_not_equal_or_180_l3886_388648

/-- Two angles in 3D space with perpendicular sides -/
structure PerpendicularAngles where
  α : Real
  β : Real
  perp_sides : Bool

/-- Predicate for angles being equal or summing to 180° -/
def equal_or_sum_180 (angles : PerpendicularAngles) : Prop :=
  angles.α = angles.β ∨ angles.α + angles.β = 180

/-- Theorem stating the existence of perpendicular angles that don't satisfy the condition -/
theorem exists_perpendicular_angles_not_equal_or_180 :
  ∃ (angles : PerpendicularAngles), angles.perp_sides ∧ ¬(equal_or_sum_180 angles) :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_angles_not_equal_or_180_l3886_388648


namespace NUMINAMATH_CALUDE_daffodil_stamps_count_l3886_388671

theorem daffodil_stamps_count 
  (rooster_stamps : ℕ) 
  (daffodil_stamps : ℕ) 
  (h1 : rooster_stamps = 2) 
  (h2 : rooster_stamps - daffodil_stamps = 0) : 
  daffodil_stamps = 2 :=
by sorry

end NUMINAMATH_CALUDE_daffodil_stamps_count_l3886_388671


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3886_388634

/-- Given a hyperbola with equation x^2 - 4y^2 = -1, its asymptotes are x ± 2y = 0 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 - 4*y^2 = -1 →
  ∃ (k : ℝ), (x + 2*y = 0 ∧ x - 2*y = 0) ∨ (2*y + x = 0 ∧ 2*y - x = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3886_388634


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3886_388649

theorem intersection_of_sets : 
  let A : Set Int := {-1, 0, 1}
  let B : Set Int := {0, 1, 2, 3}
  A ∩ B = {0, 1} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3886_388649


namespace NUMINAMATH_CALUDE_sum_of_series_l3886_388621

def series_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2 - 1

theorem sum_of_series :
  series_sum 15 - series_sum 1 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_l3886_388621


namespace NUMINAMATH_CALUDE_reciprocal_of_two_thirds_l3886_388658

theorem reciprocal_of_two_thirds : 
  (2 : ℚ) / 3 * (3 : ℚ) / 2 = 1 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_two_thirds_l3886_388658


namespace NUMINAMATH_CALUDE_equation_solution_l3886_388651

theorem equation_solution : 
  ∀ x : ℝ, (x + 2) * (x + 1) = 3 * (x + 1) ↔ x = -1 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3886_388651


namespace NUMINAMATH_CALUDE_good_number_exists_l3886_388645

/-- A function that checks if two numbers have the same digits (possibly in different order) --/
def sameDigits (a b : ℕ) : Prop := sorry

/-- A function that checks if a number is a four-digit number --/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem good_number_exists : ∃ n : ℕ, 
  isFourDigit n ∧ 
  n % 11 = 0 ∧ 
  sameDigits n (3 * n) ∧
  n = 2475 := by sorry

end NUMINAMATH_CALUDE_good_number_exists_l3886_388645


namespace NUMINAMATH_CALUDE_percent_increase_in_sales_l3886_388693

def sales_last_year : ℝ := 320
def sales_this_year : ℝ := 480

theorem percent_increase_in_sales :
  (sales_this_year - sales_last_year) / sales_last_year * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_in_sales_l3886_388693


namespace NUMINAMATH_CALUDE_four_isosceles_triangles_l3886_388668

-- Define a Point type for 2D coordinates
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Triangle type
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

-- Function to calculate the squared distance between two points
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := distanceSquared t.a t.b
  let d2 := distanceSquared t.b t.c
  let d3 := distanceSquared t.c t.a
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the five triangles
def triangleA : Triangle := ⟨⟨1, 5⟩, ⟨3, 5⟩, ⟨2, 3⟩⟩
def triangleB : Triangle := ⟨⟨4, 3⟩, ⟨4, 5⟩, ⟨6, 3⟩⟩
def triangleC : Triangle := ⟨⟨1, 2⟩, ⟨3, 1⟩, ⟨5, 2⟩⟩
def triangleD : Triangle := ⟨⟨7, 3⟩, ⟨6, 5⟩, ⟨9, 3⟩⟩
def triangleE : Triangle := ⟨⟨8, 2⟩, ⟨9, 4⟩, ⟨10, 1⟩⟩

-- Theorem stating that exactly 4 out of 5 triangles are isosceles
theorem four_isosceles_triangles :
  (isIsosceles triangleA ∧ 
   isIsosceles triangleB ∧ 
   isIsosceles triangleC ∧ 
   ¬isIsosceles triangleD ∧ 
   isIsosceles triangleE) :=
by sorry

end NUMINAMATH_CALUDE_four_isosceles_triangles_l3886_388668


namespace NUMINAMATH_CALUDE_arcsin_arccos_pi_sixth_l3886_388650

theorem arcsin_arccos_pi_sixth : 
  Real.arcsin (1/2) = π/6 ∧ Real.arccos (Real.sqrt 3/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_arccos_pi_sixth_l3886_388650


namespace NUMINAMATH_CALUDE_program_result_l3886_388643

def double_n_times (initial : ℕ) (n : ℕ) : ℕ :=
  initial * (2^n)

theorem program_result :
  double_n_times 1 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_program_result_l3886_388643


namespace NUMINAMATH_CALUDE_train_crossing_time_l3886_388641

/-- Proves that a train with given length and speed takes a specific time to cross a pole --/
theorem train_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (h1 : train_length = 125) 
  (h2 : train_speed_kmh = 90) : 
  train_length / (train_speed_kmh * 1000 / 3600) = 5 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3886_388641
