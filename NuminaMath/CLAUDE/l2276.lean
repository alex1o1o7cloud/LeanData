import Mathlib

namespace NUMINAMATH_CALUDE_height_percentage_difference_l2276_227632

theorem height_percentage_difference (a b : ℝ) (h : b = 1.25 * a) :
  (b - a) / b * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_height_percentage_difference_l2276_227632


namespace NUMINAMATH_CALUDE_smallest_c_l2276_227645

/-- A square with side length c -/
structure Square (c : ℝ) where
  side : c > 0

/-- A coloring of points on a square -/
def Coloring (c : ℝ) := Square c → Bool

/-- The distance between two points on a square -/
def distance (c : ℝ) (p q : Square c) : ℝ := sorry

/-- There exist two points of the same color with distance at least √5 -/
def hasMonochromaticPair (c : ℝ) (coloring : Coloring c) : Prop :=
  ∃ (p q : Square c), coloring p = coloring q ∧ distance c p q ≥ Real.sqrt 5

/-- The smallest possible value of c satisfying the condition -/
theorem smallest_c : 
  (∀ c : ℝ, c ≥ Real.sqrt 10 / 2 → ∀ coloring : Coloring c, hasMonochromaticPair c coloring) ∧
  (∀ c : ℝ, c < Real.sqrt 10 / 2 → ∃ coloring : Coloring c, ¬hasMonochromaticPair c coloring) :=
sorry

end NUMINAMATH_CALUDE_smallest_c_l2276_227645


namespace NUMINAMATH_CALUDE_backpack_solution_l2276_227635

/-- Represents the prices and quantities of backpacks -/
structure BackpackData where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- Conditions for the backpack problem -/
def backpack_conditions (d : BackpackData) : Prop :=
  d.price_a = 2 * d.price_b - 30 ∧
  2 * d.price_a + 3 * d.price_b = 255 ∧
  d.quantity_a + d.quantity_b = 200 ∧
  50 * d.quantity_a + 40 * d.quantity_b ≤ 8900 ∧
  d.quantity_a > 87

/-- The theorem stating the correct prices and possible purchasing plans -/
theorem backpack_solution :
  ∃ (d : BackpackData),
    backpack_conditions d ∧
    d.price_a = 60 ∧
    d.price_b = 45 ∧
    ((d.quantity_a = 88 ∧ d.quantity_b = 112) ∨
     (d.quantity_a = 89 ∧ d.quantity_b = 111) ∨
     (d.quantity_a = 90 ∧ d.quantity_b = 110)) :=
  sorry

end NUMINAMATH_CALUDE_backpack_solution_l2276_227635


namespace NUMINAMATH_CALUDE_class_average_mark_l2276_227624

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 30 →
  excluded_students = 5 →
  excluded_avg = 30 →
  remaining_avg = 90 →
  (total_students : ℝ) * (total_students * remaining_avg - excluded_students * excluded_avg) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l2276_227624


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l2276_227640

/- Define the factorial function -/
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

/- Define a function to get the largest power of 2 that divides n! -/
def largestPowerOf2DividingFactorial (n : ℕ) : ℕ :=
  (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32)

/- Define a function to get the ones digit of a number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

/- Theorem statement -/
theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  onesDigit (2^(largestPowerOf2DividingFactorial 32)) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l2276_227640


namespace NUMINAMATH_CALUDE_f_minimum_value_l2276_227643

def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ 1) ∧ (∃ x : ℝ, f x = 1) := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2276_227643


namespace NUMINAMATH_CALUDE_insufficient_info_for_production_l2276_227657

structure MachineRates where
  A : ℝ
  B : ℝ
  C : ℝ

def total_production (rates : MachineRates) (hours : ℝ) : ℝ :=
  hours * (rates.A + rates.B + rates.C)

theorem insufficient_info_for_production (P : ℝ) :
  ∀ (rates : MachineRates),
    7 * rates.A + 11 * rates.B = 305 →
    8 * rates.A + 22 * rates.C = P →
    ∃ (rates' : MachineRates),
      7 * rates'.A + 11 * rates'.B = 305 ∧
      8 * rates'.A + 22 * rates'.C = P ∧
      total_production rates 8 ≠ total_production rates' 8 :=
by
  sorry

#check insufficient_info_for_production

end NUMINAMATH_CALUDE_insufficient_info_for_production_l2276_227657


namespace NUMINAMATH_CALUDE_fly_path_distance_l2276_227676

theorem fly_path_distance (r : ℝ) (s : ℝ) (h1 : r = 58) (h2 : s = 80) : 
  let d := 2 * r
  let x := Real.sqrt (d^2 - s^2)
  d + x + s = 280 :=
by sorry

end NUMINAMATH_CALUDE_fly_path_distance_l2276_227676


namespace NUMINAMATH_CALUDE_bc_distances_l2276_227698

/-- Represents a circular road with four gas stations -/
structure CircularRoad where
  circumference : ℝ
  distAB : ℝ
  distAC : ℝ
  distCD : ℝ
  distDA : ℝ

/-- Theorem stating the possible distances between B and C -/
theorem bc_distances (road : CircularRoad)
  (h_circ : road.circumference = 100)
  (h_ab : road.distAB = 50)
  (h_ac : road.distAC = 40)
  (h_cd : road.distCD = 25)
  (h_da : road.distDA = 35) :
  ∃ (d1 d2 : ℝ), d1 = 10 ∧ d2 = 90 ∧
  (∀ (d : ℝ), (d = d1 ∨ d = d2) ↔ 
    (d = road.distAB - road.distAC ∨ 
     d = road.circumference - (road.distAB + road.distAC))) :=
by sorry

end NUMINAMATH_CALUDE_bc_distances_l2276_227698


namespace NUMINAMATH_CALUDE_women_at_soccer_game_l2276_227615

theorem women_at_soccer_game (adults : ℕ) (adult_women : ℕ) (student_surplus : ℕ) (male_students : ℕ)
  (h1 : adults = 1518)
  (h2 : adult_women = 536)
  (h3 : student_surplus = 525)
  (h4 : male_students = 1257) :
  adult_women + ((adults + student_surplus) - male_students) = 1322 :=
by sorry

end NUMINAMATH_CALUDE_women_at_soccer_game_l2276_227615


namespace NUMINAMATH_CALUDE_arithmetic_sequence_25th_term_l2276_227641

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1) * seq.d

theorem arithmetic_sequence_25th_term
  (seq : ArithmeticSequence)
  (h₃ : seq.nthTerm 3 = 7)
  (h₁₈ : seq.nthTerm 18 = 37) :
  seq.nthTerm 25 = 51 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_25th_term_l2276_227641


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l2276_227662

theorem complex_roots_theorem (p q r : ℂ) 
  (sum_eq : p + q + r = -1)
  (sum_prod_eq : p * q + p * r + q * r = -1)
  (prod_eq : p * q * r = -1) :
  (({p, q, r} : Set ℂ) = {-1, Complex.I, -Complex.I}) := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l2276_227662


namespace NUMINAMATH_CALUDE_expression_is_perfect_square_l2276_227681

/-- The expression is a perfect square when p equals 0.28 -/
theorem expression_is_perfect_square : 
  ∃ (x : ℝ), (12.86 * 12.86 + 12.86 * 0.28 + 0.14 * 0.14) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_is_perfect_square_l2276_227681


namespace NUMINAMATH_CALUDE_second_shirt_buttons_l2276_227616

/-- The number of buttons on the first type of shirt -/
def buttons_type1 : ℕ := 3

/-- The number of shirts ordered for each type -/
def shirts_per_type : ℕ := 200

/-- The total number of buttons used for all shirts -/
def total_buttons : ℕ := 1600

/-- The number of buttons on the second type of shirt -/
def buttons_type2 : ℕ := 5

theorem second_shirt_buttons :
  buttons_type2 * shirts_per_type + buttons_type1 * shirts_per_type = total_buttons :=
by sorry

end NUMINAMATH_CALUDE_second_shirt_buttons_l2276_227616


namespace NUMINAMATH_CALUDE_regular_pentagons_are_similar_l2276_227647

/-- A regular pentagon is a polygon with 5 sides of equal length and 5 angles of equal measure. -/
structure RegularPentagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Two shapes are similar if they have the same shape but not necessarily the same size. -/
def are_similar (p1 p2 : RegularPentagon) : Prop :=
  ∃ k : ℝ, k > 0 ∧ p1.side_length = k * p2.side_length

/-- Theorem: Any two regular pentagons are similar. -/
theorem regular_pentagons_are_similar (p1 p2 : RegularPentagon) : are_similar p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagons_are_similar_l2276_227647


namespace NUMINAMATH_CALUDE_gcd_problem_l2276_227614

/-- The GCD operation -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- The problem statement -/
theorem gcd_problem (n m k j : ℕ+) :
  gcd_op (gcd_op (16 * n) (20 * m)) (gcd_op (18 * k) (24 * j)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2276_227614


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l2276_227609

/-- Regular hexagon with vertices J and L -/
structure RegularHexagon where
  J : ℝ × ℝ
  L : ℝ × ℝ

/-- The area of a regular hexagon -/
def hexagon_area (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating the area of the specific regular hexagon -/
theorem specific_hexagon_area :
  let h : RegularHexagon := { J := (0, 0), L := (10, 2) }
  hexagon_area h = 156 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l2276_227609


namespace NUMINAMATH_CALUDE_kelly_at_sisters_house_l2276_227639

/-- Represents the duration of Kelly's vacation and activities --/
structure VacationDuration where
  total_days : ℕ
  travel_days : ℕ
  grandparents_days : ℕ
  brother_days : ℕ

/-- Calculates the number of days Kelly spent at her sister's house --/
def days_at_sisters_house (v : VacationDuration) : ℕ :=
  v.total_days - (v.travel_days + v.grandparents_days + v.brother_days)

/-- Theorem stating that Kelly spent 5 days at her sister's house --/
theorem kelly_at_sisters_house :
  let v : VacationDuration := {
    total_days := 21,
    travel_days := 6,
    grandparents_days := 5,
    brother_days := 5
  }
  days_at_sisters_house v = 5 := by sorry

end NUMINAMATH_CALUDE_kelly_at_sisters_house_l2276_227639


namespace NUMINAMATH_CALUDE_bert_grocery_spending_l2276_227660

/-- Represents Bert's spending scenario -/
structure BertSpending where
  initial_amount : ℚ
  hardware_fraction : ℚ
  dry_cleaning_amount : ℚ
  final_amount : ℚ

/-- Calculates the fraction spent at the grocery store -/
def grocery_fraction (b : BertSpending) : ℚ :=
  let remaining_before_grocery := b.initial_amount - (b.hardware_fraction * b.initial_amount) - b.dry_cleaning_amount
  let spent_at_grocery := remaining_before_grocery - b.final_amount
  spent_at_grocery / remaining_before_grocery

/-- Theorem stating that Bert spent 1/2 of his remaining money at the grocery store -/
theorem bert_grocery_spending (b : BertSpending) 
  (h1 : b.initial_amount = 52)
  (h2 : b.hardware_fraction = 1/4)
  (h3 : b.dry_cleaning_amount = 9)
  (h4 : b.final_amount = 15) :
  grocery_fraction b = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_bert_grocery_spending_l2276_227660


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2276_227651

/-- The volume of a cube given its surface area -/
theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 → volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2276_227651


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l2276_227675

theorem difference_of_squares_example : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l2276_227675


namespace NUMINAMATH_CALUDE_circle_mapping_l2276_227610

-- Define the complex plane
variable (z : ℂ)

-- Define the transformation function
def w (z : ℂ) : ℂ := 3 * z + 2

-- Define the original circle
def original_circle (z : ℂ) : Prop := z.re^2 + z.im^2 = 4

-- Define the mapped circle
def mapped_circle (w : ℂ) : Prop := (w.re - 2)^2 + w.im^2 = 36

-- Theorem statement
theorem circle_mapping :
  ∀ z, original_circle z → mapped_circle (w z) :=
sorry

end NUMINAMATH_CALUDE_circle_mapping_l2276_227610


namespace NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l2276_227654

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the point M
def M : ℝ × ℝ := (0, 1)

-- Define a line l that intersects Γ at two points
def l (k : ℝ) (x y : ℝ) : Prop := y = (k^2 - 1) / (4*k) * x - 1/2

-- Define the property that PQ is a diameter of the circumcircle of MPQ
def isPQDiameterOfCircumcircle (P Q : ℝ × ℝ) : Prop :=
  (P.1 - M.1) * (Q.1 - M.1) + (P.2 - M.2) * (Q.2 - M.2) = 0

theorem ellipse_intersection_fixed_point :
  ∀ (k : ℝ) (P Q : ℝ × ℝ),
    k ≠ 0 →
    Γ P.1 P.2 →
    Γ Q.1 Q.2 →
    l k P.1 P.2 →
    l k Q.1 Q.2 →
    isPQDiameterOfCircumcircle P Q →
    P ≠ M ∧ Q ≠ M →
    ∃ (x y : ℝ), l k x y ∧ x = 0 ∧ y = -1/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l2276_227654


namespace NUMINAMATH_CALUDE_exists_value_not_taken_by_phi_at_odd_l2276_227600

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- Predicate to check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

theorem exists_value_not_taken_by_phi_at_odd :
  ∃ m : ℕ, ∀ n : ℕ, isOdd n → phi n ≠ m := by sorry

end NUMINAMATH_CALUDE_exists_value_not_taken_by_phi_at_odd_l2276_227600


namespace NUMINAMATH_CALUDE_quadratic_roots_nm_l2276_227670

theorem quadratic_roots_nm (m n : ℝ) : 
  (∀ x, 2 * x^2 + m * x + n = 0 ↔ x = -2 ∨ x = 1) → 
  n^m = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_nm_l2276_227670


namespace NUMINAMATH_CALUDE_richard_david_age_diff_l2276_227688

-- Define the ages of the three sons
def david_age : ℕ := 14
def scott_age : ℕ := 6
def richard_age : ℕ := 20

-- Define the conditions
axiom david_scott_age_diff : david_age = scott_age + 8
axiom david_past_age : david_age = 11 + 3
axiom richard_future_age : richard_age + 8 = 2 * (scott_age + 8)

-- Define the theorem to prove
theorem richard_david_age_diff : richard_age = david_age + 6 := by
  sorry

end NUMINAMATH_CALUDE_richard_david_age_diff_l2276_227688


namespace NUMINAMATH_CALUDE_husband_catch_up_time_and_distance_l2276_227694

-- Define the problem parameters
def yolanda_initial_speed : ℝ := 20
def yolanda_second_speed : ℝ := 22
def yolanda_final_speed : ℝ := 18
def yolanda_first_distance : ℝ := 5
def yolanda_second_distance : ℝ := 8
def yolanda_third_distance : ℝ := 7
def yolanda_stop_time : ℝ := 12
def husband_speed : ℝ := 40
def husband_delay : ℝ := 15
def route_difference : ℝ := 10

-- Define the theorem
theorem husband_catch_up_time_and_distance : 
  let yolanda_total_distance := yolanda_first_distance + yolanda_second_distance + yolanda_third_distance
  let husband_distance := yolanda_total_distance - route_difference
  let yolanda_travel_time := yolanda_first_distance / yolanda_initial_speed * 60 + 
                             yolanda_second_distance / yolanda_second_speed * 60 + 
                             yolanda_third_distance / yolanda_final_speed * 60 + 
                             yolanda_stop_time
  let husband_travel_time := husband_distance / husband_speed * 60
  husband_distance = 10 ∧ husband_travel_time + husband_delay = 30 := by
    sorry


end NUMINAMATH_CALUDE_husband_catch_up_time_and_distance_l2276_227694


namespace NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l2276_227677

/-- The slope angle of a line with parametric equations x = 2 + t and y = 1 + (√3/3)t is π/6 -/
theorem slope_angle_of_parametric_line : 
  ∀ (t : ℝ), 
  let x := 2 + t
  let y := 1 + (Real.sqrt 3 / 3) * t
  let slope := (Real.sqrt 3 / 3)
  let slope_angle := Real.arctan slope
  slope_angle = π / 6 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l2276_227677


namespace NUMINAMATH_CALUDE_notepad_cost_l2276_227628

/-- Given the total cost of notepads, pages per notepad, and total pages bought,
    calculate the cost of each notepad. -/
theorem notepad_cost (total_cost : ℚ) (pages_per_notepad : ℕ) (total_pages : ℕ) :
  total_cost = 10 →
  pages_per_notepad = 60 →
  total_pages = 480 →
  (total_cost / (total_pages / pages_per_notepad : ℚ)) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_notepad_cost_l2276_227628


namespace NUMINAMATH_CALUDE_intersection_sum_l2276_227622

theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, x = (1/3) * y + c ↔ y = (1/3) * x + d) → 
  (3 = (1/3) * 6 + c ∧ 6 = (1/3) * 3 + d) → 
  c + d = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l2276_227622


namespace NUMINAMATH_CALUDE_negation_of_exponential_inequality_l2276_227623

theorem negation_of_exponential_inequality (P : Prop) :
  (P ↔ ∀ x : ℝ, x > 0 → Real.exp x > x + 1) →
  (¬P ↔ ∃ x : ℝ, x > 0 ∧ Real.exp x ≤ x + 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exponential_inequality_l2276_227623


namespace NUMINAMATH_CALUDE_det_dilation_matrix_3d_l2276_227633

/-- A matrix representing a dilation centered at the origin with scale factor 4 -/
def dilation_matrix (n : ℕ) (k : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal (λ _ => k)

theorem det_dilation_matrix_3d :
  let E := dilation_matrix 3 4
  Matrix.det E = 64 := by sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_3d_l2276_227633


namespace NUMINAMATH_CALUDE_janes_calculation_l2276_227655

theorem janes_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 17) 
  (h2 : x - y - z = 5) : 
  x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_janes_calculation_l2276_227655


namespace NUMINAMATH_CALUDE_group_b_more_stable_l2276_227649

/-- Represents a group of data with its variance -/
structure DataGroup where
  variance : ℝ

/-- Defines stability comparison between two data groups -/
def more_stable (a b : DataGroup) : Prop := a.variance < b.variance

/-- Theorem stating that Group B is more stable than Group A given their variances -/
theorem group_b_more_stable (group_a group_b : DataGroup)
  (h1 : group_a.variance = 0.2)
  (h2 : group_b.variance = 0.03) :
  more_stable group_b group_a := by
  sorry

#check group_b_more_stable

end NUMINAMATH_CALUDE_group_b_more_stable_l2276_227649


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2276_227625

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def is_nonagon (n : ℕ) : Prop := n = 9

theorem nonagon_diagonals :
  ∀ n : ℕ, is_nonagon n → num_diagonals n = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2276_227625


namespace NUMINAMATH_CALUDE_seven_digit_multiple_l2276_227636

theorem seven_digit_multiple : ∀ (A B C : ℕ),
  (A < 10 ∧ B < 10 ∧ C < 10) →
  (∃ (k₁ k₂ k₃ : ℕ), 
    25000000 + A * 100000 + B * 10000 + 3300 + C = 8 * k₁ ∧
    25000000 + A * 100000 + B * 10000 + 3300 + C = 9 * k₂ ∧
    25000000 + A * 100000 + B * 10000 + 3300 + C = 11 * k₃) →
  A + B + C = 14 := by
sorry

end NUMINAMATH_CALUDE_seven_digit_multiple_l2276_227636


namespace NUMINAMATH_CALUDE_equation_solution_l2276_227607

theorem equation_solution (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2276_227607


namespace NUMINAMATH_CALUDE_min_rectangles_to_cover_square_l2276_227684

/-- The width of the rectangle. -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle. -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle. -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the square that can be covered exactly by the rectangles. -/
def square_side : ℕ := rectangle_area

/-- The area of the square. -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square. -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem min_rectangles_to_cover_square : 
  num_rectangles = 12 ∧ 
  square_area % rectangle_area = 0 ∧
  ∀ n : ℕ, n < square_side → n * n % rectangle_area ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_min_rectangles_to_cover_square_l2276_227684


namespace NUMINAMATH_CALUDE_system_solution_l2276_227602

theorem system_solution :
  ∀ x y z : ℝ,
  (x + y - 2 + 4*x*y = 0 ∧
   y + z - 2 + 4*y*z = 0 ∧
   z + x - 2 + 4*z*x = 0) ↔
  ((x = -1 ∧ y = -1 ∧ z = -1) ∨
   (x = 1/2 ∧ y = 1/2 ∧ z = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2276_227602


namespace NUMINAMATH_CALUDE_find_A_l2276_227611

theorem find_A (A B C : ℚ) 
  (h1 : A = (1 / 2) * B) 
  (h2 : B = (3 / 4) * C) 
  (h3 : A + C = 55) : 
  A = 15 := by
sorry

end NUMINAMATH_CALUDE_find_A_l2276_227611


namespace NUMINAMATH_CALUDE_experimental_plans_count_l2276_227691

/-- The number of ways to select an even number of elements from a set of 6 elements -/
def evenSelectionA : ℕ := Finset.sum (Finset.filter (λ k => k % 2 = 0) (Finset.range 7)) (λ k => Nat.choose 6 k)

/-- The number of ways to select at least 2 elements from a set of 4 elements -/
def atLeastTwoSelectionB : ℕ := Finset.sum (Finset.range 3) (λ k => Nat.choose 4 (k + 2))

/-- The total number of experimental plans -/
def totalExperimentalPlans : ℕ := evenSelectionA * atLeastTwoSelectionB

theorem experimental_plans_count : totalExperimentalPlans = 352 := by
  sorry

end NUMINAMATH_CALUDE_experimental_plans_count_l2276_227691


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l2276_227695

def is_composite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ a + b = n

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → sum_of_two_composites n) ∧
  ¬sum_of_two_composites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l2276_227695


namespace NUMINAMATH_CALUDE_unique_solution_l2276_227679

/-- Represents the quantities and prices of two batches of products --/
structure BatchData where
  quantity1 : ℕ
  quantity2 : ℕ
  price1 : ℚ
  price2 : ℚ

/-- Checks if the given batch data satisfies the problem conditions --/
def satisfiesConditions (data : BatchData) : Prop :=
  data.quantity1 * data.price1 = 4000 ∧
  data.quantity2 * data.price2 = 8800 ∧
  data.quantity2 = 2 * data.quantity1 ∧
  data.price2 = data.price1 + 4

/-- Theorem stating that the only solution satisfying the conditions is 100 and 200 units --/
theorem unique_solution :
  ∀ data : BatchData, satisfiesConditions data →
    data.quantity1 = 100 ∧ data.quantity2 = 200 := by
  sorry

#check unique_solution

end NUMINAMATH_CALUDE_unique_solution_l2276_227679


namespace NUMINAMATH_CALUDE_num_true_props_l2276_227671

-- Define the propositions as boolean variables
def prop1 : Bool := true  -- All lateral edges of a regular pyramid are equal
def prop2 : Bool := false -- The lateral faces of a right prism are all congruent rectangles
def prop3 : Bool := true  -- The generatrix of a cylinder is perpendicular to the base
def prop4 : Bool := true  -- The section obtained by cutting a cone with a plane passing through the axis of rotation is always a congruent isosceles triangle

-- Define a function to count true propositions
def countTrueProps (p1 p2 p3 p4 : Bool) : Nat :=
  (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) + (if p4 then 1 else 0)

-- Theorem stating that the number of true propositions is 3
theorem num_true_props : countTrueProps prop1 prop2 prop3 prop4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_num_true_props_l2276_227671


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_0813_l2276_227626

theorem scientific_notation_of_0_0813 :
  ∃ (a : ℝ) (n : ℤ), 0.0813 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -2 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_0813_l2276_227626


namespace NUMINAMATH_CALUDE_caroling_boys_count_l2276_227618

/-- The number of boys who received 1 orange each -/
def boys_with_one_orange : ℕ := 2

/-- The number of boys who received 2 oranges each -/
def boys_with_two_oranges : ℕ := 4

/-- The number of boys who received 4 oranges -/
def boys_with_four_oranges : ℕ := 1

/-- The number of oranges received by boys with known names -/
def oranges_known_boys : ℕ := boys_with_one_orange + 2 * boys_with_two_oranges + 4 * boys_with_four_oranges

/-- The total number of oranges received by all boys -/
def total_oranges : ℕ := 23

/-- The number of oranges each of the other boys received -/
def oranges_per_other_boy : ℕ := 3

theorem caroling_boys_count : ∃ (n : ℕ), 
  n = boys_with_one_orange + boys_with_two_oranges + boys_with_four_oranges + 
      (total_oranges - oranges_known_boys) / oranges_per_other_boy ∧ 
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_caroling_boys_count_l2276_227618


namespace NUMINAMATH_CALUDE_corrected_mean_l2276_227644

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (wrong_value correct_value : ℝ) :
  n = 50 ∧ original_mean = 36 ∧ wrong_value = 23 ∧ correct_value = 45 →
  (n * original_mean + (correct_value - wrong_value)) / n = 36.44 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l2276_227644


namespace NUMINAMATH_CALUDE_wire_length_proof_l2276_227678

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 20 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 70 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l2276_227678


namespace NUMINAMATH_CALUDE_stork_bird_difference_l2276_227690

/-- Given initial birds, storks, and additional birds, calculate the difference between storks and total birds -/
theorem stork_bird_difference (initial_birds : ℕ) (storks : ℕ) (additional_birds : ℕ) : 
  initial_birds = 2 → storks = 6 → additional_birds = 3 →
  storks - (initial_birds + additional_birds) = 1 := by
  sorry

end NUMINAMATH_CALUDE_stork_bird_difference_l2276_227690


namespace NUMINAMATH_CALUDE_division_simplification_l2276_227669

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  -6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l2276_227669


namespace NUMINAMATH_CALUDE_dan_picked_nine_apples_l2276_227665

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The difference between Dan's and Benny's apple count -/
def difference : ℕ := 7

/-- The number of apples Dan picked -/
def dan_apples : ℕ := benny_apples + difference

theorem dan_picked_nine_apples : dan_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_dan_picked_nine_apples_l2276_227665


namespace NUMINAMATH_CALUDE_function_composition_theorem_l2276_227605

/-- Given two functions f and g, with f(x) = Ax - 3B² and g(x) = Bx + C,
    where B ≠ 0 and f(g(1)) = 0, prove that A = 3B² / (B + C),
    assuming B + C ≠ 0. -/
theorem function_composition_theorem (A B C : ℝ) 
  (hB : B ≠ 0) (hBC : B + C ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ A * x - 3 * B^2
  let g : ℝ → ℝ := λ x ↦ B * x + C
  f (g 1) = 0 → A = 3 * B^2 / (B + C) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_theorem_l2276_227605


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_sum_l2276_227663

theorem seventh_root_of_unity_sum (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  ∃ (sign : Bool), z + z^2 + z^4 = (-1 + (if sign then 1 else -1) * Complex.I * Real.sqrt 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_sum_l2276_227663


namespace NUMINAMATH_CALUDE_sum_is_non_horizontal_line_l2276_227642

/-- A parabola is defined by its coefficients a, b, and c. -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Function f is the original parabola translated 3 units to the right. -/
def f (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - 3)^2 + p.b * (x - 3) + p.c

/-- Function g is the reflected parabola translated 3 units to the left. -/
def g (p : Parabola) (x : ℝ) : ℝ :=
  -p.a * (x + 3)^2 - p.b * (x + 3) - p.c

/-- The sum of f and g is a non-horizontal line. -/
theorem sum_is_non_horizontal_line (p : Parabola) :
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x, f p x + g p x = m * x + k := by
  sorry

end NUMINAMATH_CALUDE_sum_is_non_horizontal_line_l2276_227642


namespace NUMINAMATH_CALUDE_disinfectant_problem_l2276_227696

/-- Represents the price and volume of a disinfectant brand -/
structure DisinfectantBrand where
  price : ℝ
  volume : ℝ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  brand1_bottles : ℕ
  brand2_bottles : ℕ

/-- Calculates the total cost of a purchase plan -/
def totalCost (brand1 brand2 : DisinfectantBrand) (plan : PurchasePlan) : ℝ :=
  brand1.price * plan.brand1_bottles + brand2.price * plan.brand2_bottles

/-- Calculates the total volume of a purchase plan -/
def totalVolume (brand1 brand2 : DisinfectantBrand) (plan : PurchasePlan) : ℝ :=
  brand1.volume * plan.brand1_bottles + brand2.volume * plan.brand2_bottles

/-- Theorem stating the properties of the disinfectant purchase problem -/
theorem disinfectant_problem (brand1 brand2 : DisinfectantBrand) 
  (h1 : brand1.volume = 200)
  (h2 : brand2.volume = 500)
  (h3 : totalCost brand1 brand2 { brand1_bottles := 3, brand2_bottles := 2 } = 80)
  (h4 : totalCost brand1 brand2 { brand1_bottles := 1, brand2_bottles := 4 } = 110)
  (h5 : ∃ (plan : PurchasePlan), totalVolume brand1 brand2 plan = 4000 ∧ 
        plan.brand1_bottles > 0 ∧ plan.brand2_bottles > 0)
  (h6 : ∃ (plan : PurchasePlan), totalCost brand1 brand2 plan = 2500)
  (h7 : (2500 / (1000 * 10 : ℝ)) * (brand1.volume * brand1.price + brand2.volume * brand2.price) = 5000) :
  brand1.price = 10 ∧ brand2.price = 25 ∧ 
  (2500 / (1000 * 10 : ℝ)) * (brand1.volume * brand1.price + brand2.volume * brand2.price) / 1000 = 5 := by
  sorry


end NUMINAMATH_CALUDE_disinfectant_problem_l2276_227696


namespace NUMINAMATH_CALUDE_sum_divisible_by_three_combinations_l2276_227661

/-- The number of integers from 1 to 300 that give remainder 0, 1, or 2 when divided by 3 -/
def count_mod_3 : ℕ := 100

/-- The total number of ways to select 3 numbers from 1 to 300 such that their sum is divisible by 3 -/
def total_combinations : ℕ := 1485100

/-- The number of ways to choose 3 elements from a set of size n -/
def choose (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 6

theorem sum_divisible_by_three_combinations :
  3 * choose count_mod_3 + count_mod_3^3 = total_combinations :=
sorry

end NUMINAMATH_CALUDE_sum_divisible_by_three_combinations_l2276_227661


namespace NUMINAMATH_CALUDE_harriets_age_l2276_227659

/-- Given information about Peter and Harriet's ages, prove Harriet's current age -/
theorem harriets_age (peter_mother_age : ℕ) (peter_age : ℕ) (harriet_age : ℕ) : 
  peter_mother_age = 60 →
  peter_age = peter_mother_age / 2 →
  peter_age + 4 = 2 * (harriet_age + 4) →
  harriet_age = 13 := by sorry

end NUMINAMATH_CALUDE_harriets_age_l2276_227659


namespace NUMINAMATH_CALUDE_actual_distance_travelled_l2276_227667

theorem actual_distance_travelled (speed1 speed2 : ℝ) (extra_distance : ℝ) 
  (h1 : speed1 = 10)
  (h2 : speed2 = 14)
  (h3 : extra_distance = 20)
  (h4 : ∀ d : ℝ, d / speed1 = (d + extra_distance) / speed2) :
  ∃ d : ℝ, d = 50 ∧ d / speed1 = (d + extra_distance) / speed2 := by
sorry

end NUMINAMATH_CALUDE_actual_distance_travelled_l2276_227667


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2276_227604

theorem complex_fraction_simplification :
  let z : ℂ := (7 + 8*I) / (3 - 4*I)
  z = 53/25 + 52/25 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2276_227604


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l2276_227692

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x^2 + 1/x^2 = 2) : x^4 + 1/x^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l2276_227692


namespace NUMINAMATH_CALUDE_part_one_part_two_l2276_227656

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1) + Real.sqrt (2 - x)}
def B (a : ℝ) : Set ℝ := {y | ∃ x ≥ a, y = 2^x}

-- Part I
theorem part_one : 
  (Set.univ \ A) ∩ B 2 = Set.Ici 4 := by sorry

-- Part II
theorem part_two : 
  ∀ a : ℝ, (Set.univ \ A) ∪ B a = Set.univ ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2276_227656


namespace NUMINAMATH_CALUDE_meaningful_fraction_l2276_227638

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l2276_227638


namespace NUMINAMATH_CALUDE_real_root_of_cubic_l2276_227658

/-- Given a cubic polynomial with real coefficients c and d, 
    if -3 + 2i is a root, then 53/5 is the real root. -/
theorem real_root_of_cubic (c d : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (fun x : ℂ => c * x ^ 3 - x ^ 2 + d * x + 30) (-3 + 2 * Complex.I) = 0 →
  (fun x : ℝ => c * x ^ 3 - x ^ 2 + d * x + 30) (53 / 5) = 0 :=
by sorry

end NUMINAMATH_CALUDE_real_root_of_cubic_l2276_227658


namespace NUMINAMATH_CALUDE_system_solution_l2276_227620

theorem system_solution : ∃ (x y : ℝ), (4 * x - y = 7) ∧ (3 * x + 4 * y = 10) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2276_227620


namespace NUMINAMATH_CALUDE_honey_distribution_l2276_227612

theorem honey_distribution (bottles : ℕ) (weight_per_bottle : ℚ) (share_per_neighbor : ℚ) :
  bottles = 4 →
  weight_per_bottle = 3 →
  share_per_neighbor = 3/4 →
  (bottles * weight_per_bottle) / share_per_neighbor = 16 := by
sorry

end NUMINAMATH_CALUDE_honey_distribution_l2276_227612


namespace NUMINAMATH_CALUDE_dihedral_angle_BAC_ACD_is_120_degrees_l2276_227617

-- Define a unit cube
def UnitCube := Set (ℝ × ℝ × ℝ)

-- Define a function to calculate the dihedral angle between two faces of a cube
def dihedralAngle (cube : UnitCube) (face1 face2 : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define the specific faces for the B-A₁C-D dihedral angle
def faceBAC (cube : UnitCube) : Set (ℝ × ℝ × ℝ) := sorry
def faceACD (cube : UnitCube) : Set (ℝ × ℝ × ℝ) := sorry

-- State the theorem
theorem dihedral_angle_BAC_ACD_is_120_degrees (cube : UnitCube) : 
  dihedralAngle cube (faceBAC cube) (faceACD cube) = 120 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_dihedral_angle_BAC_ACD_is_120_degrees_l2276_227617


namespace NUMINAMATH_CALUDE_park_width_l2276_227601

/-- Given a rectangular park with specified length, tree density, and total number of trees,
    prove that the width of the park is as calculated. -/
theorem park_width (length : ℝ) (tree_density : ℝ) (total_trees : ℝ) (width : ℝ) : 
  length = 1000 →
  tree_density = 1 / 20 →
  total_trees = 100000 →
  width = total_trees / (length * tree_density) →
  width = 2000 :=
by sorry

end NUMINAMATH_CALUDE_park_width_l2276_227601


namespace NUMINAMATH_CALUDE_initial_apples_count_l2276_227689

def cafeteria_apples (apples_handed_out : ℕ) (apples_per_pie : ℕ) (pies_made : ℕ) : ℕ :=
  apples_handed_out + apples_per_pie * pies_made

theorem initial_apples_count :
  cafeteria_apples 41 5 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_count_l2276_227689


namespace NUMINAMATH_CALUDE_altitude_equation_l2276_227672

/-- Given a triangle ABC with side equations:
    AB: 3x + 4y + 12 = 0
    BC: 4x - 3y + 16 = 0
    CA: 2x + y - 2 = 0
    The altitude from A to BC has the equation x - 2y + 4 = 0 -/
theorem altitude_equation (x y : ℝ) :
  (3 * x + 4 * y + 12 = 0) →  -- AB
  (4 * x - 3 * y + 16 = 0) →  -- BC
  (2 * x + y - 2 = 0) →       -- CA
  (x - 2 * y + 4 = 0)         -- Altitude from A to BC
:= by sorry

end NUMINAMATH_CALUDE_altitude_equation_l2276_227672


namespace NUMINAMATH_CALUDE_triangle_area_quadrilateral_area_n_gon_area_l2276_227693

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

def point (m i : ℕ) : ℝ × ℝ :=
  (fibonacci (m + 2 * i - 1), fibonacci (m + 2 * i))

def polygon_area (n m : ℕ) : ℝ :=
  let vertices := List.range n |>.map (point m)
  -- Area calculation using Shoelace formula
  sorry

theorem triangle_area (m : ℕ) :
  polygon_area 3 m = 0.5 := by sorry

theorem quadrilateral_area (m : ℕ) :
  polygon_area 4 m = 2.5 := by sorry

theorem n_gon_area (n m : ℕ) (h : n ≥ 3) :
  polygon_area n m = (fibonacci (2 * n - 2) - n + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_quadrilateral_area_n_gon_area_l2276_227693


namespace NUMINAMATH_CALUDE_odd_integer_divisor_l2276_227682

theorem odd_integer_divisor (n : ℕ) (hn : Odd n) 
  (hxy : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = 4 / n) :
  ∃ (k : ℕ), (4 * k - 1) ∣ n := by
sorry

end NUMINAMATH_CALUDE_odd_integer_divisor_l2276_227682


namespace NUMINAMATH_CALUDE_negation_quadratic_inequality_l2276_227634

theorem negation_quadratic_inequality (x : ℝ) :
  (x^2 + x - 6 < 0) → (x ≤ 2) :=
sorry

#check negation_quadratic_inequality

end NUMINAMATH_CALUDE_negation_quadratic_inequality_l2276_227634


namespace NUMINAMATH_CALUDE_ship_food_supply_l2276_227646

/-- Calculates the remaining food supply on a ship after a specific consumption pattern. -/
theorem ship_food_supply (initial_supply : ℝ) : 
  initial_supply = 400 →
  (initial_supply - 2/5 * initial_supply) - 3/5 * (initial_supply - 2/5 * initial_supply) = 96 := by
  sorry

#check ship_food_supply

end NUMINAMATH_CALUDE_ship_food_supply_l2276_227646


namespace NUMINAMATH_CALUDE_rectangle_area_18_l2276_227683

/-- A rectangle with base twice the height and area equal to perimeter has area 18 -/
theorem rectangle_area_18 (h : ℝ) (b : ℝ) (area : ℝ) (perimeter : ℝ) : 
  b = 2 * h →                        -- base is twice the height
  area = b * h →                     -- area formula
  perimeter = 2 * (b + h) →          -- perimeter formula
  area = perimeter →                 -- area is numerically equal to perimeter
  area = 18 :=                       -- prove that area is 18
by sorry

end NUMINAMATH_CALUDE_rectangle_area_18_l2276_227683


namespace NUMINAMATH_CALUDE_quadratic_solution_l2276_227648

theorem quadratic_solution (h : 108 * (3/4)^2 - 35 * (3/4) - 77 = 0) :
  108 * (-23/54)^2 - 35 * (-23/54) - 77 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2276_227648


namespace NUMINAMATH_CALUDE_m_value_proof_l2276_227673

theorem m_value_proof (m : ℝ) : 
  (m > 0) ∧ 
  (∀ x : ℝ, (x / (x - 1) < 0 → 0 < x ∧ x < m)) ∧ 
  (∃ x : ℝ, 0 < x ∧ x < m ∧ x / (x - 1) ≥ 0) →
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_m_value_proof_l2276_227673


namespace NUMINAMATH_CALUDE_meaningful_expression_l2276_227652

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 2 / Real.sqrt (x - 1)) ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2276_227652


namespace NUMINAMATH_CALUDE_systematic_sampling_in_school_l2276_227664

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | RandomDraw
  | RandomSampling
  | SystematicSampling

/-- Represents a school with classes and students -/
structure School where
  num_classes : Nat
  students_per_class : Nat
  student_numbers : Finset Nat

/-- Represents a sampling scenario -/
structure SamplingScenario where
  school : School
  selected_number : Nat

/-- Determines the sampling method used in a given scenario -/
def determineSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  sorry

theorem systematic_sampling_in_school (scenario : SamplingScenario) :
  scenario.school.num_classes = 35 →
  scenario.school.students_per_class = 56 →
  scenario.school.student_numbers = Finset.range 56 →
  scenario.selected_number = 14 →
  determineSamplingMethod scenario = SamplingMethod.SystematicSampling :=
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_in_school_l2276_227664


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_S_l2276_227680

def S : Set ℤ := {x | ∃ n : ℤ, x = n^5 - 5*n^3 + 4*n ∧ ¬(3 ∣ n)}

theorem largest_common_divisor_of_S : 
  ∀ k : ℤ, (∀ x ∈ S, k ∣ x) → k ≤ 360 ∧ 
  ∀ x ∈ S, 360 ∣ x :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_S_l2276_227680


namespace NUMINAMATH_CALUDE_area_ratio_squares_l2276_227653

/-- Given squares A, B, and C with specified properties, prove the ratio of areas of A to C -/
theorem area_ratio_squares (sideA sideB sideC : ℝ) : 
  sideA * 4 = 16 →  -- Perimeter of A is 16
  sideB * 4 = 40 →  -- Perimeter of B is 40
  sideC = 1.5 * sideA →  -- Side of C is 1.5 times side of A
  (sideA ^ 2) / (sideC ^ 2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_squares_l2276_227653


namespace NUMINAMATH_CALUDE_claras_weight_l2276_227630

theorem claras_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight + clara_weight = 240)
  (h2 : clara_weight - alice_weight = clara_weight / 3) : 
  clara_weight = 144 := by
sorry

end NUMINAMATH_CALUDE_claras_weight_l2276_227630


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l2276_227629

theorem boys_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neither_happy_nor_sad : ℕ) (total_boys : ℕ) (total_girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neither_happy_nor_sad = 20 →
  total_boys = 19 →
  total_girls = 41 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = happy_children + sad_children + neither_happy_nor_sad →
  total_children = total_boys + total_girls →
  (total_boys - (happy_boys + (sad_children - sad_girls))) = 7 :=
by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l2276_227629


namespace NUMINAMATH_CALUDE_imaginary_part_of_z2_l2276_227686

theorem imaginary_part_of_z2 (z₁ : ℂ) (h : z₁ = 1 - 2*I) :
  Complex.im ((z₁ + 1) / (z₁ - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z2_l2276_227686


namespace NUMINAMATH_CALUDE_square_side_length_l2276_227637

-- Define the right triangle PQR
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the square on the hypotenuse
structure SquareOnHypotenuse where
  triangle : RightTriangle
  side_length : ℝ
  on_hypotenuse : side_length ≤ triangle.hypotenuse
  vertex_on_legs : ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ triangle.leg1 ∧ 0 ≤ y ∧ y ≤ triangle.leg2 ∧
    x^2 + y^2 = side_length^2

-- Theorem statement
theorem square_side_length (t : RightTriangle) (s : SquareOnHypotenuse) 
  (h1 : t.leg1 = 5) (h2 : t.leg2 = 12) (h3 : s.triangle = t) :
  s.side_length = 480.525 / 101.925 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2276_227637


namespace NUMINAMATH_CALUDE_six_circle_arrangement_possible_l2276_227608

/-- A configuration of 6 circles in a plane -/
structure CircleConfiguration where
  positions : Fin 6 → ℝ × ℝ

/-- Predicate to check if a configuration allows a 7th circle to touch all 6 -/
def ValidConfiguration (config : CircleConfiguration) : Prop :=
  ∃ (center : ℝ × ℝ), ∀ i : Fin 6, 
    let (x, y) := config.positions i
    let (cx, cy) := center
    (x - cx)^2 + (y - cy)^2 = 4  -- Assuming unit radius for simplicity

/-- Predicate to check if a configuration can be achieved without measurements or lifting -/
def AchievableWithoutMeasurement (config : CircleConfiguration) : Prop :=
  sorry  -- This would require a formal definition of "without measurement"

theorem six_circle_arrangement_possible :
  ∃ (config : CircleConfiguration), 
    ValidConfiguration config ∧ AchievableWithoutMeasurement config :=
sorry

end NUMINAMATH_CALUDE_six_circle_arrangement_possible_l2276_227608


namespace NUMINAMATH_CALUDE_floor_ceil_problem_l2276_227674

theorem floor_ceil_problem : (⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉) * 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_problem_l2276_227674


namespace NUMINAMATH_CALUDE_function_through_point_l2276_227606

/-- Given a function f(x) = x^α that passes through (2, √2), prove f(9) = 3 -/
theorem function_through_point (α : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = x ^ α) → f 2 = Real.sqrt 2 → f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_through_point_l2276_227606


namespace NUMINAMATH_CALUDE_rational_division_l2276_227650

theorem rational_division (x : ℚ) : (-2 : ℚ) / x = 8 → x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_rational_division_l2276_227650


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2276_227603

theorem multiplication_subtraction_equality : 75 * 1414 - 25 * 1414 = 70700 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2276_227603


namespace NUMINAMATH_CALUDE_trapezium_side_length_l2276_227619

/-- Theorem: In a trapezium with one parallel side of length 12 cm, a distance between parallel sides
    of 14 cm, and an area of 196 square centimeters, the length of the other parallel side is 16 cm. -/
theorem trapezium_side_length 
  (side1 : ℝ) 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : side1 = 12) 
  (h2 : height = 14) 
  (h3 : area = 196) 
  (h4 : area = (side1 + side2) * height / 2) : 
  side2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l2276_227619


namespace NUMINAMATH_CALUDE_smallest_number_l2276_227697

def number_set : Set ℤ := {-3, 2, -2, 0}

theorem smallest_number : ∀ x ∈ number_set, -3 ≤ x := by sorry

end NUMINAMATH_CALUDE_smallest_number_l2276_227697


namespace NUMINAMATH_CALUDE_strawberry_area_l2276_227613

/-- The area of strawberries in a circular garden -/
theorem strawberry_area (d : ℝ) (h1 : d = 16) : ∃ (A : ℝ), A = 8 * Real.pi ∧ A = (1/8) * Real.pi * d^2 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_area_l2276_227613


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l2276_227699

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 2

-- State the theorem
theorem f_sum_symmetric (a b m : ℝ) : 
  f a b (-2) = m → f a b 2 + f a b (-2) = -4 := by
sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l2276_227699


namespace NUMINAMATH_CALUDE_prob_sum_four_twice_l2276_227668

/-- A die with 3 sides --/
def ThreeSidedDie : Type := Fin 3

/-- The sum of two dice rolls --/
def diceSum (d1 d2 : ThreeSidedDie) : Nat :=
  d1.val + d2.val + 2

/-- The probability of rolling a sum of 4 with two 3-sided dice --/
def probSumFour : ℚ :=
  3 / 9

/-- The probability of rolling a sum of 4 twice in a row with two 3-sided dice --/
theorem prob_sum_four_twice : probSumFour * probSumFour = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_four_twice_l2276_227668


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2276_227621

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 7/6 -/
theorem infinite_series_sum : 
  ∑' n : ℕ+, (3 * n - 2 : ℚ) / (n * (n + 1) * (n + 3)) = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2276_227621


namespace NUMINAMATH_CALUDE_pyramid_cross_section_distance_l2276_227666

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  -- Add any necessary fields

/-- Represents a cross-section of the pyramid -/
structure CrossSection where
  area : ℝ
  distance_from_apex : ℝ

theorem pyramid_cross_section_distance
  (pyramid : RightHexagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h : ℝ) :
  cs1.area = 125 * Real.sqrt 3 →
  cs2.area = 500 * Real.sqrt 3 →
  cs2.distance_from_apex - cs1.distance_from_apex = 10 →
  cs2.distance_from_apex = h →
  h = 20 := by
  sorry

#check pyramid_cross_section_distance

end NUMINAMATH_CALUDE_pyramid_cross_section_distance_l2276_227666


namespace NUMINAMATH_CALUDE_power_of_power_eq_expanded_power_l2276_227687

theorem power_of_power_eq_expanded_power (x : ℝ) : (2 * x^2)^3 = 8 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_eq_expanded_power_l2276_227687


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2276_227685

theorem decimal_sum_to_fraction :
  (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 : ℚ) = 12345 / 160000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2276_227685


namespace NUMINAMATH_CALUDE_stock_value_order_l2276_227627

def initial_investment : ℝ := 200

def omega_year1_change : ℝ := 1.15
def bravo_year1_change : ℝ := 0.70
def zeta_year1_change : ℝ := 1.00

def omega_year2_change : ℝ := 0.90
def bravo_year2_change : ℝ := 1.30
def zeta_year2_change : ℝ := 1.00

def omega_final : ℝ := initial_investment * omega_year1_change * omega_year2_change
def bravo_final : ℝ := initial_investment * bravo_year1_change * bravo_year2_change
def zeta_final : ℝ := initial_investment * zeta_year1_change * zeta_year2_change

theorem stock_value_order : bravo_final < zeta_final ∧ zeta_final < omega_final :=
by sorry

end NUMINAMATH_CALUDE_stock_value_order_l2276_227627


namespace NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l2276_227631

theorem largest_prime_divisor_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 10 + Nat.factorial 11) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 10 + Nat.factorial 11) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l2276_227631
