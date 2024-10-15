import Mathlib

namespace NUMINAMATH_CALUDE_cube_division_l1974_197442

theorem cube_division (n : ℕ) (h1 : n ≥ 6) (h2 : Even n) :
  ∃ (m : ℕ), m^3 = (3 * n * (n - 2)) / 4 + 2 :=
sorry

end NUMINAMATH_CALUDE_cube_division_l1974_197442


namespace NUMINAMATH_CALUDE_ending_number_of_range_l1974_197491

theorem ending_number_of_range (n : ℕ) (h1 : n = 10) (h2 : ∀ k ∈ Finset.range n, 15 + 5 * k ∣ 5) :
  15 + 5 * (n - 1) = 60 :=
sorry

end NUMINAMATH_CALUDE_ending_number_of_range_l1974_197491


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l1974_197484

theorem rectangular_field_dimensions (m : ℝ) : 
  m > 3 ∧ (2 * m + 9) * (m - 3) = 55 →
  m = (-3 + Real.sqrt 665) / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l1974_197484


namespace NUMINAMATH_CALUDE_wage_increase_constant_wage_increase_l1974_197466

/-- Represents the regression equation for worker's wage based on labor productivity -/
def wage_equation (x : ℝ) : ℝ := 10 + 70 * x

/-- Proves that an increase of 1 in labor productivity results in an increase of 70 in wage -/
theorem wage_increase (x : ℝ) : wage_equation (x + 1) - wage_equation x = 70 := by
  sorry

/-- Proves that the wage increase is constant for any labor productivity value -/
theorem constant_wage_increase (x y : ℝ) : 
  wage_equation (x + 1) - wage_equation x = wage_equation (y + 1) - wage_equation y := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_constant_wage_increase_l1974_197466


namespace NUMINAMATH_CALUDE_quartic_sum_to_quadratic_sum_l1974_197477

theorem quartic_sum_to_quadratic_sum (x : ℝ) (h : 45 = x^4 + 1/x^4) : 
  x^2 + 1/x^2 = Real.sqrt 47 := by
  sorry

end NUMINAMATH_CALUDE_quartic_sum_to_quadratic_sum_l1974_197477


namespace NUMINAMATH_CALUDE_even_monotone_function_inequality_l1974_197435

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

theorem even_monotone_function_inequality (f : ℝ → ℝ) (m : ℝ)
  (h_even : is_even f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_ineq : f (m + 1) < f (3 * m - 1)) :
  m > 1 ∨ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_even_monotone_function_inequality_l1974_197435


namespace NUMINAMATH_CALUDE_q_polynomial_form_l1974_197455

theorem q_polynomial_form (q : ℝ → ℝ) :
  (∀ x, q x + (2*x^6 + 4*x^4 + 10*x^2) = (5*x^4 + 15*x^3 + 30*x^2 + 10*x + 10)) →
  (∀ x, q x = -2*x^6 + x^4 + 15*x^3 + 20*x^2 + 10*x + 10) := by
sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l1974_197455


namespace NUMINAMATH_CALUDE_distance_between_points_l1974_197405

theorem distance_between_points : Real.sqrt ((0 - 6)^2 + (18 - 0)^2) = 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1974_197405


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_three_l1974_197458

theorem least_positive_integer_to_multiple_of_three :
  ∃ (n : ℕ), n > 0 ∧ (527 + n) % 3 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (527 + m) % 3 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_three_l1974_197458


namespace NUMINAMATH_CALUDE_close_interval_is_zero_one_l1974_197462

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + x + 2
def g (x : ℝ) : ℝ := 2*x + 1

-- Define what it means for two functions to be "close"
def are_close (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- State the theorem
theorem close_interval_is_zero_one :
  are_close f g 0 1 ∧
  ∀ a b, a < 0 ∨ b > 1 → ¬(are_close f g a b) :=
sorry

end NUMINAMATH_CALUDE_close_interval_is_zero_one_l1974_197462


namespace NUMINAMATH_CALUDE_max_coins_identifiable_l1974_197437

/-- The maximum number of coins that can be tested to identify one counterfeit (lighter) coin -/
def max_coins (n : ℕ) : ℕ := 2 * n^2 + 1

/-- A balance scale used for weighing coins -/
structure BalanceScale :=
  (weigh : ℕ → ℕ → Bool)

/-- Represents the process of identifying a counterfeit coin -/
structure CoinIdentification :=
  (n : ℕ)  -- Number of weighings allowed
  (coins : ℕ)  -- Total number of coins
  (scale : BalanceScale)
  (max_weighings_per_coin : ℕ := 2)  -- Maximum number of times each coin can be weighed

/-- Theorem stating the maximum number of coins that can be tested -/
theorem max_coins_identifiable (ci : CoinIdentification) :
  ci.coins ≤ max_coins ci.n ↔
  ∃ (strategy : Unit), true  -- Placeholder for the existence of a valid identification strategy
:= by sorry

end NUMINAMATH_CALUDE_max_coins_identifiable_l1974_197437


namespace NUMINAMATH_CALUDE_high_school_math_team_payment_l1974_197469

theorem high_school_math_team_payment (B : ℕ) : 
  B < 10 → (100 + 10 * B + 3) % 13 = 0 → B = 4 := by
sorry

end NUMINAMATH_CALUDE_high_school_math_team_payment_l1974_197469


namespace NUMINAMATH_CALUDE_three_number_ratio_problem_l1974_197440

theorem three_number_ratio_problem (x y z : ℝ) 
  (h_sum : x + y + z = 120)
  (h_ratio1 : x / y = 3 / 4)
  (h_ratio2 : y / z = 5 / 7)
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0) :
  y = 800 / 21 := by
  sorry

end NUMINAMATH_CALUDE_three_number_ratio_problem_l1974_197440


namespace NUMINAMATH_CALUDE_arithmetic_sequence_51st_term_l1974_197489

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_51st_term 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 2) : 
  a 51 = 101 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_51st_term_l1974_197489


namespace NUMINAMATH_CALUDE_no_valid_license_plate_divisible_by_8_l1974_197483

/-- Represents a 4-digit number of the form aaab -/
structure LicensePlate where
  a : Nat
  b : Nat
  h1 : a < 10
  h2 : b < 10

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem to be proved -/
theorem no_valid_license_plate_divisible_by_8 :
  ¬∃ (plate : LicensePlate),
    (∀ (child_age : Nat), child_age ≥ 1 → child_age ≤ 10 → (1000 * plate.a + 100 * plate.a + 10 * plate.a + plate.b) % child_age = 0) ∧
    isPrime (10 * plate.a + plate.b) ∧
    (1000 * plate.a + 100 * plate.a + 10 * plate.a + plate.b) % 8 = 0 :=
by sorry


end NUMINAMATH_CALUDE_no_valid_license_plate_divisible_by_8_l1974_197483


namespace NUMINAMATH_CALUDE_subset_union_equality_l1974_197429

theorem subset_union_equality (n : ℕ) (A : Fin (n + 1) → Set (Fin n)) 
  (h : ∀ i, (A i).Nonempty) :
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧
  (⋃ (i ∈ I), A i) = (⋃ (j ∈ J), A j) := by
sorry

end NUMINAMATH_CALUDE_subset_union_equality_l1974_197429


namespace NUMINAMATH_CALUDE_probability_of_triangle_formation_l1974_197420

/-- Regular 15-gon with unit circumradius -/
def regular_15gon : Set (ℝ × ℝ) := sorry

/-- Set of all segments in the 15-gon -/
def segments (poly : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

/-- Function to calculate the length of a segment -/
def segment_length (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Predicate to check if three segments form a triangle with positive area -/
def forms_triangle (s1 s2 s3 : Set (ℝ × ℝ)) : Prop := sorry

/-- The total number of ways to choose 3 segments from the 15-gon -/
def total_combinations : ℕ := Nat.choose 105 3

/-- The number of valid triangles formed by three segments -/
def valid_triangles : ℕ := sorry

theorem probability_of_triangle_formation :
  (valid_triangles : ℚ) / total_combinations = 323 / 429 := by sorry

end NUMINAMATH_CALUDE_probability_of_triangle_formation_l1974_197420


namespace NUMINAMATH_CALUDE_circle_equation_with_given_diameter_l1974_197410

/-- The standard equation of a circle with diameter endpoints (0,2) and (4,4) -/
theorem circle_equation_with_given_diameter :
  let p₁ : ℝ × ℝ := (0, 2)
  let p₂ : ℝ × ℝ := (4, 4)
  let M : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (t : ℝ), p = (1 - t) • p₁ + t • p₂ ∧ 0 ≤ t ∧ t ≤ 1}
  ∀ (x y : ℝ), (x, y) ∈ M ↔ (x - 2)^2 + (y - 3)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_given_diameter_l1974_197410


namespace NUMINAMATH_CALUDE_hexagon_segment_probability_l1974_197400

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of elements in T -/
def T_size : ℕ := 15

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := 17/35

theorem hexagon_segment_probability :
  (num_sides / T_size) * ((num_sides - 1) / (T_size - 1)) +
  (num_diagonals / T_size) * ((num_diagonals - 1) / (T_size - 1)) = prob_same_length := by
  sorry

end NUMINAMATH_CALUDE_hexagon_segment_probability_l1974_197400


namespace NUMINAMATH_CALUDE_sculpture_height_proof_l1974_197433

/-- The height of the sculpture in inches -/
def sculpture_height : ℝ := 34

/-- The height of the base in inches -/
def base_height : ℝ := 4

/-- The combined height of the sculpture and base in feet -/
def total_height_feet : ℝ := 3.1666666666666665

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

theorem sculpture_height_proof :
  sculpture_height = total_height_feet * feet_to_inches - base_height :=
by sorry

end NUMINAMATH_CALUDE_sculpture_height_proof_l1974_197433


namespace NUMINAMATH_CALUDE_johns_dancing_time_l1974_197432

theorem johns_dancing_time (john_initial : ℝ) (john_after : ℝ) (james : ℝ) 
  (h1 : john_after = 5)
  (h2 : james = john_initial + 1 + john_after + (1/3) * (john_initial + 1 + john_after))
  (h3 : john_initial + john_after + james = 20) :
  john_initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_johns_dancing_time_l1974_197432


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l1974_197413

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_time : ℕ
  change_time : ℕ
  num_changes : ℕ

/-- Calculates the probability of observing a color change -/
def probability_of_change (cycle : TrafficLightCycle) : ℚ :=
  (cycle.change_time * cycle.num_changes : ℚ) / cycle.total_time

/-- Theorem: The probability of observing a color change in the given traffic light cycle is 2/9 -/
theorem traffic_light_change_probability :
  let cycle : TrafficLightCycle := ⟨90, 5, 4⟩
  probability_of_change cycle = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l1974_197413


namespace NUMINAMATH_CALUDE_regression_line_equation_l1974_197438

/-- Proves that the regression line equation is y = -x + 3 given the conditions -/
theorem regression_line_equation (b : ℝ) :
  (∃ (x y : ℝ), y = b * x + 3 ∧ (1, 2) = (x, y)) →
  (∀ (x y : ℝ), y = -x + 3) :=
by sorry

end NUMINAMATH_CALUDE_regression_line_equation_l1974_197438


namespace NUMINAMATH_CALUDE_g_range_l1974_197479

noncomputable def g (x : ℝ) : ℝ := (Real.arcsin x)^3 - (Real.arccos x)^3

theorem g_range :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 →
  (Real.arccos x + Real.arcsin x = π / 2) →
  ∃ y ∈ Set.Icc (-((7 * π^3) / 16)) ((π^3) / 16), g x = y :=
by sorry

end NUMINAMATH_CALUDE_g_range_l1974_197479


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_zero_l1974_197436

theorem sum_of_reciprocals_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) : 
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_zero_l1974_197436


namespace NUMINAMATH_CALUDE_square_minus_one_factorization_l1974_197446

theorem square_minus_one_factorization (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_square_minus_one_factorization_l1974_197446


namespace NUMINAMATH_CALUDE_inequality_proof_l1974_197493

theorem inequality_proof (x y z : ℝ) 
  (h1 : x^2 + y*z ≠ 0) (h2 : y^2 + z*x ≠ 0) (h3 : z^2 + x*y ≠ 0) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1974_197493


namespace NUMINAMATH_CALUDE_songs_to_learn_l1974_197448

/-- Given that Billy can play 24 songs and his music book contains 52 songs,
    prove that the number of songs he still needs to learn is 28. -/
theorem songs_to_learn (songs_can_play : ℕ) (total_songs : ℕ) 
  (h1 : songs_can_play = 24) (h2 : total_songs = 52) : 
  total_songs - songs_can_play = 28 := by
  sorry

end NUMINAMATH_CALUDE_songs_to_learn_l1974_197448


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l1974_197465

def probability_exactly_3_heads (n : ℕ) (p : ℚ) : ℚ :=
  ↑(Nat.choose n 3) * p^3 * (1 - p)^(n - 3)

def probability_all_heads (n : ℕ) (p : ℚ) : ℚ :=
  p^n

theorem fair_coin_probability_difference :
  let n : ℕ := 4
  let p : ℚ := 1/2
  (probability_exactly_3_heads n p) - (probability_all_heads n p) = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l1974_197465


namespace NUMINAMATH_CALUDE_roy_school_days_l1974_197422

/-- Represents the number of hours Roy spends on sports activities in school each day -/
def daily_sports_hours : ℕ := 2

/-- Represents the number of days Roy missed within a week -/
def missed_days : ℕ := 2

/-- Represents the total hours Roy spent on sports in school for the week -/
def weekly_sports_hours : ℕ := 6

/-- Represents the number of days Roy goes to school in a week -/
def school_days : ℕ := 5

theorem roy_school_days : 
  daily_sports_hours * (school_days - missed_days) = weekly_sports_hours ∧ 
  school_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_roy_school_days_l1974_197422


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l1974_197475

/-- The number of points on the circle -/
def num_points : ℕ := 8

/-- The number of chords to be selected -/
def num_selected_chords : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := num_points.choose 2

/-- The number of ways to select the required number of chords -/
def ways_to_select_chords : ℕ := total_chords.choose num_selected_chords

/-- The number of ways to choose points that form a convex quadrilateral -/
def convex_quadrilaterals : ℕ := num_points.choose 4

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := convex_quadrilaterals / ways_to_select_chords

theorem convex_quadrilateral_probability : probability = 2 / 585 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l1974_197475


namespace NUMINAMATH_CALUDE_problem_statement_l1974_197443

theorem problem_statement :
  (∀ n : ℕ, n > 1 → ¬(n ∣ (2^n - 1))) ∧
  (∀ n : ℕ, Nat.Prime n → (n^2 ∣ (2^n + 1)) → n = 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1974_197443


namespace NUMINAMATH_CALUDE_right_triangle_congruence_l1974_197444

-- Define a right-angled triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define congruence for right-angled triangles
def congruent (t1 t2 : RightTriangle) : Prop :=
  t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2 ∧ t1.hypotenuse = t2.hypotenuse

-- Theorem: Two right-angled triangles with two equal legs are congruent
theorem right_triangle_congruence (t1 t2 : RightTriangle) 
  (h : t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2) : congruent t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_congruence_l1974_197444


namespace NUMINAMATH_CALUDE_four_purchase_options_l1974_197487

/-- Represents the number of different ways to buy masks and alcohol wipes -/
def purchase_options : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 2 * p.2 = 30) (Finset.product (Finset.range 31) (Finset.range 31))).card

/-- Theorem stating that there are exactly 4 ways to purchase masks and alcohol wipes -/
theorem four_purchase_options : purchase_options = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_purchase_options_l1974_197487


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1974_197495

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := 4 * a + 2 * b

-- State the theorem
theorem diamond_equation_solution :
  ∃ x : ℚ, diamond 3 (diamond x 7) = 5 ∧ x = -35/8 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l1974_197495


namespace NUMINAMATH_CALUDE_expression_simplification_l1974_197447

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((x + 3) / x - 1) / ((x^2 - 1) / (x^2 + x)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1974_197447


namespace NUMINAMATH_CALUDE_max_teams_is_eight_l1974_197426

/-- Represents the number of teams that can be formed given the number of climbers in each skill level and the required composition of each team. -/
def max_teams (advanced intermediate beginner : ℕ) 
              (adv_per_team int_per_team beg_per_team : ℕ) : ℕ :=
  min (advanced / adv_per_team)
      (min (intermediate / int_per_team)
           (beginner / beg_per_team))

/-- Theorem stating that the maximum number of teams that can be formed is 8. -/
theorem max_teams_is_eight : 
  max_teams 45 70 57 5 8 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_teams_is_eight_l1974_197426


namespace NUMINAMATH_CALUDE_area_parallelogram_from_diagonals_l1974_197453

/-- A quadrilateral in a plane -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- The diagonals of a quadrilateral -/
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- A parallelogram with sides parallel and equal to given line segments -/
def parallelogram_from_diagonals (d : (ℝ × ℝ) × (ℝ × ℝ)) : Quadrilateral := sorry

/-- The theorem stating that the area of the parallelogram formed by the diagonals
    is twice the area of the original quadrilateral -/
theorem area_parallelogram_from_diagonals (q : Quadrilateral) :
  area (parallelogram_from_diagonals (diagonals q)) = 2 * area q := by sorry

end NUMINAMATH_CALUDE_area_parallelogram_from_diagonals_l1974_197453


namespace NUMINAMATH_CALUDE_tucker_bought_three_boxes_l1974_197485

def tissues_per_box : ℕ := 160
def used_tissues : ℕ := 210
def remaining_tissues : ℕ := 270

theorem tucker_bought_three_boxes :
  (used_tissues + remaining_tissues) / tissues_per_box = 3 :=
by sorry

end NUMINAMATH_CALUDE_tucker_bought_three_boxes_l1974_197485


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1974_197459

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines_a_value (a : ℝ) :
  let l1 : ℝ → ℝ → Prop := λ x y => a * x + (3 - a) * y + 1 = 0
  let l2 : ℝ → ℝ → Prop := λ x y => x - 2 * y = 0
  let m1 : ℝ := a / (a - 3)  -- slope of l1
  let m2 : ℝ := 1 / 2        -- slope of l2
  perpendicular m1 m2 → a = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1974_197459


namespace NUMINAMATH_CALUDE_unique_natural_number_a_l1974_197428

theorem unique_natural_number_a : ∃! (a : ℕ), 
  (1000 ≤ 4 * a^2) ∧ (4 * a^2 < 10000) ∧ 
  (1000 ≤ (4/3) * a^3) ∧ ((4/3) * a^3 < 10000) ∧
  (∃ (n : ℕ), (4/3) * a^3 = n) :=
by sorry

end NUMINAMATH_CALUDE_unique_natural_number_a_l1974_197428


namespace NUMINAMATH_CALUDE_rulers_in_drawer_l1974_197460

/-- The total number of rulers in a drawer after an addition. -/
def total_rulers (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: Given 11 initial rulers and 14 added rulers, the total is 25. -/
theorem rulers_in_drawer : total_rulers 11 14 = 25 := by
  sorry

end NUMINAMATH_CALUDE_rulers_in_drawer_l1974_197460


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l1974_197403

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem pirate_treasure_distribution : 
  ∃ (x : ℕ), 
    x > 0 ∧ 
    sum_of_first_n x = 3 * x ∧ 
    x + 3 * x = 20 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l1974_197403


namespace NUMINAMATH_CALUDE_sum_of_preceding_terms_l1974_197415

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define our specific sequence
def our_sequence (a : ℕ → ℕ) : Prop :=
  arithmetic_sequence a ∧ a 1 = 3 ∧ a 2 = 8 ∧ ∃ k : ℕ, a k = 33 ∧ ∀ m : ℕ, m > k → a m > 33

theorem sum_of_preceding_terms (a : ℕ → ℕ) (h : our_sequence a) :
  ∃ n : ℕ, a n + a (n + 1) = 51 ∧ a (n + 2) = 33 :=
sorry

end NUMINAMATH_CALUDE_sum_of_preceding_terms_l1974_197415


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l1974_197457

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -75 + 40*I ∧ (5 + 7*I)^2 = -75 + 40*I →
  (-5 - 7*I)^2 = -75 + 40*I :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l1974_197457


namespace NUMINAMATH_CALUDE_complex_sixth_power_sum_l1974_197451

theorem complex_sixth_power_sum : 
  (((-1 + Complex.I * Real.sqrt 3) / 2) ^ 6 + 
   ((-1 - Complex.I * Real.sqrt 3) / 2) ^ 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sixth_power_sum_l1974_197451


namespace NUMINAMATH_CALUDE_system_solutions_correct_l1974_197423

theorem system_solutions_correct :
  -- System (1)
  let x₁ := 1
  let y₁ := 2
  -- System (2)
  let x₂ := (1 : ℚ) / 2
  let y₂ := 5
  -- Prove that these solutions satisfy the equations
  (x₁ = 5 - 2 * y₁ ∧ 3 * x₁ - y₁ = 1) ∧
  (2 * x₂ - y₂ = -4 ∧ 4 * x₂ - 5 * y₂ = -23) := by
  sorry


end NUMINAMATH_CALUDE_system_solutions_correct_l1974_197423


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1974_197434

theorem sqrt_equation_solution :
  ∃! x : ℚ, Real.sqrt (5 - 4 * x) = 6 :=
by
  use -31/4
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1974_197434


namespace NUMINAMATH_CALUDE_company_workforce_company_workforce_proof_l1974_197445

theorem company_workforce (initial_female_percentage : ℚ) 
                          (additional_male_workers : ℕ) 
                          (final_female_percentage : ℚ) : Prop :=
  initial_female_percentage = 60 / 100 →
  additional_male_workers = 22 →
  final_female_percentage = 55 / 100 →
  ∃ (initial_employees final_employees : ℕ),
    (initial_employees : ℚ) * initial_female_percentage = 
      (final_employees : ℚ) * final_female_percentage ∧
    final_employees = initial_employees + additional_male_workers ∧
    final_employees = 264

-- The proof of the theorem
theorem company_workforce_proof : 
  company_workforce (60 / 100) 22 (55 / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_company_workforce_company_workforce_proof_l1974_197445


namespace NUMINAMATH_CALUDE_daniels_initial_noodles_l1974_197452

/-- Represents the number of noodles Daniel had initially -/
def initial_noodles : ℕ := sorry

/-- Represents the number of noodles Daniel gave away -/
def noodles_given_away : ℕ := 12

/-- Represents the number of noodles Daniel has now -/
def remaining_noodles : ℕ := 54

/-- Theorem stating that Daniel's initial number of noodles was 66 -/
theorem daniels_initial_noodles : 
  initial_noodles = noodles_given_away + remaining_noodles := by
  sorry

end NUMINAMATH_CALUDE_daniels_initial_noodles_l1974_197452


namespace NUMINAMATH_CALUDE_line_intersects_at_least_one_l1974_197404

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (contained_in : Line → Plane → Prop)
variable (intersects : Line → Line → Prop)
variable (skew : Line → Line → Prop)
variable (plane_intersection : Plane → Plane → Line → Prop)

-- State the theorem
theorem line_intersects_at_least_one 
  (a b l : Line) (α β : Plane) :
  skew a b →
  contained_in a α →
  contained_in b β →
  plane_intersection α β l →
  (intersects l a) ∨ (intersects l b) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_at_least_one_l1974_197404


namespace NUMINAMATH_CALUDE_sum_of_2010_3_array_remainder_of_sum_l1974_197490

/-- Definition of the sum of a pq-array --/
def pq_array_sum (p q : ℕ) : ℚ :=
  (1 / (1 - 1 / (2 * p))) * (1 / (1 - 1 / q))

/-- Theorem stating the sum of the specific 1/2010,3-array --/
theorem sum_of_2010_3_array :
  pq_array_sum 2010 3 = 6030 / 4019 := by
  sorry

/-- Theorem for the remainder when numerator + denominator is divided by 2010 --/
theorem remainder_of_sum :
  (6030 + 4019) % 2010 = 1009 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_2010_3_array_remainder_of_sum_l1974_197490


namespace NUMINAMATH_CALUDE_smallest_superior_discount_l1974_197482

def successive_discount (single_discount : ℝ) (times : ℕ) : ℝ :=
  1 - (1 - single_discount) ^ times

theorem smallest_superior_discount : ∃ (n : ℕ), n = 37 ∧
  (∀ (m : ℕ), m < n →
    (m : ℝ) / 100 ≤ successive_discount (12 / 100) 3 ∨
    (m : ℝ) / 100 ≤ successive_discount (20 / 100) 2 ∨
    (m : ℝ) / 100 ≤ successive_discount (8 / 100) 4) ∧
  (37 : ℝ) / 100 > successive_discount (12 / 100) 3 ∧
  (37 : ℝ) / 100 > successive_discount (20 / 100) 2 ∧
  (37 : ℝ) / 100 > successive_discount (8 / 100) 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_superior_discount_l1974_197482


namespace NUMINAMATH_CALUDE_two_three_four_forms_triangle_one_two_three_not_triangle_two_two_four_not_triangle_two_three_six_not_triangle_triangle_formation_theorem_l1974_197409

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem stating that (2, 3, 4) can form a triangle
theorem two_three_four_forms_triangle :
  can_form_triangle 2 3 4 := by sorry

-- Theorem stating that (1, 2, 3) cannot form a triangle
theorem one_two_three_not_triangle :
  ¬ can_form_triangle 1 2 3 := by sorry

-- Theorem stating that (2, 2, 4) cannot form a triangle
theorem two_two_four_not_triangle :
  ¬ can_form_triangle 2 2 4 := by sorry

-- Theorem stating that (2, 3, 6) cannot form a triangle
theorem two_three_six_not_triangle :
  ¬ can_form_triangle 2 3 6 := by sorry

-- Main theorem combining all results
theorem triangle_formation_theorem :
  can_form_triangle 2 3 4 ∧
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 2 2 4 ∧
  ¬ can_form_triangle 2 3 6 := by sorry

end NUMINAMATH_CALUDE_two_three_four_forms_triangle_one_two_three_not_triangle_two_two_four_not_triangle_two_three_six_not_triangle_triangle_formation_theorem_l1974_197409


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l1974_197424

/-- Given a hyperbola with equation x²/36 - y²/b² = 1 where b > 0,
    eccentricity e = 5/3, and a point P on the hyperbola such that |PF₁| = 15,
    prove that |PF₂| = 27 -/
theorem hyperbola_focal_distance (b : ℝ) (P : ℝ × ℝ) :
  b > 0 →
  (P.1^2 / 36 - P.2^2 / b^2 = 1) →
  (Real.sqrt (36 + b^2) / 6 = 5 / 3) →
  (Real.sqrt ((P.1 + Real.sqrt (36 + b^2))^2 + P.2^2) = 15) →
  Real.sqrt ((P.1 - Real.sqrt (36 + b^2))^2 + P.2^2) = 27 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l1974_197424


namespace NUMINAMATH_CALUDE_equilibrium_exists_l1974_197431

/-- Represents the equilibrium state of two connected vessels with different liquids -/
def EquilibriumState (H : ℝ) : Prop :=
  ∃ (h_water h_gasoline : ℝ),
    -- Initial conditions
    0 < H ∧
    -- Valve position
    0.15 * H < 0.9 * H ∧
    -- Initial liquid levels
    0.9 * H = 0.9 * H ∧
    -- Densities
    let ρ_water : ℝ := 1000
    let ρ_gasoline : ℝ := 600
    -- Equilibrium condition
    ρ_water * (0.75 * H - (0.9 * H - h_water)) = 
      ρ_water * (h_water - 0.15 * H) + ρ_gasoline * (H - h_water) ∧
    -- Final water level
    h_water = 0.69 * H ∧
    -- Final gasoline level
    h_gasoline = H

/-- Theorem stating that the equilibrium state exists for any positive vessel height -/
theorem equilibrium_exists (H : ℝ) (h_pos : 0 < H) : EquilibriumState H := by
  sorry

#check equilibrium_exists

end NUMINAMATH_CALUDE_equilibrium_exists_l1974_197431


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1974_197498

/-- The eccentricity of an ellipse with equation x²/m² + y²/9 = 1 (m > 0) and one focus at (4, 0) is 4/5 -/
theorem ellipse_eccentricity (m : ℝ) (h1 : m > 0) : 
  let ellipse := { (x, y) : ℝ × ℝ | x^2 / m^2 + y^2 / 9 = 1 }
  let focus : ℝ × ℝ := (4, 0)
  focus ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 ∧ p ∈ ellipse } →
  (∃ (a b c : ℝ), a > b ∧ b > 0 ∧ c > 0 ∧ a^2 = m^2 ∧ b^2 = 9 ∧ c^2 = a^2 - b^2 ∧ c / a = 4 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1974_197498


namespace NUMINAMATH_CALUDE_intersection_contains_two_elements_l1974_197414

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = 3 * p.1^2}
def N : Set (ℝ × ℝ) := {p | p.2 = 5 * p.1}

-- State the theorem
theorem intersection_contains_two_elements :
  ∃ (a b : ℝ × ℝ), a ≠ b ∧ a ∈ M ∩ N ∧ b ∈ M ∩ N ∧
  ∀ c, c ∈ M ∩ N → c = a ∨ c = b :=
sorry

end NUMINAMATH_CALUDE_intersection_contains_two_elements_l1974_197414


namespace NUMINAMATH_CALUDE_paving_cost_l1974_197427

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 400) :
  length * width * rate = 8250 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l1974_197427


namespace NUMINAMATH_CALUDE_certain_number_problem_l1974_197478

theorem certain_number_problem : ∃ x : ℚ, (5/6 : ℚ) * x = (5/16 : ℚ) * x + 100 ∧ x = 192 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1974_197478


namespace NUMINAMATH_CALUDE_nested_fraction_equals_seven_halves_l1974_197416

theorem nested_fraction_equals_seven_halves :
  2 + 2 / (1 + 1 / (2 + 1)) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equals_seven_halves_l1974_197416


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1974_197439

theorem quadratic_roots_relation (a b c : ℚ) : 
  (∃ r s : ℚ, (5 * r^2 + 2 * r - 4 = 0) ∧ (5 * s^2 + 2 * s - 4 = 0) ∧ 
   (a * (r - 3)^2 + b * (r - 3) + c = 0) ∧ (a * (s - 3)^2 + b * (s - 3) + c = 0)) →
  c = 47/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1974_197439


namespace NUMINAMATH_CALUDE_seat_distribution_correct_l1974_197471

/-- Represents the total number of seats on the airplane -/
def total_seats : ℕ := 90

/-- Represents the number of seats in First Class -/
def first_class_seats : ℕ := 36

/-- Represents the proportion of seats in Business Class -/
def business_class_proportion : ℚ := 1/5

/-- Represents the proportion of seats in Premium Economy -/
def premium_economy_proportion : ℚ := 2/5

/-- Theorem stating that the given seat distribution is correct -/
theorem seat_distribution_correct : 
  first_class_seats + 
  (business_class_proportion * total_seats).floor + 
  (premium_economy_proportion * total_seats).floor + 
  (total_seats - first_class_seats - 
   (business_class_proportion * total_seats).floor - 
   (premium_economy_proportion * total_seats).floor) = total_seats :=
by sorry

end NUMINAMATH_CALUDE_seat_distribution_correct_l1974_197471


namespace NUMINAMATH_CALUDE_penny_percentage_theorem_l1974_197408

theorem penny_percentage_theorem (initial_pennies : ℕ) 
                                 (old_pennies : ℕ) 
                                 (final_pennies : ℕ) : 
  initial_pennies = 200 →
  old_pennies = 30 →
  final_pennies = 136 →
  (initial_pennies - old_pennies) * (1 - 20 / 100) = final_pennies :=
by
  sorry

end NUMINAMATH_CALUDE_penny_percentage_theorem_l1974_197408


namespace NUMINAMATH_CALUDE_picnic_gender_difference_l1974_197449

/-- Given a group of people at a picnic, prove the difference between men and women -/
theorem picnic_gender_difference 
  (total : ℕ) 
  (men : ℕ) 
  (women : ℕ) 
  (children : ℕ) 
  (h1 : total = 200)
  (h2 : men + women = children + 20)
  (h3 : men = 65)
  (h4 : total = men + women + children) :
  men - women = 20 := by
sorry

end NUMINAMATH_CALUDE_picnic_gender_difference_l1974_197449


namespace NUMINAMATH_CALUDE_house_selling_price_l1974_197464

theorem house_selling_price (commission_rate : ℝ) (commission : ℝ) (selling_price : ℝ) :
  commission_rate = 0.06 →
  commission = 8880 →
  commission = commission_rate * selling_price →
  selling_price = 148000 := by
  sorry

end NUMINAMATH_CALUDE_house_selling_price_l1974_197464


namespace NUMINAMATH_CALUDE_incenter_coords_l1974_197421

/-- Triangle DEF with side lengths d, e, f -/
structure Triangle where
  d : ℝ
  e : ℝ
  f : ℝ

/-- Barycentric coordinates -/
structure BarycentricCoords where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : BarycentricCoords :=
  sorry

/-- The theorem stating that the barycentric coordinates of the incenter
    of triangle DEF with side lengths 8, 15, 17 are (8/40, 15/40, 17/40) -/
theorem incenter_coords :
  let t : Triangle := { d := 8, e := 15, f := 17 }
  let i : BarycentricCoords := incenter t
  i.x = 8/40 ∧ i.y = 15/40 ∧ i.z = 17/40 ∧ i.x + i.y + i.z = 1 := by
  sorry

end NUMINAMATH_CALUDE_incenter_coords_l1974_197421


namespace NUMINAMATH_CALUDE_billys_remaining_crayons_l1974_197497

/-- Given Billy's initial number of crayons and the number eaten by a hippopotamus,
    this theorem proves that the remaining number of crayons is the difference between
    the initial number and the number eaten. -/
theorem billys_remaining_crayons (initial : ℕ) (eaten : ℕ) :
  initial ≥ eaten → initial - eaten = initial - eaten :=
by
  sorry

end NUMINAMATH_CALUDE_billys_remaining_crayons_l1974_197497


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1974_197481

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The product of the first n terms of a sequence -/
def SequenceProduct (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc i => acc * b (i + 1)) 1

theorem geometric_sequence_property (b : ℕ → ℝ) (h : GeometricSequence b) (h7 : b 7 = 1) :
  ∀ n : ℕ+, n < 13 → SequenceProduct b n = SequenceProduct b (13 - n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1974_197481


namespace NUMINAMATH_CALUDE_bill_vote_difference_l1974_197494

theorem bill_vote_difference (total : ℕ) (initial_for initial_against revote_for revote_against : ℕ) :
  total = 400 →
  initial_for + initial_against = total →
  revote_for + revote_against = total →
  (revote_for : ℚ) - revote_against = 3 * (initial_against - initial_for) →
  (revote_for : ℚ) = 13 / 12 * initial_against →
  revote_for - initial_for = 36 :=
by sorry

end NUMINAMATH_CALUDE_bill_vote_difference_l1974_197494


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l1974_197407

theorem line_ellipse_intersection (k : ℝ) : ∃ (x y : ℝ), 
  (y = k * x + 1 - k) ∧ (x^2 / 9 + y^2 / 4 = 1) := by
  sorry

#check line_ellipse_intersection

end NUMINAMATH_CALUDE_line_ellipse_intersection_l1974_197407


namespace NUMINAMATH_CALUDE_ophelia_pay_reaches_93_l1974_197496

/-- Ophelia's weekly earnings function -/
def earnings (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 51
  else 51 + 100 * (n - 1)

/-- Average weekly pay after n weeks -/
def average_pay (n : ℕ) : ℚ :=
  if n = 0 then 0 else (earnings n) / n

/-- Theorem: Ophelia's average weekly pay reaches $93 after 7 weeks -/
theorem ophelia_pay_reaches_93 :
  ∃ n : ℕ, n > 0 ∧ average_pay n = 93 ∧ ∀ m : ℕ, m > 0 ∧ m < n → average_pay m < 93 :=
sorry

end NUMINAMATH_CALUDE_ophelia_pay_reaches_93_l1974_197496


namespace NUMINAMATH_CALUDE_magnitude_a_minus_b_l1974_197425

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (3, -2)

theorem magnitude_a_minus_b : 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_minus_b_l1974_197425


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l1974_197486

theorem red_shirt_pairs (green_students : ℕ) (red_students : ℕ) (total_students : ℕ) 
  (total_pairs : ℕ) (green_green_pairs : ℕ) :
  green_students = 63 →
  red_students = 69 →
  total_students = 132 →
  total_pairs = 66 →
  green_green_pairs = 27 →
  ∃ red_red_pairs : ℕ, red_red_pairs = 30 ∧ 
    red_red_pairs = total_pairs - green_green_pairs - (green_students - 2 * green_green_pairs) :=
by sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_l1974_197486


namespace NUMINAMATH_CALUDE_triathlon_bike_speed_l1974_197468

/-- Triathlon problem -/
theorem triathlon_bike_speed 
  (total_time : ℝ) 
  (swim_speed swim_distance : ℝ) 
  (run_speed run_distance : ℝ) 
  (bike_distance : ℝ) :
  total_time = 2.5 →
  swim_speed = 2 →
  swim_distance = 0.25 →
  run_speed = 5 →
  run_distance = 3 →
  bike_distance = 20 →
  ∃ bike_speed : ℝ, 
    (swim_distance / swim_speed + run_distance / run_speed + bike_distance / bike_speed = total_time) ∧
    (abs (bike_speed - 11.27) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_triathlon_bike_speed_l1974_197468


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l1974_197488

theorem weekend_rain_probability (p_saturday p_sunday : ℝ) 
  (h1 : p_saturday = 0.6)
  (h2 : p_sunday = 0.7)
  (h3 : 0 ≤ p_saturday ∧ p_saturday ≤ 1)
  (h4 : 0 ≤ p_sunday ∧ p_sunday ≤ 1) :
  1 - (1 - p_saturday) * (1 - p_sunday) = 0.88 := by
sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l1974_197488


namespace NUMINAMATH_CALUDE_dividingChordLength_l1974_197456

/-- A hexagon inscribed in a circle with alternating side lengths -/
structure AlternatingHexagon where
  /-- The length of three consecutive sides -/
  shortSide : ℝ
  /-- The length of the other three consecutive sides -/
  longSide : ℝ
  /-- The short sides are indeed shorter than the long sides -/
  shortLessThanLong : shortSide < longSide

/-- The chord dividing the hexagon into two trapezoids -/
def dividingChord (h : AlternatingHexagon) : ℝ := sorry

theorem dividingChordLength (h : AlternatingHexagon) 
  (h_short : h.shortSide = 4)
  (h_long : h.longSide = 6) :
  dividingChord h = 480 / 49 := by
  sorry

end NUMINAMATH_CALUDE_dividingChordLength_l1974_197456


namespace NUMINAMATH_CALUDE_complex_square_one_minus_i_l1974_197411

theorem complex_square_one_minus_i :
  (1 - Complex.I) ^ 2 = -2 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_square_one_minus_i_l1974_197411


namespace NUMINAMATH_CALUDE_quadratic_equation_consequence_l1974_197417

theorem quadratic_equation_consequence (m : ℝ) (h : m^2 + 2*m - 1 = 0) :
  2*m^2 + 4*m - 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_consequence_l1974_197417


namespace NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l1974_197402

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometricSequenceTerm (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem fifteenth_term_of_geometric_sequence :
  let a := 5
  let r := (1/2 : ℚ)
  let n := 15
  geometricSequenceTerm a r n = 5/16384 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l1974_197402


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1974_197480

/-- The minimum value of k*sin^4(x) + cos^4(x) for k ≥ 0 is 0 -/
theorem min_value_trig_expression (k : ℝ) (hk : k ≥ 0) :
  ∃ m : ℝ, m = 0 ∧ ∀ x : ℝ, k * Real.sin x ^ 4 + Real.cos x ^ 4 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1974_197480


namespace NUMINAMATH_CALUDE_no_sum_of_cubes_l1974_197470

theorem no_sum_of_cubes (n : ℕ) : ¬∃ (x y : ℕ), 10^(3*n + 1) = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_sum_of_cubes_l1974_197470


namespace NUMINAMATH_CALUDE_omega_identity_l1974_197412

theorem omega_identity (ω : ℂ) (h : ω = -1/2 + Complex.I * (Real.sqrt 3) / 2) :
  1 + ω = -1/ω := by sorry

end NUMINAMATH_CALUDE_omega_identity_l1974_197412


namespace NUMINAMATH_CALUDE_repeating_digit_equation_solutions_l1974_197450

def repeating_digit (d : ℕ) (n : ℕ) : ℕ :=
  d * ((10^n - 1) / 9)

theorem repeating_digit_equation_solutions :
  ∀ a b c : ℕ,
    (∀ n : ℕ, repeating_digit a n * 10^n + repeating_digit b n + 1 = (repeating_digit c n + 1)^2) →
    ((a = 0 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 5 ∧ c = 3) ∨
     (a = 4 ∧ b = 8 ∧ c = 6) ∨
     (a = 9 ∧ b = 9 ∧ c = 9)) :=
by sorry

end NUMINAMATH_CALUDE_repeating_digit_equation_solutions_l1974_197450


namespace NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_3_7_13_l1974_197474

theorem smallest_six_digit_divisible_by_3_7_13 : ∀ n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧ n % 3 = 0 ∧ n % 7 = 0 ∧ n % 13 = 0 →
  100191 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_3_7_13_l1974_197474


namespace NUMINAMATH_CALUDE_inequality_addition_l1974_197430

theorem inequality_addition (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l1974_197430


namespace NUMINAMATH_CALUDE_tan_beta_value_l1974_197499

theorem tan_beta_value (θ β : Real) (h1 : (2 : Real) = 2 * Real.cos θ) 
  (h2 : (-3 : Real) = 2 * Real.sin θ) (h3 : β = θ - 3 * Real.pi / 4) : 
  Real.tan β = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l1974_197499


namespace NUMINAMATH_CALUDE_normas_cards_l1974_197406

/-- Proves that Norma's total number of cards is 158.0 given the initial and found amounts -/
theorem normas_cards (initial_cards : Real) (found_cards : Real) 
  (h1 : initial_cards = 88.0) 
  (h2 : found_cards = 70.0) : 
  initial_cards + found_cards = 158.0 := by
sorry

end NUMINAMATH_CALUDE_normas_cards_l1974_197406


namespace NUMINAMATH_CALUDE_zeros_of_f_l1974_197463

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem stating that 3 and -1 are the zeros of the function f
theorem zeros_of_f : 
  (f 3 = 0 ∧ f (-1) = 0) ∧ 
  ∀ x : ℝ, f x = 0 → x = 3 ∨ x = -1 :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l1974_197463


namespace NUMINAMATH_CALUDE_school_pupils_count_school_pupils_count_proof_l1974_197418

theorem school_pupils_count : ℕ → ℕ → ℕ → Prop :=
  fun num_girls girls_boys_diff total_pupils =>
    (num_girls = 868) →
    (girls_boys_diff = 281) →
    (total_pupils = num_girls + (num_girls - girls_boys_diff)) →
    total_pupils = 1455

-- The proof is omitted
theorem school_pupils_count_proof : school_pupils_count 868 281 1455 := by
  sorry

end NUMINAMATH_CALUDE_school_pupils_count_school_pupils_count_proof_l1974_197418


namespace NUMINAMATH_CALUDE_three_round_layoffs_result_l1974_197454

def layoff_round (employees : ℕ) : ℕ :=
  (employees * 10) / 100

def remaining_after_layoff (employees : ℕ) : ℕ :=
  employees - layoff_round employees

def total_layoffs (initial_employees : ℕ) : ℕ :=
  let first_round := layoff_round initial_employees
  let after_first := remaining_after_layoff initial_employees
  let second_round := layoff_round after_first
  let after_second := remaining_after_layoff after_first
  let third_round := layoff_round after_second
  first_round + second_round + third_round

theorem three_round_layoffs_result :
  total_layoffs 1000 = 271 :=
sorry

end NUMINAMATH_CALUDE_three_round_layoffs_result_l1974_197454


namespace NUMINAMATH_CALUDE_bike_sharing_selection_l1974_197401

theorem bike_sharing_selection (yellow_bikes : ℕ) (blue_bikes : ℕ) (inspect_yellow : ℕ) (inspect_blue : ℕ) :
  yellow_bikes = 6 →
  blue_bikes = 4 →
  inspect_yellow = 4 →
  inspect_blue = 4 →
  (Nat.choose blue_bikes 2 * Nat.choose yellow_bikes 2 +
   Nat.choose blue_bikes 3 * Nat.choose yellow_bikes 1 +
   Nat.choose blue_bikes 4) = 115 :=
by sorry

end NUMINAMATH_CALUDE_bike_sharing_selection_l1974_197401


namespace NUMINAMATH_CALUDE_cistern_leak_time_l1974_197467

theorem cistern_leak_time 
  (fill_time_A : ℝ) 
  (fill_time_both : ℝ) 
  (leak_time_B : ℝ) : 
  fill_time_A = 10 →
  fill_time_both = 29.999999999999993 →
  (1 / fill_time_A - 1 / leak_time_B = 1 / fill_time_both) →
  leak_time_B = 15 := by
sorry

end NUMINAMATH_CALUDE_cistern_leak_time_l1974_197467


namespace NUMINAMATH_CALUDE_power_product_simplification_l1974_197492

theorem power_product_simplification (a : ℝ) : ((-2 * a)^2) * (a^4) = 4 * (a^6) := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l1974_197492


namespace NUMINAMATH_CALUDE_unique_apartment_number_l1974_197473

def is_valid_apartment_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (100 * a + 10 * c + b) +
    (100 * b + 10 * a + c) +
    (100 * b + 10 * c + a) +
    (100 * c + 10 * a + b) +
    (100 * c + 10 * b + a) = 2017

theorem unique_apartment_number :
  ∃! n : ℕ, is_valid_apartment_number n ∧ n = 425 :=
sorry

end NUMINAMATH_CALUDE_unique_apartment_number_l1974_197473


namespace NUMINAMATH_CALUDE_desk_final_price_l1974_197472

/-- Calculates the final price of an auctioned item given initial price, price increase per bid, and number of bids -/
def final_price (initial_price : ℕ) (price_increase : ℕ) (num_bids : ℕ) : ℕ :=
  initial_price + price_increase * num_bids

/-- Theorem stating the final price of the desk after the bidding war -/
theorem desk_final_price :
  final_price 15 5 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_desk_final_price_l1974_197472


namespace NUMINAMATH_CALUDE_smallest_cube_root_integer_l1974_197419

theorem smallest_cube_root_integer (m n : ℕ) (r : ℝ) : 
  (∃ m : ℕ, ∃ r : ℝ, 
    m > 0 ∧ 
    r > 0 ∧ 
    r < 1/1000 ∧ 
    (m : ℝ)^(1/3 : ℝ) = n + r) → 
  n ≥ 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_integer_l1974_197419


namespace NUMINAMATH_CALUDE_peg_arrangement_count_l1974_197441

/-- The number of ways to distribute colored pegs on a square board. -/
def peg_arrangements : ℕ :=
  let red_pegs := 6
  let green_pegs := 5
  let blue_pegs := 4
  let yellow_pegs := 3
  let orange_pegs := 2
  let board_size := 6
  Nat.factorial red_pegs * Nat.factorial green_pegs * Nat.factorial blue_pegs *
  Nat.factorial yellow_pegs * Nat.factorial orange_pegs

/-- Theorem stating the number of valid peg arrangements. -/
theorem peg_arrangement_count :
  peg_arrangements = 12441600 := by
  sorry

end NUMINAMATH_CALUDE_peg_arrangement_count_l1974_197441


namespace NUMINAMATH_CALUDE_distance_between_points_l1974_197476

def point1 : ℝ × ℝ := (3, 7)
def point2 : ℝ × ℝ := (-5, 2)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 89 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1974_197476


namespace NUMINAMATH_CALUDE_remaining_integers_count_l1974_197461

def S : Finset Nat := Finset.range 51 \ {0}

theorem remaining_integers_count : 
  (S.filter (fun n => n % 2 ≠ 0 ∧ n % 3 ≠ 0)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_remaining_integers_count_l1974_197461
