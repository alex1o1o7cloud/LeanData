import Mathlib

namespace NUMINAMATH_CALUDE_freyja_age_l3528_352803

/-- Represents the ages of the people in the problem -/
structure Ages where
  kaylin : ℕ
  sarah : ℕ
  eli : ℕ
  freyja : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.kaylin = ages.sarah - 5 ∧
  ages.sarah = 2 * ages.eli ∧
  ages.eli = ages.freyja + 9 ∧
  ages.kaylin = 33

/-- The theorem stating Freyja's age given the problem conditions -/
theorem freyja_age (ages : Ages) (h : problem_conditions ages) : ages.freyja = 10 := by
  sorry


end NUMINAMATH_CALUDE_freyja_age_l3528_352803


namespace NUMINAMATH_CALUDE_cousin_distribution_count_l3528_352862

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribution_count (n k : ℕ) : ℕ :=
  sorry

/-- The number of cousins -/
def num_cousins : ℕ := 5

/-- The number of rooms -/
def num_rooms : ℕ := 5

/-- Theorem stating that the number of ways to distribute 5 cousins into 5 rooms is 37 -/
theorem cousin_distribution_count :
  distribution_count num_cousins num_rooms = 37 := by sorry

end NUMINAMATH_CALUDE_cousin_distribution_count_l3528_352862


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3528_352840

theorem quadratic_equation_solutions (x : ℝ) :
  x^2 = 8*x - 15 →
  (∃ s p : ℝ, s = 8 ∧ p = 15 ∧
    (∀ x₁ x₂ : ℝ, x₁^2 = 8*x₁ - 15 ∧ x₂^2 = 8*x₂ - 15 → x₁ + x₂ = s ∧ x₁ * x₂ = p)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3528_352840


namespace NUMINAMATH_CALUDE_lindas_bakery_profit_l3528_352847

/-- Calculate Linda's total profit for the day given her bread sales strategy -/
theorem lindas_bakery_profit :
  let total_loaves : ℕ := 60
  let morning_price : ℚ := 3
  let afternoon_price : ℚ := 3/2
  let evening_price : ℚ := 1
  let production_cost : ℚ := 1
  let morning_sales : ℕ := total_loaves / 3
  let afternoon_sales : ℕ := (total_loaves - morning_sales) / 2
  let evening_sales : ℕ := total_loaves - morning_sales - afternoon_sales
  let total_revenue : ℚ := morning_sales * morning_price + 
                           afternoon_sales * afternoon_price + 
                           evening_sales * evening_price
  let total_cost : ℚ := total_loaves * production_cost
  let profit : ℚ := total_revenue - total_cost
  profit = 50 := by
sorry

end NUMINAMATH_CALUDE_lindas_bakery_profit_l3528_352847


namespace NUMINAMATH_CALUDE_max_earnings_is_zero_l3528_352818

/-- Represents the state of the boxes and Sisyphus's earnings -/
structure BoxState where
  a : ℕ  -- number of stones in box A
  b : ℕ  -- number of stones in box B
  c : ℕ  -- number of stones in box C
  earnings : ℤ  -- Sisyphus's current earnings (can be negative)

/-- Represents a move of a stone from one box to another -/
inductive Move
  | AtoB | AtoC | BtoA | BtoC | CtoA | CtoB

/-- Applies a move to the current state and returns the new state -/
def applyMove (state : BoxState) (move : Move) : BoxState :=
  match move with
  | Move.AtoB => { state with 
      a := state.a - 1, 
      b := state.b + 1, 
      earnings := state.earnings + state.b - (state.a - 1) }
  | Move.AtoC => { state with 
      a := state.a - 1, 
      c := state.c + 1, 
      earnings := state.earnings + state.c - (state.a - 1) }
  | Move.BtoA => { state with 
      b := state.b - 1, 
      a := state.a + 1, 
      earnings := state.earnings + state.a - (state.b - 1) }
  | Move.BtoC => { state with 
      b := state.b - 1, 
      c := state.c + 1, 
      earnings := state.earnings + state.c - (state.b - 1) }
  | Move.CtoA => { state with 
      c := state.c - 1, 
      a := state.a + 1, 
      earnings := state.earnings + state.a - (state.c - 1) }
  | Move.CtoB => { state with 
      c := state.c - 1, 
      b := state.b + 1, 
      earnings := state.earnings + state.b - (state.c - 1) }

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state -/
def applyMoves (initialState : BoxState) (moves : MoveSequence) : BoxState :=
  moves.foldl applyMove initialState

/-- Theorem: The maximum earnings of Sisyphus is 0 -/
theorem max_earnings_is_zero (initialState : BoxState) (moves : MoveSequence) :
  let finalState := applyMoves initialState moves
  (finalState.a = initialState.a ∧ 
   finalState.b = initialState.b ∧ 
   finalState.c = initialState.c) →
  finalState.earnings ≤ 0 := by
  sorry

#check max_earnings_is_zero

end NUMINAMATH_CALUDE_max_earnings_is_zero_l3528_352818


namespace NUMINAMATH_CALUDE_cube_face_sum_l3528_352823

theorem cube_face_sum (a b c d e f : ℕ+) : 
  a * b * c + a * e * c + a * b * f + a * e * f + 
  d * b * c + d * e * c + d * b * f + d * e * f = 1001 →
  a + b + c + d + e + f = 31 := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l3528_352823


namespace NUMINAMATH_CALUDE_parallelepiped_volume_example_l3528_352805

/-- The volume of a parallelepiped with given dimensions -/
def parallelepipedVolume (base height depth : ℝ) : ℝ :=
  base * depth * height

/-- Theorem: The volume of a parallelepiped with base 28 cm, height 32 cm, and depth 15 cm is 13440 cubic centimeters -/
theorem parallelepiped_volume_example : parallelepipedVolume 28 32 15 = 13440 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_example_l3528_352805


namespace NUMINAMATH_CALUDE_angle_complement_when_supplement_is_110_l3528_352808

/-- If the supplement of an angle is 110°, then its complement is 20°. -/
theorem angle_complement_when_supplement_is_110 (x : ℝ) : 
  x + 110 = 180 → 90 - (180 - 110) = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_when_supplement_is_110_l3528_352808


namespace NUMINAMATH_CALUDE_fraction_simplification_l3528_352851

theorem fraction_simplification : 
  (2+4+6+8+10+12+14+16+18+20) / (1+2+3+4+5+6+7+8+9+10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3528_352851


namespace NUMINAMATH_CALUDE_x_minus_y_equals_eight_l3528_352865

theorem x_minus_y_equals_eight (x y : ℝ) 
  (h1 : |x| + x - y = 14) 
  (h2 : x + |y| + y = 6) : 
  x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_eight_l3528_352865


namespace NUMINAMATH_CALUDE_faucet_turning_is_rotational_motion_l3528_352864

/-- A motion that involves revolving around a center and changing direction -/
structure FaucetTurning where
  revolves_around_center : Bool
  direction_changes : Bool

/-- Definition of rotational motion -/
def is_rotational_motion (motion : FaucetTurning) : Prop :=
  motion.revolves_around_center ∧ motion.direction_changes

/-- Theorem: Turning a faucet by hand is a rotational motion -/
theorem faucet_turning_is_rotational_motion :
  ∀ (faucet_turning : FaucetTurning),
  faucet_turning.revolves_around_center = true →
  faucet_turning.direction_changes = true →
  is_rotational_motion faucet_turning :=
by
  sorry

end NUMINAMATH_CALUDE_faucet_turning_is_rotational_motion_l3528_352864


namespace NUMINAMATH_CALUDE_batsman_highest_score_l3528_352854

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (overall_average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (i : overall_average = 63)
  (j : score_difference = 150)
  (k : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (total_innings : ℚ) * overall_average = 
      ((total_innings - 2 : ℕ) : ℚ) * average_excluding_extremes + highest_score + lowest_score ∧
    highest_score = 248 := by
  sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l3528_352854


namespace NUMINAMATH_CALUDE_nested_cube_root_l3528_352807

theorem nested_cube_root (N : ℝ) (h : N > 1) :
  (N * (N * (N * N^(1/3))^(1/3))^(1/3))^(1/3) = N^(40/81) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_root_l3528_352807


namespace NUMINAMATH_CALUDE_problem_statement_l3528_352898

theorem problem_statement : 
  (∃ x : ℝ, x - 2 > 0) ∧ ¬(∀ x : ℝ, x ≥ 0 → Real.sqrt x < x) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3528_352898


namespace NUMINAMATH_CALUDE_expression_evaluation_l3528_352867

theorem expression_evaluation : 
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3528_352867


namespace NUMINAMATH_CALUDE_biff_ticket_cost_l3528_352885

/-- Represents the cost of Biff's bus ticket in dollars -/
def ticket_cost : ℝ := 11

/-- Represents the cost of drinks and snacks in dollars -/
def snacks_cost : ℝ := 3

/-- Represents the cost of headphones in dollars -/
def headphones_cost : ℝ := 16

/-- Represents Biff's hourly rate for online work in dollars per hour -/
def online_rate : ℝ := 12

/-- Represents the hourly cost of WiFi access in dollars per hour -/
def wifi_cost : ℝ := 2

/-- Represents the duration of the bus trip in hours -/
def trip_duration : ℝ := 3

theorem biff_ticket_cost :
  ticket_cost + snacks_cost + headphones_cost + wifi_cost * trip_duration =
  online_rate * trip_duration :=
by sorry

end NUMINAMATH_CALUDE_biff_ticket_cost_l3528_352885


namespace NUMINAMATH_CALUDE_equation_solution_l3528_352801

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3528_352801


namespace NUMINAMATH_CALUDE_simplify_expression_l3528_352855

theorem simplify_expression (x : ℝ) :
  3 * x^3 + 5 * x + 16 * x^2 + 15 - (7 - 3 * x^3 - 5 * x - 16 * x^2) = 
  6 * x^3 + 32 * x^2 + 10 * x + 8 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3528_352855


namespace NUMINAMATH_CALUDE_no_integer_roots_l3528_352853

theorem no_integer_roots : ∀ x : ℤ, x^3 - 4*x^2 - 4*x + 24 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3528_352853


namespace NUMINAMATH_CALUDE_evaluate_expression_l3528_352860

theorem evaluate_expression : 3000 * (3000 ^ 2999) ^ 2 = 3000 ^ 5999 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3528_352860


namespace NUMINAMATH_CALUDE_jasmine_concentration_proof_l3528_352884

/-- Proves that adding 5 liters of jasmine and 15 liters of water to an 80-liter solution
    with 10% jasmine results in a new solution with 13% jasmine concentration. -/
theorem jasmine_concentration_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_jasmine : ℝ) (added_water : ℝ) (final_concentration : ℝ) : 
  initial_volume = 80 →
  initial_concentration = 0.10 →
  added_jasmine = 5 →
  added_water = 15 →
  final_concentration = 0.13 →
  (initial_volume * initial_concentration + added_jasmine) / 
  (initial_volume + added_jasmine + added_water) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_jasmine_concentration_proof_l3528_352884


namespace NUMINAMATH_CALUDE_triangle_transformation_indefinite_l3528_352873

/-- A triangle can undergo the given transformation indefinitely iff it's equilateral -/
theorem triangle_transformation_indefinite (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  (∀ n : ℕ, ∃ a' b' c' : ℝ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    a' = (-a + b + c) / 2 ∧ 
    b' = (a - b + c) / 2 ∧ 
    c' = (a + b - c) / 2) ↔ 
  (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_transformation_indefinite_l3528_352873


namespace NUMINAMATH_CALUDE_initial_number_of_girls_l3528_352821

/-- The number of girls in the initial group -/
def n : ℕ := sorry

/-- The initial average weight of the girls -/
def A : ℝ := sorry

/-- The weight of the new girl -/
def new_weight : ℝ := 100

/-- The weight of the girl being replaced -/
def replaced_weight : ℝ := 50

/-- The increase in average weight -/
def avg_increase : ℝ := 5

theorem initial_number_of_girls :
  (n * A - replaced_weight + new_weight) / n = A + avg_increase →
  n = 10 := by sorry

end NUMINAMATH_CALUDE_initial_number_of_girls_l3528_352821


namespace NUMINAMATH_CALUDE_tournament_schools_l3528_352846

theorem tournament_schools (n : ℕ) : 
  (∀ (school : ℕ), school ≤ n → ∃ (team : Fin 4 → ℕ), 
    (∀ i j, i ≠ j → team i ≠ team j) ∧ 
    (∃ (theo leah mark nora : ℕ), 
      theo = (4 * n + 1) / 2 ∧
      leah = 48 ∧ 
      mark = 75 ∧ 
      nora = 97 ∧
      theo < leah ∧ theo < mark ∧ theo < nora ∧
      (∀ k, k ∈ [theo, leah, mark, nora] → k ≤ 4 * n) ∧
      (∀ k, k ∉ [theo, leah, mark, nora] → 
        (k < theo ∧ k ≤ 4 * n - 3) ∨ (k > theo ∧ k ≤ 4 * n)))) → 
  n = 36 := by
sorry

end NUMINAMATH_CALUDE_tournament_schools_l3528_352846


namespace NUMINAMATH_CALUDE_angle_C_measure_side_c_length_l3528_352831

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition1 (t : Triangle) : Prop :=
  t.b * Real.cos t.A + t.a * Real.cos t.B = -2 * t.c * Real.cos t.C

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.b = 2 * t.a ∧ 
  (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3

-- Theorem 1
theorem angle_C_measure (t : Triangle) (h : satisfiesCondition1 t) : t.C = 2 * π / 3 := by
  sorry

-- Theorem 2
theorem side_c_length (t : Triangle) (h1 : satisfiesCondition1 t) (h2 : satisfiesCondition2 t) : 
  t.c = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_side_c_length_l3528_352831


namespace NUMINAMATH_CALUDE_existence_of_coefficients_l3528_352809

/-- Two polynomials with coefficients A, B, C, D -/
def poly1 (A B C D : ℝ) (x : ℝ) : ℝ := x^6 + 4*x^5 + A*x^4 + B*x^3 + C*x^2 + D*x + 1
def poly2 (A B C D : ℝ) (x : ℝ) : ℝ := x^6 - 4*x^5 + A*x^4 - B*x^3 + C*x^2 - D*x + 1

/-- The product of the two polynomials -/
def product (A B C D : ℝ) (x : ℝ) : ℝ := (poly1 A B C D x) * (poly2 A B C D x)

/-- Theorem stating the existence of coefficients satisfying the conditions -/
theorem existence_of_coefficients : ∃ (A B C D : ℝ), 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0) ∧ 
  (∃ (k : ℝ), ∀ (x : ℝ), product A B C D x = x^12 + k*x^6 + 1) ∧
  (∃ (b c : ℝ), ∀ (x : ℝ), 
    poly1 A B C D x = (x^3 + 2*x^2 + b*x + c)^2 ∧
    poly2 A B C D x = (x^3 - 2*x^2 + b*x - c)^2) :=
sorry

end NUMINAMATH_CALUDE_existence_of_coefficients_l3528_352809


namespace NUMINAMATH_CALUDE_music_class_size_l3528_352869

theorem music_class_size (total_students : ℕ) (art_music_overlap : ℕ) :
  total_students = 60 →
  art_music_overlap = 8 →
  ∃ (art_only music_only : ℕ),
    total_students = art_only + music_only + art_music_overlap ∧
    art_only + art_music_overlap = music_only + art_music_overlap + 10 →
    music_only + art_music_overlap = 33 :=
by sorry

end NUMINAMATH_CALUDE_music_class_size_l3528_352869


namespace NUMINAMATH_CALUDE_milk_water_ratio_in_combined_mixture_l3528_352872

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Calculates the ratio of milk to water in a mixture -/
def ratioMilkToWater (m : Mixture) : ℚ × ℚ :=
  (m.milk, m.water)

/-- Combines multiple mixtures into a single mixture -/
def combineMixtures (mixtures : List Mixture) : Mixture :=
  { milk := mixtures.map (·.milk) |>.sum,
    water := mixtures.map (·.water) |>.sum }

theorem milk_water_ratio_in_combined_mixture :
  let m1 := Mixture.mk (7 : ℚ) (2 : ℚ)
  let m2 := Mixture.mk (8 : ℚ) (1 : ℚ)
  let m3 := Mixture.mk (9 : ℚ) (3 : ℚ)
  let combined := combineMixtures [m1, m2, m3]
  ratioMilkToWater combined = (29, 7) := by
  sorry

#check milk_water_ratio_in_combined_mixture

end NUMINAMATH_CALUDE_milk_water_ratio_in_combined_mixture_l3528_352872


namespace NUMINAMATH_CALUDE_monochromatic_subgrid_exists_l3528_352832

/-- Represents a cell color -/
inductive Color
| Black
| White

/-- Represents the grid -/
def Grid := Fin 3 → Fin 7 → Color

/-- Checks if a 2x2 subgrid has all cells of the same color -/
def has_monochromatic_2x2_subgrid (g : Grid) : Prop :=
  ∃ (i : Fin 2) (j : Fin 6),
    g i j = g i (j + 1) ∧
    g i j = g (i + 1) j ∧
    g i j = g (i + 1) (j + 1)

/-- Main theorem: Any 3x7 grid with black and white cells contains a monochromatic 2x2 subgrid -/
theorem monochromatic_subgrid_exists (g : Grid) : 
  has_monochromatic_2x2_subgrid g :=
sorry

end NUMINAMATH_CALUDE_monochromatic_subgrid_exists_l3528_352832


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3528_352800

def f (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 6

theorem roots_of_polynomial :
  (f (-1) = 0) ∧ (f 1 = 0) ∧ (f 6 = 0) ∧
  (∀ x : ℝ, f x = 0 → x = -1 ∨ x = 1 ∨ x = 6) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3528_352800


namespace NUMINAMATH_CALUDE_total_cookies_calculation_l3528_352895

/-- The number of cookies Kristy baked -/
def total_cookies : ℕ := sorry

/-- The number of cookies Kristy ate -/
def kristy_ate : ℕ := 3

/-- The number of cookies Kristy's brother took -/
def brother_took : ℕ := 2

/-- The number of cookies the first friend took -/
def first_friend_took : ℕ := 4

/-- The number of cookies the second friend took (net) -/
def second_friend_took : ℕ := 4

/-- The number of cookies the third friend took -/
def third_friend_took : ℕ := 8

/-- The number of cookies the fourth friend took -/
def fourth_friend_took : ℕ := 3

/-- The number of cookies the fifth friend took -/
def fifth_friend_took : ℕ := 7

/-- The number of cookies left -/
def cookies_left : ℕ := 5

/-- Theorem stating that the total number of cookies is equal to the sum of all distributed cookies and the remaining cookies -/
theorem total_cookies_calculation :
  total_cookies = kristy_ate + brother_took + first_friend_took + second_friend_took +
                  third_friend_took + fourth_friend_took + fifth_friend_took + cookies_left :=
by sorry

end NUMINAMATH_CALUDE_total_cookies_calculation_l3528_352895


namespace NUMINAMATH_CALUDE_social_practice_choices_l3528_352866

/-- The number of classes in the first year of high school -/
def first_year_classes : Nat := 14

/-- The number of classes in the second year of high school -/
def second_year_classes : Nat := 14

/-- The number of classes in the third year of high school -/
def third_year_classes : Nat := 15

/-- The number of ways to choose students from 1 class to participate in social practice activities -/
def choose_one_class : Nat := first_year_classes + second_year_classes + third_year_classes

/-- The number of ways to choose students from one class in each grade to participate in social practice activities -/
def choose_one_from_each : Nat := first_year_classes * second_year_classes * third_year_classes

/-- The number of ways to choose students from 2 classes to participate in social practice activities, with the requirement that these 2 classes are from different grades -/
def choose_two_different_grades : Nat := 
  first_year_classes * second_year_classes + 
  first_year_classes * third_year_classes + 
  second_year_classes * third_year_classes

theorem social_practice_choices : 
  choose_one_class = 43 ∧ 
  choose_one_from_each = 2940 ∧ 
  choose_two_different_grades = 616 := by sorry

end NUMINAMATH_CALUDE_social_practice_choices_l3528_352866


namespace NUMINAMATH_CALUDE_salt_added_amount_l3528_352857

/-- Represents the salt solution problem --/
structure SaltSolution where
  initial_volume : ℝ
  initial_salt_concentration : ℝ
  evaporation_fraction : ℝ
  water_added : ℝ
  final_salt_concentration : ℝ

/-- Calculates the amount of salt added to the solution --/
def salt_added (s : SaltSolution) : ℝ :=
  let initial_salt := s.initial_volume * s.initial_salt_concentration
  let water_evaporated := s.initial_volume * s.evaporation_fraction
  let remaining_volume := s.initial_volume - water_evaporated
  let new_volume := remaining_volume + s.water_added
  let final_salt := new_volume * s.final_salt_concentration
  final_salt - initial_salt

/-- The theorem stating the amount of salt added --/
theorem salt_added_amount (s : SaltSolution) 
  (h1 : s.initial_volume = 149.99999999999994)
  (h2 : s.initial_salt_concentration = 0.20)
  (h3 : s.evaporation_fraction = 0.25)
  (h4 : s.water_added = 10)
  (h5 : s.final_salt_concentration = 1/3) :
  ∃ ε > 0, |salt_added s - 10.83| < ε :=
sorry

end NUMINAMATH_CALUDE_salt_added_amount_l3528_352857


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3528_352888

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : a + b ≤ c * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3528_352888


namespace NUMINAMATH_CALUDE_C_satisfies_equation_C_specific_value_l3528_352820

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2
def B (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

-- Define C as a function of x and y
def C (x y : ℝ) : ℝ := -x^2 + 10*x*y - y^2

-- Theorem 1: C satisfies the given equation
theorem C_satisfies_equation (x y : ℝ) : 3 * A x y - 2 * B x y + C x y = 0 := by
  sorry

-- Theorem 2: C equals -57/4 when x = 1/2 and y = -2
theorem C_specific_value : C (1/2) (-2) = -57/4 := by
  sorry

end NUMINAMATH_CALUDE_C_satisfies_equation_C_specific_value_l3528_352820


namespace NUMINAMATH_CALUDE_no_real_roots_l3528_352810

theorem no_real_roots (a b c : ℕ) (ha : a < 1000000) (hb : b < 1000000) (hc : c < 1000000) :
  ¬∃ x : ℝ, (a * x^2)^(1/21) + (b * x)^(1/21) + c^(1/21) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3528_352810


namespace NUMINAMATH_CALUDE_range_sum_bounds_l3528_352829

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- State the theorem
theorem range_sum_bounds :
  ∃ (m n : ℝ), (∀ x, m ≤ f x ∧ f x ≤ n) ∧
  (m = -5 ∧ n = 4) →
  1 ≤ m + n ∧ m + n ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_range_sum_bounds_l3528_352829


namespace NUMINAMATH_CALUDE_calculate_expression_l3528_352889

theorem calculate_expression : 125^2 - 50 * 125 + 25^2 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3528_352889


namespace NUMINAMATH_CALUDE_ellipse_angle_tangent_product_l3528_352876

/-- Given an ellipse with eccentricity e and a point P on the ellipse,
    if α is the angle PF₁F₂ and β is the angle PF₂F₁, where F₁ and F₂ are the foci,
    then tan(α/2) * tan(β/2) = (1 - e) / (1 + e) -/
theorem ellipse_angle_tangent_product (a b : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (α β e : ℝ)
  (h_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (h_foci : F₁ ≠ F₂)
  (h_eccentricity : e = Real.sqrt (a^2 - b^2) / a)
  (h_angle_α : α = Real.arccos ((P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2)) /
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)))
  (h_angle_β : β = Real.arccos ((P.1 - F₂.1) * (F₁.1 - F₂.1) + (P.2 - F₂.2) * (F₁.2 - F₂.2)) /
    (Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) * Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)))
  : Real.tan (α/2) * Real.tan (β/2) = (1 - e) / (1 + e) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_angle_tangent_product_l3528_352876


namespace NUMINAMATH_CALUDE_mary_seashells_l3528_352893

theorem mary_seashells (sam_seashells : ℕ) (total_seashells : ℕ) 
  (h1 : sam_seashells = 18) 
  (h2 : total_seashells = 65) : 
  total_seashells - sam_seashells = 47 := by
  sorry

end NUMINAMATH_CALUDE_mary_seashells_l3528_352893


namespace NUMINAMATH_CALUDE_constant_expression_in_linear_system_l3528_352881

theorem constant_expression_in_linear_system (a k : ℝ) (x y : ℝ → ℝ) :
  (∀ a, x a + 2 * y a = -a + 1) →
  (∀ a, x a - 3 * y a = 4 * a + 6) →
  (∃ c, ∀ a, k * x a - y a = c) →
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_constant_expression_in_linear_system_l3528_352881


namespace NUMINAMATH_CALUDE_small_tile_position_l3528_352870

/-- Represents a position on the 7x7 grid -/
structure Position :=
  (x : Fin 7)
  (y : Fin 7)

/-- Represents a 1x3 tile on the grid -/
structure Tile :=
  (start : Position)
  (horizontal : Bool)

/-- The configuration of the 7x7 grid -/
structure GridConfig :=
  (tiles : Finset Tile)
  (small_tile : Position)

/-- Checks if a position is at the center or adjoins a boundary -/
def is_center_or_boundary (p : Position) : Prop :=
  p.x = 0 ∨ p.x = 3 ∨ p.x = 6 ∨ p.y = 0 ∨ p.y = 3 ∨ p.y = 6

/-- Checks if the configuration is valid -/
def is_valid_config (config : GridConfig) : Prop :=
  config.tiles.card = 16 ∧
  ∀ t ∈ config.tiles, (t.horizontal → t.start.y < 6) ∧
                      (¬t.horizontal → t.start.x < 6)

theorem small_tile_position (config : GridConfig) 
  (h : is_valid_config config) :
  is_center_or_boundary config.small_tile :=
sorry

end NUMINAMATH_CALUDE_small_tile_position_l3528_352870


namespace NUMINAMATH_CALUDE_remainder_452867_div_9_l3528_352882

theorem remainder_452867_div_9 : 452867 % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_452867_div_9_l3528_352882


namespace NUMINAMATH_CALUDE_motel_weekly_charge_l3528_352842

/-- The weekly charge for Casey's motel stay --/
def weekly_charge : ℕ → Prop :=
  fun w => 
    let months : ℕ := 3
    let weeks_per_month : ℕ := 4
    let monthly_rate : ℕ := 1000
    let savings : ℕ := 360
    let total_weeks : ℕ := months * weeks_per_month
    let total_monthly_cost : ℕ := months * monthly_rate
    (total_weeks * w = total_monthly_cost + savings) ∧ (w = 280)

/-- Proof that the weekly charge is $280 --/
theorem motel_weekly_charge : weekly_charge 280 := by
  sorry

end NUMINAMATH_CALUDE_motel_weekly_charge_l3528_352842


namespace NUMINAMATH_CALUDE_triangle_special_angle_l3528_352875

/-- Given a triangle ABC where b = c and a² = 2b²(1 - sin A), prove that A = π/4 -/
theorem triangle_special_angle (a b c : ℝ) (A : ℝ) 
  (h1 : b = c) 
  (h2 : a^2 = 2 * b^2 * (1 - Real.sin A)) : 
  A = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l3528_352875


namespace NUMINAMATH_CALUDE_segment_ratio_l3528_352834

/-- Given a line segment AD with points B and C on it, where AB = 3BD and AC = 5CD,
    the length of BC is 1/12 of the length of AD. -/
theorem segment_ratio (A B C D : ℝ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : C ≤ D)
  (h4 : B - A = 3 * (D - B)) (h5 : C - A = 5 * (D - C)) :
  (C - B) = (1 / 12) * (D - A) := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l3528_352834


namespace NUMINAMATH_CALUDE_hedge_trimming_charge_equals_685_l3528_352806

/-- Calculates the total charge for trimming a hedge with various shapes -/
def hedge_trimming_charge (basic_trim_price : ℚ) (sphere_price : ℚ) (pyramid_price : ℚ) 
  (cube_price : ℚ) (combined_shape_extra : ℚ) (total_boxwoods : ℕ) (sphere_count : ℕ) 
  (pyramid_count : ℕ) (cube_count : ℕ) (sphere_pyramid_count : ℕ) (sphere_cube_count : ℕ) : ℚ :=
  let basic_trim_total := basic_trim_price * total_boxwoods
  let sphere_total := sphere_price * sphere_count
  let pyramid_total := pyramid_price * pyramid_count
  let cube_total := cube_price * cube_count
  let sphere_pyramid_total := (sphere_price + pyramid_price + combined_shape_extra) * sphere_pyramid_count
  let sphere_cube_total := (sphere_price + cube_price + combined_shape_extra) * sphere_cube_count
  basic_trim_total + sphere_total + pyramid_total + cube_total + sphere_pyramid_total + sphere_cube_total

/-- The total charge for trimming the hedge is $685.00 -/
theorem hedge_trimming_charge_equals_685 : 
  hedge_trimming_charge 5 15 20 25 10 40 2 5 3 4 2 = 685 := by
  sorry

end NUMINAMATH_CALUDE_hedge_trimming_charge_equals_685_l3528_352806


namespace NUMINAMATH_CALUDE_rectangle_area_l3528_352886

/-- Represents a rectangle with given properties -/
structure Rectangle where
  breadth : ℝ
  length : ℝ
  perimeter : ℝ

/-- Theorem: Area of a specific rectangle -/
theorem rectangle_area (r : Rectangle) 
  (h1 : r.length = 3 * r.breadth) 
  (h2 : r.perimeter = 88) : 
  r.length * r.breadth = 363 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l3528_352886


namespace NUMINAMATH_CALUDE_bus_capacity_l3528_352813

/-- The number of students that can be accommodated by a given number of buses,
    each with a specified number of columns and rows of seats. -/
def total_students (buses : ℕ) (columns : ℕ) (rows : ℕ) : ℕ :=
  buses * columns * rows

/-- Theorem stating that 6 buses with 4 columns and 10 rows each can accommodate 240 students. -/
theorem bus_capacity : total_students 6 4 10 = 240 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_l3528_352813


namespace NUMINAMATH_CALUDE_fraction_sum_l3528_352837

theorem fraction_sum : (3 : ℚ) / 8 + 9 / 12 + 5 / 6 = 47 / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3528_352837


namespace NUMINAMATH_CALUDE_circle_properties_l3528_352825

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 5 = 0

-- Define the point M
def point_M : ℝ × ℝ := (3, -1)

-- Define the point N
def point_N : ℝ × ℝ := (1, 2)

-- Define the equation of the required circle
def required_circle (x y : ℝ) : Prop := (x - 20/7)^2 + (y - 15/14)^2 = 845/196

-- Theorem statement
theorem circle_properties :
  -- The required circle passes through point M
  required_circle point_M.1 point_M.2 ∧
  -- The required circle passes through point N
  required_circle point_N.1 point_N.2 ∧
  -- The required circle is tangent to circle C at point N
  (∃ (t : ℝ), t ≠ 0 ∧
    ∀ (x y : ℝ),
      circle_C x y ↔ required_circle x y ∨
      ((x - point_N.1) = t * (40/7 - 2*point_N.1) ∧
       (y - point_N.2) = t * (30/7 - 2*point_N.2))) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3528_352825


namespace NUMINAMATH_CALUDE_romans_remaining_coins_l3528_352828

/-- Represents the problem of calculating Roman's remaining gold coins --/
theorem romans_remaining_coins 
  (initial_worth : ℕ) 
  (coins_sold : ℕ) 
  (money_after_sale : ℕ) 
  (h1 : initial_worth = 20)
  (h2 : coins_sold = 3)
  (h3 : money_after_sale = 12) :
  initial_worth / (money_after_sale / coins_sold) - coins_sold = 2 :=
sorry

end NUMINAMATH_CALUDE_romans_remaining_coins_l3528_352828


namespace NUMINAMATH_CALUDE_angle_C_is_60_degrees_a_and_b_values_l3528_352827

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧
  t.c = Real.sqrt 7 ∧
  4 * (Real.sin (t.A + t.B) / 2)^2 - Real.cos (2 * t.C) = 7/2 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem 1: Prove that C = 60°
theorem angle_C_is_60_degrees (t : Triangle) 
  (h : triangle_conditions t) : t.C = Real.pi / 3 := by
  sorry

-- Theorem 2: Prove that if a > b, then a = 3 and b = 2
theorem a_and_b_values (t : Triangle) 
  (h1 : triangle_conditions t) (h2 : t.a > t.b) : t.a = 3 ∧ t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_60_degrees_a_and_b_values_l3528_352827


namespace NUMINAMATH_CALUDE_cube_after_carving_l3528_352826

def cube_side_length : ℝ := 9

-- Volume of the cube after carving the cross-shaped groove
def remaining_volume : ℝ := 639

-- Surface area of the cube after carving the cross-shaped groove
def new_surface_area : ℝ := 510

-- Theorem statement
theorem cube_after_carving (groove_volume : ℝ) (groove_surface_area : ℝ) :
  cube_side_length ^ 3 - groove_volume = remaining_volume ∧
  6 * cube_side_length ^ 2 + groove_surface_area = new_surface_area :=
by sorry

end NUMINAMATH_CALUDE_cube_after_carving_l3528_352826


namespace NUMINAMATH_CALUDE_total_cuts_after_six_operations_l3528_352874

def cuts_in_operation (n : ℕ) : ℕ :=
  3 * 4^(n - 1)

def total_cuts (n : ℕ) : ℕ :=
  (List.range n).map (cuts_in_operation ∘ (· + 1)) |> List.sum

theorem total_cuts_after_six_operations :
  total_cuts 6 = 4095 := by
  sorry

end NUMINAMATH_CALUDE_total_cuts_after_six_operations_l3528_352874


namespace NUMINAMATH_CALUDE_intersection_cylinders_in_sphere_l3528_352859

/-- Theorem: Intersection of three perpendicular unit cylinders is contained in a sphere of radius √(3/2) -/
theorem intersection_cylinders_in_sphere (a b c d e f : ℝ) (x y z : ℝ) : 
  ((x - a)^2 + (y - b)^2 ≤ 1) →
  ((y - c)^2 + (z - d)^2 ≤ 1) →
  ((z - e)^2 + (x - f)^2 ≤ 1) →
  ∃ (center_x center_y center_z : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2 ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_cylinders_in_sphere_l3528_352859


namespace NUMINAMATH_CALUDE_first_player_always_wins_l3528_352852

/-- Represents a point on a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a color of a dot --/
inductive Color
  | Red
  | Blue

/-- Represents a dot on the plane --/
structure Dot where
  point : Point
  color : Color

/-- Represents the game state --/
structure GameState where
  dots : List Dot

/-- Represents a player's strategy --/
def Strategy := GameState → Point

/-- Checks if three points form an equilateral triangle --/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop := sorry

/-- The main theorem stating that the first player can always win --/
theorem first_player_always_wins :
  ∀ (second_player_strategy : Strategy),
  ∃ (first_player_strategy : Strategy) (n : ℕ),
  ∀ (game : GameState),
  ∃ (p1 p2 p3 : Point),
  (p1 ∈ game.dots.map Dot.point) ∧
  (p2 ∈ game.dots.map Dot.point) ∧
  (p3 ∈ game.dots.map Dot.point) ∧
  isEquilateralTriangle p1 p2 p3 :=
sorry

end NUMINAMATH_CALUDE_first_player_always_wins_l3528_352852


namespace NUMINAMATH_CALUDE_second_derivative_sin_plus_cos_l3528_352880

open Real

theorem second_derivative_sin_plus_cos :
  let f : ℝ → ℝ := fun x ↦ sin x + cos x
  ∀ x : ℝ, (deriv^[2] f) x = -(cos x) - sin x := by
  sorry

end NUMINAMATH_CALUDE_second_derivative_sin_plus_cos_l3528_352880


namespace NUMINAMATH_CALUDE_original_number_proof_l3528_352830

theorem original_number_proof (x : ℚ) : 1 + 1 / x = 8 / 3 → x = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3528_352830


namespace NUMINAMATH_CALUDE_lighter_box_weight_l3528_352844

/-- Given a shipment of boxes with the following properties:
  * There are 30 boxes in total
  * Some boxes weigh W pounds (lighter boxes)
  * The rest of the boxes weigh 20 pounds (heavier boxes)
  * The initial average weight is 18 pounds
  * After removing 15 of the 20-pound boxes, the new average weight is 16 pounds
  Prove that the weight of the lighter boxes (W) is 16 pounds. -/
theorem lighter_box_weight (total_boxes : ℕ) (W : ℝ) (heavy_box_weight : ℝ) 
  (initial_avg : ℝ) (new_avg : ℝ) (removed_boxes : ℕ) :
  total_boxes = 30 →
  heavy_box_weight = 20 →
  initial_avg = 18 →
  new_avg = 16 →
  removed_boxes = 15 →
  (∃ (light_boxes heavy_boxes : ℕ), 
    light_boxes + heavy_boxes = total_boxes ∧
    (light_boxes * W + heavy_boxes * heavy_box_weight) / total_boxes = initial_avg ∧
    ((light_boxes * W + (heavy_boxes - removed_boxes) * heavy_box_weight) / 
      (total_boxes - removed_boxes) = new_avg)) →
  W = 16 := by
sorry

end NUMINAMATH_CALUDE_lighter_box_weight_l3528_352844


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_intersection_A_B_in_U_l3528_352861

-- Define the sets A, B, and U
def A : Set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ -2}
def B : Set ℝ := {x : ℝ | x + 3 ≥ 0}
def U : Set ℝ := {x : ℝ | x ≤ -1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | -3 ≤ x ∧ x ≤ -2} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -4} := by sorry

-- Theorem for the complement of A ∩ B in U
theorem complement_intersection_A_B_in_U : (A ∩ B)ᶜ ∩ U = {x : ℝ | x < -3 ∨ (-2 < x ∧ x ≤ -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_intersection_A_B_in_U_l3528_352861


namespace NUMINAMATH_CALUDE_triangle_area_l3528_352804

/-- Given a triangle with perimeter 42 cm and inradius 5.0 cm, its area is 105 cm². -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 42 → inradius = 5 → area = perimeter / 2 * inradius → area = 105 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3528_352804


namespace NUMINAMATH_CALUDE_sector_area_l3528_352890

/-- The area of a sector with radius 2 and central angle 2π/3 is 4π/3 -/
theorem sector_area (r : ℝ) (θ : ℝ) (area : ℝ) : 
  r = 2 → θ = 2 * π / 3 → area = (1 / 2) * r^2 * θ → area = 4 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3528_352890


namespace NUMINAMATH_CALUDE_album_ratio_proof_l3528_352896

/-- Prove that given the conditions, the ratio of Katrina's albums to Bridget's albums is 6:1 -/
theorem album_ratio_proof (miriam katrina bridget adele : ℕ) 
  (h1 : miriam = 5 * katrina)
  (h2 : ∃ n : ℕ, katrina = n * bridget)
  (h3 : bridget = adele - 15)
  (h4 : miriam + katrina + bridget + adele = 585)
  (h5 : adele = 30) :
  katrina / bridget = 6 := by
sorry

end NUMINAMATH_CALUDE_album_ratio_proof_l3528_352896


namespace NUMINAMATH_CALUDE_max_person_money_100_2000_380_l3528_352856

/-- Given a group of people and their money distribution, 
    calculate the maximum amount one person can have. -/
def maxPersonMoney (n : ℕ) (total : ℕ) (maxTen : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum amount one person can have 
    under the given conditions. -/
theorem max_person_money_100_2000_380 : 
  maxPersonMoney 100 2000 380 = 218 := by sorry

end NUMINAMATH_CALUDE_max_person_money_100_2000_380_l3528_352856


namespace NUMINAMATH_CALUDE_max_quotient_is_21996_l3528_352877

def is_valid_divisor (d : ℕ) : Prop :=
  d ≥ 10 ∧ d < 100

def quotient_hundreds_condition (dividend : ℕ) (divisor : ℕ) : Prop :=
  let q := dividend / divisor
  (q / 100) * divisor ≥ 200 ∧ (q / 100) * divisor < 300

def max_quotient_dividend (dividends : List ℕ) : ℕ := sorry

theorem max_quotient_is_21996 :
  let dividends := [21944, 21996, 24054, 24111]
  ∃ d : ℕ, is_valid_divisor d ∧ 
           quotient_hundreds_condition (max_quotient_dividend dividends) d ∧
           max_quotient_dividend dividends = 21996 := by sorry

end NUMINAMATH_CALUDE_max_quotient_is_21996_l3528_352877


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l3528_352835

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  longerBase : ℝ
  baseAngle : ℝ
  height : ℝ

/-- Calculate the area of the isosceles trapezoid -/
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is 100 -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    longerBase := 20,
    baseAngle := Real.arcsin 0.6,
    height := 9
  }
  trapezoidArea t = 100 := by sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l3528_352835


namespace NUMINAMATH_CALUDE_dog_accessible_area_l3528_352850

/-- Represents the shed's dimensions and rope configuration --/
structure DogTieSetup where
  shedSideLength : ℝ
  ropeLength : ℝ
  attachmentDistance : ℝ

/-- Calculates the area accessible to the dog --/
def accessibleArea (setup : DogTieSetup) : ℝ :=
  sorry

/-- Theorem stating the area accessible to the dog --/
theorem dog_accessible_area (setup : DogTieSetup) 
  (h1 : setup.shedSideLength = 30)
  (h2 : setup.ropeLength = 10)
  (h3 : setup.attachmentDistance = 5) :
  accessibleArea setup = 37.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_dog_accessible_area_l3528_352850


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l3528_352899

def f (x : ℝ) : ℝ := -x^2 + abs x

theorem f_is_even_and_decreasing :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l3528_352899


namespace NUMINAMATH_CALUDE_factorial_division_l3528_352822

theorem factorial_division : Nat.factorial 9 / Nat.factorial (9 - 3) = 504 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3528_352822


namespace NUMINAMATH_CALUDE_necklace_profit_is_1500_l3528_352833

/-- Calculates the profit from selling necklaces --/
def calculate_profit (charms_per_necklace : ℕ) (cost_per_charm : ℕ) (selling_price : ℕ) (necklaces_sold : ℕ) : ℕ :=
  let cost_per_necklace := charms_per_necklace * cost_per_charm
  let profit_per_necklace := selling_price - cost_per_necklace
  profit_per_necklace * necklaces_sold

/-- Proves that the profit from selling 30 necklaces is $1500 --/
theorem necklace_profit_is_1500 :
  calculate_profit 10 15 200 30 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_necklace_profit_is_1500_l3528_352833


namespace NUMINAMATH_CALUDE_secant_min_value_l3528_352894

/-- The secant function -/
noncomputable def sec (x : ℝ) : ℝ := 1 / Real.cos x

/-- The function y = a sec(bx) -/
noncomputable def f (a b x : ℝ) : ℝ := a * sec (b * x)

theorem secant_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, f a b x ≥ a) ∧ (∃ x, f a b x = a) →
  (∀ x, f a b x ≥ 3) ∧ (∃ x, f a b x = 3) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_secant_min_value_l3528_352894


namespace NUMINAMATH_CALUDE_john_sneezing_fit_duration_l3528_352863

/-- Calculates the duration of a sneezing fit given the time between sneezes and the number of sneezes. -/
def sneezingFitDuration (timeBetweenSneezes : ℕ) (numberOfSneezes : ℕ) : ℕ :=
  timeBetweenSneezes * numberOfSneezes

/-- Proves that a sneezing fit with 3 seconds between sneezes and 40 sneezes lasts 120 seconds. -/
theorem john_sneezing_fit_duration :
  sneezingFitDuration 3 40 = 120 := by
  sorry

#eval sneezingFitDuration 3 40

end NUMINAMATH_CALUDE_john_sneezing_fit_duration_l3528_352863


namespace NUMINAMATH_CALUDE_S_equals_seven_l3528_352848

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) -
  1 / (Real.sqrt 15 - Real.sqrt 14) +
  1 / (Real.sqrt 14 - Real.sqrt 13) -
  1 / (Real.sqrt 13 - Real.sqrt 12) +
  1 / (Real.sqrt 12 - 3)

theorem S_equals_seven : S = 7 := by
  sorry

end NUMINAMATH_CALUDE_S_equals_seven_l3528_352848


namespace NUMINAMATH_CALUDE_min_value_expression_l3528_352868

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 2) : 
  1/((2 - x)*(2 - y)*(2 - z)) + 1/((2 + x)*(2 + y)*(2 + z)) + 1/(1 + (x+y+z)/3) ≥ 2 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 1/((2 - x)*(2 - y)*(2 - z)) + 1/((2 + x)*(2 + y)*(2 + z)) + 1/(1 + (x+y+z)/3) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3528_352868


namespace NUMINAMATH_CALUDE_abs_neg_2023_l3528_352891

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l3528_352891


namespace NUMINAMATH_CALUDE_production_proof_l3528_352839

def average_production_problem (n : ℕ) (past_average current_average : ℚ) : Prop :=
  let past_total := n * past_average
  let today_production := (n + 1) * current_average - past_total
  today_production = 95

theorem production_proof :
  average_production_problem 8 50 55 := by
  sorry

end NUMINAMATH_CALUDE_production_proof_l3528_352839


namespace NUMINAMATH_CALUDE_right_triangle_pythagorean_l3528_352887

theorem right_triangle_pythagorean (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → -- Ensuring positive lengths
  (a^2 + b^2 = c^2) → -- Pythagorean theorem
  ((a = 12 ∧ b = 5) → c = 13) ∧ -- Part 1
  ((c = 10 ∧ b = 9) → a = Real.sqrt 19) -- Part 2
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_pythagorean_l3528_352887


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l3528_352815

theorem ratio_sum_problem (x y z a : ℚ) : 
  x / y = 3 / 4 →
  y / z = 4 / 6 →
  y = 15 * a + 5 →
  x + y + z = 52 →
  a = 11 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l3528_352815


namespace NUMINAMATH_CALUDE_gcf_40_56_l3528_352816

theorem gcf_40_56 : Nat.gcd 40 56 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcf_40_56_l3528_352816


namespace NUMINAMATH_CALUDE_magic_king_episodes_l3528_352892

/-- Calculates the total number of episodes for a TV show with the given parameters -/
def total_episodes (total_seasons : ℕ) (episodes_first_half : ℕ) (episodes_second_half : ℕ) : ℕ :=
  let half_seasons := total_seasons / 2
  half_seasons * episodes_first_half + half_seasons * episodes_second_half

/-- Theorem stating that a show with 10 seasons, 20 episodes per season in the first half,
    and 25 episodes per season in the second half, has a total of 225 episodes -/
theorem magic_king_episodes :
  total_episodes 10 20 25 = 225 := by
  sorry

#eval total_episodes 10 20 25

end NUMINAMATH_CALUDE_magic_king_episodes_l3528_352892


namespace NUMINAMATH_CALUDE_pencil_difference_l3528_352802

/-- The price of a single pencil in dollars -/
def pencil_price : ℚ := 0.04

/-- The number of pencils Jamar bought -/
def jamar_pencils : ℕ := 81

/-- The number of pencils Michael bought -/
def michael_pencils : ℕ := 104

/-- The amount Jamar paid in dollars -/
def jamar_paid : ℚ := 2.32

/-- The amount Michael paid in dollars -/
def michael_paid : ℚ := 3.24

theorem pencil_difference : 
  (pencil_price > 0.01) ∧ 
  (jamar_paid = pencil_price * jamar_pencils) ∧
  (michael_paid = pencil_price * michael_pencils) ∧
  (∃ n : ℕ, n^2 = jamar_pencils) →
  michael_pencils - jamar_pencils = 23 := by
sorry

end NUMINAMATH_CALUDE_pencil_difference_l3528_352802


namespace NUMINAMATH_CALUDE_unique_solution_prime_cube_equation_l3528_352849

theorem unique_solution_prime_cube_equation :
  ∀ (p m n : ℕ), 
    Prime p → 
    1 + p^n = m^3 → 
    p = 7 ∧ n = 1 ∧ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_prime_cube_equation_l3528_352849


namespace NUMINAMATH_CALUDE_fraction_well_defined_at_negative_one_l3528_352871

theorem fraction_well_defined_at_negative_one :
  ∀ x : ℝ, x = -1 → (x^2 + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_fraction_well_defined_at_negative_one_l3528_352871


namespace NUMINAMATH_CALUDE_doris_monthly_expenses_l3528_352814

/-- Calculates Doris's monthly expenses based on her work schedule and hourly rate -/
def monthly_expenses (hourly_rate : ℕ) (weekday_hours : ℕ) (saturday_hours : ℕ) (weeks : ℕ) : ℕ :=
  let weekly_hours := weekday_hours * 5 + saturday_hours
  let weekly_earnings := hourly_rate * weekly_hours
  weekly_earnings * weeks

/-- Proves that Doris's monthly expenses are $1200 given her work schedule and hourly rate -/
theorem doris_monthly_expenses :
  monthly_expenses 20 3 5 3 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_doris_monthly_expenses_l3528_352814


namespace NUMINAMATH_CALUDE_eggplant_basket_weight_l3528_352845

def cucumber_baskets : ℕ := 25
def eggplant_baskets : ℕ := 32
def total_weight : ℕ := 1870
def cucumber_basket_weight : ℕ := 30

theorem eggplant_basket_weight :
  (total_weight - cucumber_baskets * cucumber_basket_weight) / eggplant_baskets =
  (1870 - 25 * 30) / 32 := by
  sorry

end NUMINAMATH_CALUDE_eggplant_basket_weight_l3528_352845


namespace NUMINAMATH_CALUDE_complex_magnitude_l3528_352836

theorem complex_magnitude (a : ℝ) (z : ℂ) : 
  z = a + Complex.I ∧ z^2 + z = 1 - 3*Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3528_352836


namespace NUMINAMATH_CALUDE_gracie_number_l3528_352817

/-- Represents the counting pattern for a student --/
def student_count (n : ℕ) : Set ℕ :=
  {m | m ≤ 2000 ∧ m ≠ 0 ∧ ∃ k, m = 5*k + 1 ∨ m = 5*k + 2 ∨ m = 5*k + 4 ∨ m = 5*k + 5}

/-- Represents the numbers skipped by a student --/
def student_skip (n : ℕ) : Set ℕ :=
  {m | m ≤ 2000 ∧ m ≠ 0 ∧ ∃ k, m = 5^n * (5*k - 2)}

/-- The set of numbers said by the first n students --/
def numbers_said (n : ℕ) : Set ℕ :=
  if n = 0 then ∅ else (student_count n) ∪ (numbers_said (n-1)) \ (student_skip n)

theorem gracie_number :
  ∃! x, x ∈ {m | 1 ≤ m ∧ m ≤ 2000} \ (numbers_said 7) ∧ x = 1623 :=
sorry

end NUMINAMATH_CALUDE_gracie_number_l3528_352817


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l3528_352883

-- Define the circles
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 1}

-- Define the centers of the circles
def center₁ : ℝ × ℝ := (0, 0)
def center₂ : ℝ × ℝ := (2, 0)

-- Define the radii of the circles
def radius₁ : ℝ := 1
def radius₂ : ℝ := 1

-- Theorem statement
theorem circles_externally_tangent :
  let d := Real.sqrt ((center₂.1 - center₁.1)^2 + (center₂.2 - center₁.2)^2)
  d = radius₁ + radius₂ := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l3528_352883


namespace NUMINAMATH_CALUDE_numerator_exceeds_denominator_l3528_352841

theorem numerator_exceeds_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 →
  (4 * x + 5 > 10 - 3 * x ↔ 5 / 7 < x ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_numerator_exceeds_denominator_l3528_352841


namespace NUMINAMATH_CALUDE_sports_enthusiasts_l3528_352897

theorem sports_enthusiasts (I A B : Finset ℕ) : 
  Finset.card I = 100 → 
  Finset.card A = 63 → 
  Finset.card B = 75 → 
  38 ≤ Finset.card (A ∩ B) ∧ Finset.card (A ∩ B) ≤ 63 := by
  sorry

end NUMINAMATH_CALUDE_sports_enthusiasts_l3528_352897


namespace NUMINAMATH_CALUDE_percentage_problem_l3528_352811

theorem percentage_problem (y : ℝ) (h1 : y > 0) (h2 : y * (y / 100) = 9) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3528_352811


namespace NUMINAMATH_CALUDE_cards_bought_equals_difference_l3528_352878

/-- The number of baseball cards Sam bought is equal to the difference between
    Mike's initial number of cards and his current number of cards. -/
theorem cards_bought_equals_difference (initial_cards current_cards cards_bought : ℕ) :
  initial_cards = 87 →
  current_cards = 74 →
  cards_bought = initial_cards - current_cards →
  cards_bought = 13 := by
  sorry

end NUMINAMATH_CALUDE_cards_bought_equals_difference_l3528_352878


namespace NUMINAMATH_CALUDE_hannah_stocking_stuffers_l3528_352812

/-- The number of candy canes per stocking -/
def candy_canes_per_stocking : ℕ := 4

/-- The number of beanie babies per stocking -/
def beanie_babies_per_stocking : ℕ := 2

/-- The number of books per stocking -/
def books_per_stocking : ℕ := 1

/-- The number of kids Hannah has -/
def number_of_kids : ℕ := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stocking_stuffers : ℕ := 
  (candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking) * number_of_kids

theorem hannah_stocking_stuffers : total_stocking_stuffers = 21 := by
  sorry

end NUMINAMATH_CALUDE_hannah_stocking_stuffers_l3528_352812


namespace NUMINAMATH_CALUDE_prime_divisor_of_binomial_coefficients_l3528_352838

theorem prime_divisor_of_binomial_coefficients (p : ℕ) (n : ℕ) (h_p : Prime p) (h_n : n > 1) :
  (∀ x : ℕ, 1 ≤ x ∧ x < n → p ∣ Nat.choose n x) ↔ ∃ a : ℕ, a > 0 ∧ n = p^a :=
sorry

end NUMINAMATH_CALUDE_prime_divisor_of_binomial_coefficients_l3528_352838


namespace NUMINAMATH_CALUDE_bottle_cap_boxes_l3528_352843

theorem bottle_cap_boxes (total_caps : ℕ) (caps_per_box : ℕ) (h1 : total_caps = 316) (h2 : caps_per_box = 4) :
  total_caps / caps_per_box = 79 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_boxes_l3528_352843


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l3528_352879

-- Define the circumference of each circle
def circle_circumference : ℝ := 48

-- Define the angle subtended by each arc (in degrees)
def arc_angle : ℝ := 90

-- Define the number of circles
def num_circles : ℕ := 3

-- Theorem statement
theorem shaded_region_perimeter :
  let arc_length := (arc_angle / 360) * circle_circumference
  (num_circles : ℝ) * arc_length = 36 := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l3528_352879


namespace NUMINAMATH_CALUDE_sphere_surface_area_given_cone_l3528_352824

/-- Given a cone and a sphere with equal volumes, where the radius of the base of the cone
    is twice the radius of the sphere, and the height of the cone is 1,
    prove that the surface area of the sphere is 4π. -/
theorem sphere_surface_area_given_cone (r : ℝ) :
  (4 / 3 * π * r^3 = 1 / 3 * π * (2*r)^2 * 1) →
  4 * π * r^2 = 4 * π := by
  sorry

#check sphere_surface_area_given_cone

end NUMINAMATH_CALUDE_sphere_surface_area_given_cone_l3528_352824


namespace NUMINAMATH_CALUDE_focus_of_our_parabola_l3528_352819

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola x^2 = 4y -/
def our_parabola : Parabola :=
  { equation := fun x y => x^2 = 4*y }

theorem focus_of_our_parabola :
  focus our_parabola = (0, 1) := by sorry

end NUMINAMATH_CALUDE_focus_of_our_parabola_l3528_352819


namespace NUMINAMATH_CALUDE_range_of_k_l3528_352858

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}
def B (k : ℝ) : Set ℝ := {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

-- State the theorem
theorem range_of_k (k : ℝ) : A ∩ B k = B k → -1 ≤ k ∧ k ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l3528_352858
