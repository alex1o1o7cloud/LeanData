import Mathlib

namespace min_team_size_proof_l2794_279460

def P₁ : ℝ := 0.3

def individual_prob : ℝ := 0.1

def P₂ (n : ℕ) : ℝ := 1 - (1 - individual_prob) ^ n

def min_team_size : ℕ := 4

theorem min_team_size_proof :
  ∀ n : ℕ, (P₂ n ≥ P₁) → n ≥ min_team_size :=
sorry

end min_team_size_proof_l2794_279460


namespace garden_shorter_side_l2794_279495

theorem garden_shorter_side (perimeter : ℝ) (area : ℝ) : perimeter = 60 ∧ area = 200 → ∃ x y : ℝ, x ≤ y ∧ 2*x + 2*y = perimeter ∧ x*y = area ∧ x = 10 := by
  sorry

end garden_shorter_side_l2794_279495


namespace june_birth_percentage_l2794_279424

theorem june_birth_percentage (total_scientists : ℕ) (june_born : ℕ) 
  (h1 : total_scientists = 200) (h2 : june_born = 18) :
  (june_born : ℚ) / total_scientists * 100 = 9 := by
  sorry

end june_birth_percentage_l2794_279424


namespace divisibility_of_m_l2794_279482

theorem divisibility_of_m (m : ℤ) : m = 76^2006 - 76 → 100 ∣ m := by
  sorry

end divisibility_of_m_l2794_279482


namespace nancy_pears_l2794_279438

theorem nancy_pears (total_pears alyssa_pears : ℕ) 
  (h1 : total_pears = 59)
  (h2 : alyssa_pears = 42) :
  total_pears - alyssa_pears = 17 := by
sorry

end nancy_pears_l2794_279438


namespace product_of_fractions_l2794_279493

theorem product_of_fractions : (1 / 3 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * (3 / 7 : ℚ) = 6 / 35 := by
  sorry

end product_of_fractions_l2794_279493


namespace plane_speed_calculation_l2794_279433

/-- Two planes traveling in opposite directions -/
structure TwoPlanes where
  speed_west : ℝ
  speed_east : ℝ
  time : ℝ
  total_distance : ℝ

/-- The theorem stating the conditions and the result to be proved -/
theorem plane_speed_calculation (planes : TwoPlanes) 
  (h1 : planes.speed_west = 275)
  (h2 : planes.time = 3.5)
  (h3 : planes.total_distance = 2100)
  : planes.speed_east = 325 := by
  sorry

#check plane_speed_calculation

end plane_speed_calculation_l2794_279433


namespace ellipse_major_axis_length_l2794_279444

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The major axis length of the ellipse is 6.4 --/
theorem ellipse_major_axis_length :
  let cylinder_radius : ℝ := 2
  let major_minor_ratio : ℝ := 1.6
  major_axis_length cylinder_radius major_minor_ratio = 6.4 := by
  sorry

#eval major_axis_length 2 1.6

end ellipse_major_axis_length_l2794_279444


namespace table_tennis_sequences_l2794_279409

/-- Represents a sequence of matches in the table tennis competition -/
def MatchSequence := List ℕ

/-- The number of players in each team -/
def teamSize : ℕ := 5

/-- Calculates the number of possible sequences for a given player finishing the competition -/
def sequencesForPlayer (player : ℕ) : ℕ := sorry

/-- Calculates the total number of possible sequences for one team winning -/
def totalSequencesOneTeam : ℕ :=
  (List.range teamSize).map sequencesForPlayer |>.sum

/-- The total number of possible sequences in the competition -/
def totalSequences : ℕ := 2 * totalSequencesOneTeam

theorem table_tennis_sequences :
  totalSequences = 252 := by sorry

end table_tennis_sequences_l2794_279409


namespace graduates_second_degree_l2794_279467

theorem graduates_second_degree (total : ℕ) (job : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 73 → job = 32 → both = 13 → neither = 9 → 
  ∃ (second_degree : ℕ), second_degree = 45 := by
sorry

end graduates_second_degree_l2794_279467


namespace units_digit_of_fraction_l2794_279486

theorem units_digit_of_fraction (n : ℕ) : n = 1994 → (5^n + 6^n) % 7 = 5 → (5^n + 6^n) % 10 = 1 → (5^n + 6^n) / 7 % 10 = 4 := by
  sorry

end units_digit_of_fraction_l2794_279486


namespace division_problem_l2794_279404

theorem division_problem (dividend quotient remainder : ℕ) (divisor : ℚ) : 
  dividend = 12 → quotient = 9 → remainder = 8 → 
  dividend = (divisor * quotient) + remainder → 
  divisor = 4/9 := by
sorry

end division_problem_l2794_279404


namespace bamboo_nine_sections_l2794_279425

/-- Given an arithmetic sequence of 9 terms, prove that if the sum of the first 4 terms is 3
    and the sum of the last 3 terms is 4, then the 5th term is 67/66 -/
theorem bamboo_nine_sections 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_sum_first_four : a 1 + a 2 + a 3 + a 4 = 3)
  (h_sum_last_three : a 7 + a 8 + a 9 = 4) :
  a 5 = 67 / 66 := by
sorry

end bamboo_nine_sections_l2794_279425


namespace complex_sum_theorem_l2794_279483

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 2 → 
  g = -a - c - e → 
  3 * ((a + b * I) + (c + d * I) + (e + f * I) + (g + h * I)) = 2 * I → 
  d + f + h = -4/3 := by
sorry

end complex_sum_theorem_l2794_279483


namespace no_positive_integer_solution_l2794_279426

theorem no_positive_integer_solution :
  ¬∃ (p q r : ℕ+), 
    (p^2 : ℚ) / q = 4 / 5 ∧
    (q : ℚ) / r^2 = 2 / 3 ∧
    (p : ℚ) / r^3 = 6 / 7 :=
by sorry

end no_positive_integer_solution_l2794_279426


namespace product_inequality_l2794_279472

theorem product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (2 + x) * (2 + y) * (2 + z) ≥ 27 := by
  sorry

end product_inequality_l2794_279472


namespace complex_modulus_l2794_279422

theorem complex_modulus (z : ℂ) (a : ℝ) : 
  z = a + Complex.I ∧ z + z = 1 - 3 * Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l2794_279422


namespace contractor_fine_proof_l2794_279497

/-- Calculates the fine per day of absence for a contractor --/
def calculate_fine_per_day (total_days : ℕ) (pay_per_day : ℚ) (total_payment : ℚ) (absent_days : ℕ) : ℚ :=
  let worked_days := total_days - absent_days
  let total_earned := pay_per_day * worked_days
  (total_earned - total_payment) / absent_days

/-- Proves that the fine per day of absence is 7.5 given the contract conditions --/
theorem contractor_fine_proof :
  calculate_fine_per_day 30 25 425 10 = 7.5 := by
  sorry


end contractor_fine_proof_l2794_279497


namespace quiz_mistakes_l2794_279454

theorem quiz_mistakes (total_items : ℕ) (score_percentage : ℚ) : 
  total_items = 25 → score_percentage = 80 / 100 → 
  total_items - (score_percentage * total_items).num = 5 := by
sorry

end quiz_mistakes_l2794_279454


namespace find_a_l2794_279447

def round_to_two_decimal_places (x : ℚ) : ℚ :=
  (⌊x * 100 + 0.5⌋ : ℚ) / 100

theorem find_a : ∃ (a : ℕ), round_to_two_decimal_places (1.322 - (a : ℚ) / 99) = 1.10 ∧ a = 22 := by
  sorry

end find_a_l2794_279447


namespace jack_shirts_per_kid_l2794_279455

theorem jack_shirts_per_kid (num_kids : ℕ) (buttons_per_shirt : ℕ) (total_buttons : ℕ) 
  (h1 : num_kids = 3)
  (h2 : buttons_per_shirt = 7)
  (h3 : total_buttons = 63) :
  total_buttons / buttons_per_shirt / num_kids = 3 := by
sorry

end jack_shirts_per_kid_l2794_279455


namespace third_group_frequency_count_l2794_279481

theorem third_group_frequency_count :
  ∀ (n₁ n₂ n₃ n₄ n₅ : ℕ),
  n₁ + n₂ + n₃ = 160 →
  n₃ + n₄ + n₅ = 260 →
  (n₃ : ℝ) / (n₁ + n₂ + n₃ + n₄ + n₅ : ℝ) = 0.20 →
  n₃ = 70 :=
by sorry

end third_group_frequency_count_l2794_279481


namespace expression_factorization_l2794_279445

theorem expression_factorization (a b c : ℝ) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end expression_factorization_l2794_279445


namespace quadratic_root_range_l2794_279437

theorem quadratic_root_range (m : ℝ) : 
  (∃ (α : ℂ), (α.re = 0 ∧ α.im ≠ 0) ∧ 
    (α ^ 2 - (2 * m - 1) * α + m ^ 2 + 1 = 0) ∧
    (Complex.abs α ≤ 2)) →
  (m > -3/4 ∧ m ≤ Real.sqrt 3) :=
by sorry

end quadratic_root_range_l2794_279437


namespace segment_length_l2794_279484

/-- Given three points on a line, prove that the length of AC is either 7 or 1 -/
theorem segment_length (A B C : ℝ) : 
  (B - A = 4) → (C - B = 3) → (C - A = 7 ∨ C - A = 1) := by sorry

end segment_length_l2794_279484


namespace total_fish_caught_l2794_279435

def fishing_problem (leo_fish agrey_fish total_fish : ℕ) : Prop :=
  (leo_fish = 40) ∧
  (agrey_fish = leo_fish + 20) ∧
  (total_fish = leo_fish + agrey_fish)

theorem total_fish_caught : ∃ (leo_fish agrey_fish total_fish : ℕ),
  fishing_problem leo_fish agrey_fish total_fish ∧ total_fish = 100 :=
by
  sorry

end total_fish_caught_l2794_279435


namespace quadratic_factor_problem_l2794_279434

theorem quadratic_factor_problem (a b : ℝ) :
  (∀ x, x^2 + 6*x + a = (x + 5)*(x + b)) → b = 1 ∧ a = 5 := by
  sorry

end quadratic_factor_problem_l2794_279434


namespace unitPrice_is_constant_l2794_279480

/-- Represents the data from a fuel dispenser --/
structure FuelDispenser :=
  (amount : ℝ)
  (unitPrice : ℝ)
  (unitPricePerYuanPerLiter : ℝ)

/-- The fuel dispenser data from the problem --/
def fuelData : FuelDispenser :=
  { amount := 116.64,
    unitPrice := 18,
    unitPricePerYuanPerLiter := 6.48 }

/-- Predicate to check if a value is constant in the fuel dispenser context --/
def isConstant (f : FuelDispenser → ℝ) : Prop :=
  ∀ (d1 d2 : FuelDispenser), d1.unitPrice = d2.unitPrice → f d1 = f d2

/-- Theorem stating that the unit price is constant --/
theorem unitPrice_is_constant :
  isConstant (λ d : FuelDispenser => d.unitPrice) :=
sorry

end unitPrice_is_constant_l2794_279480


namespace cuboidal_box_volume_l2794_279403

/-- Represents a cuboidal box with given face areas -/
structure CuboidalBox where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ

/-- Calculates the volume of a cuboidal box given its face areas -/
def volume (box : CuboidalBox) : ℝ :=
  sorry

/-- Theorem stating that a cuboidal box with face areas 120, 72, and 60 has volume 720 -/
theorem cuboidal_box_volume :
  ∀ (box : CuboidalBox),
    box.area1 = 120 ∧ box.area2 = 72 ∧ box.area3 = 60 →
    volume box = 720 :=
by sorry

end cuboidal_box_volume_l2794_279403


namespace sum_of_integers_l2794_279443

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 32) : 
  x + y = 2 * Real.sqrt 41 := by
  sorry

end sum_of_integers_l2794_279443


namespace cleaning_time_theorem_l2794_279491

/-- Represents the grove of trees -/
structure Grove :=
  (rows : ℕ)
  (columns : ℕ)

/-- Calculates the time to clean each tree without help -/
def time_per_tree_without_help (g : Grove) (total_time_with_help : ℕ) : ℚ :=
  let total_trees := g.rows * g.columns
  let time_per_tree_with_help := total_time_with_help / total_trees
  2 * time_per_tree_with_help

theorem cleaning_time_theorem (g : Grove) (h : g.rows = 4 ∧ g.columns = 5) :
  time_per_tree_without_help g 60 = 6 := by
  sorry

end cleaning_time_theorem_l2794_279491


namespace horner_method_v2_l2794_279431

def f (x : ℝ) : ℝ := 4*x^4 + 3*x^3 - 6*x^2 + x - 1

def horner_v0 : ℝ := 4

def horner_v1 (x : ℝ) : ℝ := horner_v0 * x + 3

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x - 6

theorem horner_method_v2 : horner_v2 (-1) = -5 := by
  sorry

end horner_method_v2_l2794_279431


namespace dice_probability_l2794_279490

def num_dice : ℕ := 6
def sides_per_die : ℕ := 8

def probability_at_least_two_same : ℚ := 3781 / 4096

theorem dice_probability :
  probability_at_least_two_same = 1 - (sides_per_die.factorial / (sides_per_die - num_dice).factorial) / sides_per_die ^ num_dice :=
by sorry

end dice_probability_l2794_279490


namespace period_2_gym_class_size_l2794_279478

theorem period_2_gym_class_size :
  ∀ (period_2_size : ℕ),
  (2 * period_2_size - 5 = 11) →
  period_2_size = 8 := by
sorry

end period_2_gym_class_size_l2794_279478


namespace ratio_problem_l2794_279430

theorem ratio_problem (w x y z : ℝ) 
  (h1 : w / x = 1 / 3) 
  (h2 : w / y = 2 / 3) 
  (h3 : w / z = 3 / 5) 
  (hw : w ≠ 0) : 
  (x + y) / z = 27 / 10 := by
sorry

end ratio_problem_l2794_279430


namespace greatest_number_l2794_279451

theorem greatest_number (p q r s t : ℝ) 
  (h1 : r < s) 
  (h2 : t > q) 
  (h3 : q > p) 
  (h4 : t < r) : 
  s = max p (max q (max r (max s t))) := by
sorry

end greatest_number_l2794_279451


namespace sum_of_solutions_eq_six_l2794_279496

theorem sum_of_solutions_eq_six :
  ∃ (M₁ M₂ : ℝ), (M₁ * (M₁ - 6) = -5) ∧ (M₂ * (M₂ - 6) = -5) ∧ (M₁ + M₂ = 6) :=
by sorry

end sum_of_solutions_eq_six_l2794_279496


namespace constant_function_l2794_279487

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x y : ℝ, f (x + y) = f x * f (a - y) + f y * f (a - x)

theorem constant_function (f : ℝ → ℝ) (h1 : f 0 = 1/2) (h2 : FunctionalEq f) :
    ∀ x : ℝ, f x = 1/2 := by
  sorry

end constant_function_l2794_279487


namespace perpendicular_lines_minimum_value_l2794_279421

theorem perpendicular_lines_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (-(1 : ℝ) / (a - 4)) * (-2 * b) = 1) : 
  ∃ (x : ℝ), ∀ (y : ℝ), (a + 2) / (a + 1) + 1 / (2 * b) ≥ x ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
  (-(1 : ℝ) / (a₀ - 4)) * (-2 * b₀) = 1 ∧ 
  (a₀ + 2) / (a₀ + 1) + 1 / (2 * b₀) = x ∧ 
  x = 9 / 5 :=
sorry

end perpendicular_lines_minimum_value_l2794_279421


namespace doughnuts_given_away_is_30_l2794_279406

/-- The number of doughnuts in each box -/
def doughnuts_per_box : ℕ := 10

/-- The total number of doughnuts made for the day -/
def total_doughnuts : ℕ := 300

/-- The number of boxes sold -/
def boxes_sold : ℕ := 27

/-- The number of doughnuts given away at the end of the day -/
def doughnuts_given_away : ℕ := total_doughnuts - (boxes_sold * doughnuts_per_box)

theorem doughnuts_given_away_is_30 : doughnuts_given_away = 30 := by
  sorry

end doughnuts_given_away_is_30_l2794_279406


namespace ricciana_long_jump_l2794_279407

/-- Ricciana's long jump problem -/
theorem ricciana_long_jump :
  ∀ (ricciana_run margarita_run ricciana_jump margarita_jump : ℕ),
  ricciana_run = 20 →
  margarita_run = 18 →
  margarita_jump = 2 * ricciana_jump - 1 →
  margarita_run + margarita_jump = ricciana_run + ricciana_jump + 1 →
  ricciana_jump = 22 := by
sorry

end ricciana_long_jump_l2794_279407


namespace simplify_fraction_l2794_279473

theorem simplify_fraction (a : ℝ) (h : a = 2) : (15 * a^4) / (75 * a^3) = 2 / 5 := by
  sorry

end simplify_fraction_l2794_279473


namespace mean_visits_between_200_and_300_l2794_279428

def website_visits : List Nat := [300, 400, 300, 200, 200]

theorem mean_visits_between_200_and_300 :
  let mean := (website_visits.sum : ℚ) / website_visits.length
  200 < mean ∧ mean < 300 := by
  sorry

end mean_visits_between_200_and_300_l2794_279428


namespace katie_cole_miles_ratio_l2794_279464

theorem katie_cole_miles_ratio :
  ∀ (miles_xavier miles_katie miles_cole : ℕ),
    miles_xavier = 3 * miles_katie →
    miles_xavier = 84 →
    miles_cole = 7 →
    miles_katie / miles_cole = 4 := by
  sorry

end katie_cole_miles_ratio_l2794_279464


namespace complex_equation_real_solution_l2794_279474

theorem complex_equation_real_solution :
  ∀ x : ℝ, (x^2 + Complex.I * x + 6 : ℂ) = (2 * Complex.I + 5 * x : ℂ) → x = 2 := by
  sorry

end complex_equation_real_solution_l2794_279474


namespace max_value_quadratic_inequality_l2794_279453

/-- Given a quadratic inequality ax² + bx + c > 0 with solution set {x | -1 < x < 3},
    the maximum value of b - c + 1/a is -2 -/
theorem max_value_quadratic_inequality (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 3) →
  (∃ m : ℝ, ∀ a' b' c' : ℝ, 
    (∀ x, a'*x^2 + b'*x + c' > 0 ↔ -1 < x ∧ x < 3) →
    b' - c' + 1/a' ≤ m) ∧
  (∃ a₀ b₀ c₀ : ℝ, 
    (∀ x, a₀*x^2 + b₀*x + c₀ > 0 ↔ -1 < x ∧ x < 3) ∧
    b₀ - c₀ + 1/a₀ = -2) :=
by sorry

end max_value_quadratic_inequality_l2794_279453


namespace quadrilateral_inequality_l2794_279477

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (a b c d e f : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (c_pos : c > 0)
  (d_pos : d > 0)
  (e_pos : e > 0)
  (f_pos : f > 0)

-- Define what it means for a quadrilateral to be cyclic
def is_cyclic (q : Quadrilateral) : Prop :=
  q.e^2 + q.f^2 = q.b^2 + q.d^2 + 2*q.a*q.c

-- State the theorem
theorem quadrilateral_inequality (q : Quadrilateral) :
  q.e^2 + q.f^2 ≤ q.b^2 + q.d^2 + 2*q.a*q.c ∧
  (q.e^2 + q.f^2 = q.b^2 + q.d^2 + 2*q.a*q.c ↔ is_cyclic q) :=
sorry

end quadrilateral_inequality_l2794_279477


namespace plates_needed_is_38_l2794_279440

/-- The number of plates needed for a week given the eating habits of Matt's family -/
def plates_needed : ℕ :=
  let days_with_son := 3
  let days_with_parents := 7 - days_with_son
  let plates_per_person_with_son := 1
  let plates_per_person_with_parents := 2
  let people_with_son := 2
  let people_with_parents := 4
  
  (days_with_son * people_with_son * plates_per_person_with_son) +
  (days_with_parents * people_with_parents * plates_per_person_with_parents)

theorem plates_needed_is_38 : plates_needed = 38 := by
  sorry

end plates_needed_is_38_l2794_279440


namespace cube_with_specific_digits_l2794_279408

theorem cube_with_specific_digits : ∃! n : ℕ, 
  (n^3 ≥ 30000 ∧ n^3 < 40000) ∧ 
  (n^3 % 10 = 4) ∧
  (n = 34) := by
  sorry

end cube_with_specific_digits_l2794_279408


namespace remainder_sum_mod_13_l2794_279492

theorem remainder_sum_mod_13 (a b c d : ℤ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
  sorry

end remainder_sum_mod_13_l2794_279492


namespace student_average_problem_l2794_279441

theorem student_average_problem :
  let total_students : ℕ := 25
  let group_a_students : ℕ := 15
  let group_b_students : ℕ := 10
  let group_b_average : ℚ := 90
  let total_average : ℚ := 84
  let group_a_average : ℚ := (total_students * total_average - group_b_students * group_b_average) / group_a_students
  group_a_average = 80 := by sorry

end student_average_problem_l2794_279441


namespace train_crossing_bridge_time_l2794_279450

/-- Represents the problem of a train crossing a bridge -/
def TrainCrossingBridge (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : Prop :=
  let total_distance : ℝ := train_length + bridge_length
  let train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
  let crossing_time : ℝ := total_distance / train_speed_mps
  crossing_time = 72.5

/-- Theorem stating that a train 250 meters long, running at 72 kmph, 
    takes 72.5 seconds to cross a bridge 1,200 meters in length -/
theorem train_crossing_bridge_time :
  TrainCrossingBridge 250 72 1200 := by
  sorry

#check train_crossing_bridge_time

end train_crossing_bridge_time_l2794_279450


namespace curve_C_properties_l2794_279456

-- Define the curve C
def C (x : ℝ) : ℝ := x^3 + 5*x^2 + 3*x

-- State the theorem
theorem curve_C_properties :
  -- The derivative of C is 3x² + 10x + 3
  (∀ x : ℝ, deriv C x = 3*x^2 + 10*x + 3) ∧
  -- The equation of the tangent line to C at x = 1 is 16x - y - 7 = 0
  (∀ y : ℝ, (C 1 = y) → (16 - y - 7 = 0 ↔ ∃ x : ℝ, y = 16*(x - 1) + C 1)) :=
by
  sorry

end curve_C_properties_l2794_279456


namespace maria_towels_l2794_279417

/-- The number of towels Maria ended up with after shopping and giving some to her mother. -/
def towels_maria_kept (green_towels white_towels towels_given : ℕ) : ℕ :=
  green_towels + white_towels - towels_given

/-- Theorem stating that Maria ended up with 22 towels -/
theorem maria_towels :
  towels_maria_kept 35 21 34 = 22 := by
  sorry

end maria_towels_l2794_279417


namespace sum_of_solutions_quadratic_l2794_279458

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ r s : ℝ, (72 - 18*x - x^2 = 0 ↔ (x = r ∨ x = s)) ∧ r + s = -18) :=
by sorry

end sum_of_solutions_quadratic_l2794_279458


namespace amelia_win_probability_l2794_279416

/-- The probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 3/7

/-- The probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 1/4

/-- The probability of Amelia winning the game -/
def p_amelia_wins : ℚ := 9/14

/-- The game described in the problem -/
def coin_game (p_a p_b : ℚ) : ℚ :=
  let p_amelia_first := p_a * (1 - p_b)
  let p_blaine_first := (1 - p_a) * p_b
  let p_both_tails := (1 - p_a) * (1 - p_b)
  let p_amelia_alternate := p_both_tails * (p_a / (1 - (1 - p_a) * (1 - p_b)))
  p_amelia_first + p_amelia_alternate

theorem amelia_win_probability :
  coin_game p_amelia p_blaine = p_amelia_wins :=
sorry

end amelia_win_probability_l2794_279416


namespace probability_larger_than_40_l2794_279442

def digits : Finset Nat := {1, 2, 3, 4, 5}

def is_valid_selection (a b : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ a ≠ b

def is_larger_than_40 (a b : Nat) : Prop :=
  is_valid_selection a b ∧ 10 * a + b > 40

def total_selections : Nat :=
  digits.card * (digits.card - 1)

def favorable_selections : Nat :=
  (digits.filter (λ x => x ≥ 4)).card * (digits.card - 1)

theorem probability_larger_than_40 :
  (favorable_selections : ℚ) / total_selections = 2 / 5 := by
  sorry

end probability_larger_than_40_l2794_279442


namespace trig_system_solution_l2794_279405

theorem trig_system_solution (x y : ℝ) (m n : ℤ) :
  (Real.sin x * Real.cos y = 0.25) ∧ (Real.sin y * Real.cos x = 0.75) →
  ((x = Real.pi / 6 + Real.pi * (m - n : ℝ) ∧ y = Real.pi / 3 + Real.pi * (m + n : ℝ)) ∨
   (x = -Real.pi / 6 + Real.pi * (m - n : ℝ) ∧ y = 2 * Real.pi / 3 + Real.pi * (m + n : ℝ))) :=
by sorry

end trig_system_solution_l2794_279405


namespace no_primes_divisible_by_46_l2794_279499

theorem no_primes_divisible_by_46 : ∀ p : ℕ, Nat.Prime p → ¬(46 ∣ p) := by
  sorry

end no_primes_divisible_by_46_l2794_279499


namespace shampoo_bottles_l2794_279465

theorem shampoo_bottles (medium_capacity : ℕ) (jumbo_capacity : ℕ) (unusable_space : ℕ) :
  medium_capacity = 45 →
  jumbo_capacity = 720 →
  unusable_space = 20 →
  (Nat.ceil ((jumbo_capacity - unusable_space : ℚ) / medium_capacity) : ℕ) = 16 := by
  sorry

end shampoo_bottles_l2794_279465


namespace refurbished_to_new_tshirt_ratio_l2794_279461

/-- The price of a new T-shirt in dollars -/
def new_tshirt_price : ℚ := 5

/-- The price of a pair of pants in dollars -/
def pants_price : ℚ := 4

/-- The price of a skirt in dollars -/
def skirt_price : ℚ := 6

/-- The total income from selling 2 new T-shirts, 1 pair of pants, 4 skirts, and 6 refurbished T-shirts -/
def total_income : ℚ := 53

/-- The number of new T-shirts sold -/
def new_tshirts_sold : ℕ := 2

/-- The number of pants sold -/
def pants_sold : ℕ := 1

/-- The number of skirts sold -/
def skirts_sold : ℕ := 4

/-- The number of refurbished T-shirts sold -/
def refurbished_tshirts_sold : ℕ := 6

/-- Theorem stating that the ratio of the price of a refurbished T-shirt to the price of a new T-shirt is 1/2 -/
theorem refurbished_to_new_tshirt_ratio :
  (total_income - (new_tshirt_price * new_tshirts_sold + pants_price * pants_sold + skirt_price * skirts_sold)) / refurbished_tshirts_sold / new_tshirt_price = 1 / 2 := by
  sorry

end refurbished_to_new_tshirt_ratio_l2794_279461


namespace triangle_PQR_area_l2794_279457

/-- The area of a triangle with vertices P(-4, 2), Q(6, 2), and R(2, -5) is 35 square units. -/
theorem triangle_PQR_area : 
  let P : ℝ × ℝ := (-4, 2)
  let Q : ℝ × ℝ := (6, 2)
  let R : ℝ × ℝ := (2, -5)
  let triangle_area := abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)
  triangle_area = 35 := by sorry

end triangle_PQR_area_l2794_279457


namespace consecutive_odd_integers_l2794_279432

theorem consecutive_odd_integers (x y z : ℤ) : 
  (∃ k : ℤ, x = 2*k + 1) →  -- x is odd
  y = x + 2 →               -- y is the next consecutive odd integer
  z = y + 2 →               -- z is the next consecutive odd integer after y
  y + z = x + 17 →          -- sum of last two is 17 more than the first
  x = 11 := by              -- the first integer is 11
sorry

end consecutive_odd_integers_l2794_279432


namespace jay_change_is_twenty_l2794_279479

-- Define the prices of items and the payment amount
def book_price : ℕ := 25
def pen_price : ℕ := 4
def ruler_price : ℕ := 1
def payment : ℕ := 50

-- Define the change received
def change : ℕ := payment - (book_price + pen_price + ruler_price)

-- Theorem statement
theorem jay_change_is_twenty : change = 20 := by
  sorry

end jay_change_is_twenty_l2794_279479


namespace triangular_cross_section_solids_l2794_279462

-- Define the set of all possible solids
inductive Solid
  | Prism
  | Pyramid
  | Frustum
  | Cylinder
  | Cone
  | TruncatedCone
  | Sphere

-- Define a predicate for solids that can have a triangular cross-section
def hasTriangularCrossSection (s : Solid) : Prop :=
  match s with
  | Solid.Prism => true
  | Solid.Pyramid => true
  | Solid.Frustum => true
  | Solid.Cone => true
  | _ => false

-- Define the set of solids that can have a triangular cross-section
def solidsWithTriangularCrossSection : Set Solid :=
  {s : Solid | hasTriangularCrossSection s}

-- Theorem statement
theorem triangular_cross_section_solids :
  solidsWithTriangularCrossSection = {Solid.Prism, Solid.Pyramid, Solid.Frustum, Solid.Cone} :=
by sorry

end triangular_cross_section_solids_l2794_279462


namespace angle_with_special_supplement_and_complement_l2794_279476

theorem angle_with_special_supplement_and_complement :
  ∀ x : ℝ,
  (0 < x) →
  (x < 180) →
  (180 - x = 4 * (90 - x)) →
  x = 60 := by
  sorry

end angle_with_special_supplement_and_complement_l2794_279476


namespace town_average_age_l2794_279475

theorem town_average_age (k : ℕ) (h_k : k > 0) : 
  let num_children := 3 * k
  let num_adults := 2 * k
  let avg_age_children := 10
  let avg_age_adults := 40
  let total_population := num_children + num_adults
  let total_age := num_children * avg_age_children + num_adults * avg_age_adults
  (total_age : ℚ) / total_population = 22 :=
by sorry

end town_average_age_l2794_279475


namespace library_visitors_average_l2794_279471

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (h1 : sunday_visitors = 510) (h2 : other_day_visitors = 240) 
  (h3 : days_in_month = 30) : 
  (5 * sunday_visitors + 25 * other_day_visitors) / days_in_month = 285 :=
by
  sorry

end library_visitors_average_l2794_279471


namespace valid_fraction_pairs_l2794_279468

def is_valid_pair (x y : ℚ) : Prop :=
  ∃ (A B : ℕ+) (r : ℚ),
    x = (A : ℚ) * (1/10 + 1/70) ∧
    y = (B : ℚ) * (1/10 + 1/70) ∧
    x + y = 8 ∧
    r > 1 ∧
    ∃ (C D : ℕ), C > 1 ∧ D > 1 ∧ x = C * r ∧ y = D * r

theorem valid_fraction_pairs :
  (is_valid_pair (16/7) (40/7) ∧
   is_valid_pair (24/7) (32/7) ∧
   is_valid_pair (16/5) (24/5) ∧
   is_valid_pair 4 4) ∧
  ∀ x y, is_valid_pair x y →
    ((x = 16/7 ∧ y = 40/7) ∨
     (x = 24/7 ∧ y = 32/7) ∨
     (x = 16/5 ∧ y = 24/5) ∨
     (x = 4 ∧ y = 4) ∨
     (y = 16/7 ∧ x = 40/7) ∨
     (y = 24/7 ∧ x = 32/7) ∨
     (y = 16/5 ∧ x = 24/5)) :=
by sorry


end valid_fraction_pairs_l2794_279468


namespace polygon_interior_angle_sum_l2794_279400

theorem polygon_interior_angle_sum (n : ℕ) (h : n > 2) :
  (n * 40 = 360) →
  (n - 2) * 180 = 1260 := by
  sorry

end polygon_interior_angle_sum_l2794_279400


namespace xyz_product_l2794_279420

/-- Given complex numbers x, y, and z satisfying the specified equations,
    prove that their product equals 260/3. -/
theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -25) :
  x * y * z = 260 / 3 := by
  sorry

end xyz_product_l2794_279420


namespace abc_modulo_seven_l2794_279411

theorem abc_modulo_seven (a b c : ℕ) 
  (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : (a + 2*b + 3*c) % 7 = 0)
  (h2 : (2*a + 3*b + c) % 7 = 4)
  (h3 : (3*a + b + 2*c) % 7 = 4) :
  (a * b * c) % 7 = 6 := by
sorry

end abc_modulo_seven_l2794_279411


namespace ascending_order_abab_l2794_279449

theorem ascending_order_abab (a b : ℝ) (ha : a > 0) (hb : b < 0) (hab : a + b > 0) :
  -a < b ∧ b < -b ∧ -b < a := by sorry

end ascending_order_abab_l2794_279449


namespace shooter_conditional_probability_l2794_279401

/-- Given a shooter with probabilities of hitting a target, prove the conditional probability of hitting the target in a subsequent shot. -/
theorem shooter_conditional_probability
  (p_single : ℝ)
  (p_twice : ℝ)
  (h_single : p_single = 0.7)
  (h_twice : p_twice = 0.4) :
  p_twice / p_single = 4 / 7 := by
  sorry

end shooter_conditional_probability_l2794_279401


namespace gumballs_last_42_days_l2794_279423

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_pair : ℕ := 9

/-- The number of pairs of earrings Kim brings on day 1 -/
def day1_pairs : ℕ := 3

/-- The number of pairs of earrings Kim brings on day 2 -/
def day2_pairs : ℕ := 2 * day1_pairs

/-- The number of pairs of earrings Kim brings on day 3 -/
def day3_pairs : ℕ := day2_pairs - 1

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The total number of gumballs Kim receives -/
def total_gumballs : ℕ := gumballs_per_pair * (day1_pairs + day2_pairs + day3_pairs)

/-- The number of days the gumballs will last -/
def days_gumballs_last : ℕ := total_gumballs / gumballs_eaten_per_day

theorem gumballs_last_42_days : days_gumballs_last = 42 := by
  sorry

end gumballs_last_42_days_l2794_279423


namespace polygon_interior_exterior_angles_equal_l2794_279429

theorem polygon_interior_exterior_angles_equal (n : ℕ) : 
  n ≥ 3 → (n - 2) * 180 = 360 → n = 4 := by sorry

end polygon_interior_exterior_angles_equal_l2794_279429


namespace smallest_number_l2794_279413

theorem smallest_number : ∀ (a b c d : ℝ), 
  a = -2 ∧ b = (1 : ℝ) / 2 ∧ c = 0 ∧ d = -Real.sqrt 2 →
  a < b ∧ a < c ∧ a < d :=
by sorry

end smallest_number_l2794_279413


namespace initial_marbles_l2794_279485

theorem initial_marbles (initial remaining given : ℕ) : 
  remaining = initial - given → 
  given = 8 → 
  remaining = 79 → 
  initial = 87 := by sorry

end initial_marbles_l2794_279485


namespace cone_height_equals_radius_l2794_279494

/-- The height of a cone formed by rolling a semicircular sheet of iron -/
def coneHeight (R : ℝ) : ℝ := R

/-- Theorem stating that the height of the cone is equal to the radius of the semicircular sheet -/
theorem cone_height_equals_radius (R : ℝ) (h : R > 0) : 
  coneHeight R = R := by sorry

end cone_height_equals_radius_l2794_279494


namespace solve_equation_l2794_279452

theorem solve_equation : ∃ y : ℝ, 3 * y - 6 = |-20 + 5| ∧ y = 7 := by
  sorry

end solve_equation_l2794_279452


namespace gmat_question_percentages_l2794_279463

theorem gmat_question_percentages :
  ∀ (first_correct second_correct both_correct neither_correct : ℝ),
    first_correct = 85 →
    neither_correct = 5 →
    both_correct = 55 →
    second_correct = 100 - neither_correct - (first_correct - both_correct) →
    second_correct = 65 := by
  sorry

end gmat_question_percentages_l2794_279463


namespace cost_price_75_equals_selling_price_40_implies_87_5_percent_gain_l2794_279427

/-- Calculates the gain percent given the ratio of cost price to selling price -/
def gainPercent (costPriceRatio sellingPriceRatio : ℕ) : ℚ :=
  ((sellingPriceRatio : ℚ) / (costPriceRatio : ℚ) - 1) * 100

/-- Theorem stating that if the cost price of 75 articles equals the selling price of 40 articles, 
    then the gain percent is 87.5% -/
theorem cost_price_75_equals_selling_price_40_implies_87_5_percent_gain :
  gainPercent 75 40 = 87.5 := by sorry

end cost_price_75_equals_selling_price_40_implies_87_5_percent_gain_l2794_279427


namespace min_races_for_fifty_horses_l2794_279418

/-- Represents the minimum number of races needed to find the top k fastest horses
    from a total of n horses, racing at most m horses at a time. -/
def min_races (n m k : ℕ) : ℕ :=
  sorry

/-- The theorem stating that for 50 horses, racing 3 at a time,
    19 races are needed to find the top 5 fastest horses. -/
theorem min_races_for_fifty_horses :
  min_races 50 3 5 = 19 := by sorry

end min_races_for_fifty_horses_l2794_279418


namespace count_D_eq_3_is_18_l2794_279466

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D_eq_3 : ℕ := sorry

theorem count_D_eq_3_is_18 : count_D_eq_3 = 18 := by sorry

end count_D_eq_3_is_18_l2794_279466


namespace horizontal_shift_right_l2794_279439

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the horizontal shift
def horizontalShift (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  fun x ↦ f (x - a)

-- Theorem statement
theorem horizontal_shift_right (a : ℝ) :
  ∀ x : ℝ, (horizontalShift f a) x = f (x - a) :=
by
  sorry

-- Note: This theorem states that for all real x,
-- the horizontally shifted function is equal to f(x - a),
-- which is equivalent to shifting the graph of f(x) right by a units.

end horizontal_shift_right_l2794_279439


namespace even_function_sum_ab_eq_two_l2794_279402

/-- A function f is even on an interval if f(-x) = f(x) for all x in the interval -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x, x ∈ Set.Icc a b → f (-x) = f x

theorem even_function_sum_ab_eq_two (a b : ℝ) :
  let f := fun x => a * x^2 + (b - 1) * x + 3 * a
  let domain := Set.Icc (a - 3) (2 * a)
  IsEvenOn f (a - 3) (2 * a) → a + b = 2 := by
  sorry

end even_function_sum_ab_eq_two_l2794_279402


namespace lcm_equality_and_inequality_l2794_279446

theorem lcm_equality_and_inequality (a b c : ℕ) : 
  (Nat.lcm a (a + 5) = Nat.lcm b (b + 5) → a = b) ∧
  (Nat.lcm a b ≠ Nat.lcm (a + c) (b + c)) := by
  sorry

end lcm_equality_and_inequality_l2794_279446


namespace complete_square_d_value_l2794_279419

/-- Given a quadratic equation x^2 - 6x + 5 = 0, prove that when converted to the form (x + c)^2 = d, the value of d is 4 -/
theorem complete_square_d_value (x : ℝ) : 
  (x^2 - 6*x + 5 = 0) → 
  (∃ c d : ℝ, (x + c)^2 = d ∧ x^2 - 6*x + 5 = 0) →
  (∃ c : ℝ, (x + c)^2 = 4 ∧ x^2 - 6*x + 5 = 0) :=
by sorry


end complete_square_d_value_l2794_279419


namespace factorial_equation_solution_l2794_279489

theorem factorial_equation_solution :
  ∃! (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  (5 * 4 * 3 * 2 * 1)^8 + (5 * 4 * 3 * 2 * 1)^7 = 4000000000000000 + a * 100000000000000 + 356000000000000 + 400000000000 + 80000000000 + b * 10000000000 + 80000000000 ∧
  a = 3 ∧ b = 6 := by
sorry

end factorial_equation_solution_l2794_279489


namespace extra_charge_per_wand_l2794_279488

def total_wands_bought : ℕ := 3
def cost_per_wand : ℚ := 60
def wands_sold : ℕ := 2
def total_collected : ℚ := 130

theorem extra_charge_per_wand :
  (total_collected / wands_sold) - cost_per_wand = 5 :=
by sorry

end extra_charge_per_wand_l2794_279488


namespace vector_coordinates_proof_l2794_279414

theorem vector_coordinates_proof :
  ∀ (a b : ℝ × ℝ),
    (‖a‖ = 3) →
    (b = (1, 2)) →
    (a.1 * b.1 + a.2 * b.2 = 0) →
    ((a = (-6 * Real.sqrt 5 / 5, 3 * Real.sqrt 5 / 5)) ∨
     (a = (6 * Real.sqrt 5 / 5, -3 * Real.sqrt 5 / 5))) := by
  sorry

end vector_coordinates_proof_l2794_279414


namespace appliance_savings_l2794_279412

def in_store_price : ℚ := 104.50
def tv_payment : ℚ := 24.80
def tv_shipping : ℚ := 10.80
def in_store_discount : ℚ := 5

theorem appliance_savings : 
  (4 * tv_payment + tv_shipping - (in_store_price - in_store_discount)) * 100 = 1050 := by
  sorry

end appliance_savings_l2794_279412


namespace factorization_a_squared_minus_four_a_l2794_279469

theorem factorization_a_squared_minus_four_a (a : ℝ) : a^2 - 4*a = a*(a - 4) := by
  sorry

end factorization_a_squared_minus_four_a_l2794_279469


namespace subtract_like_terms_l2794_279415

theorem subtract_like_terms (a : ℝ) : 4 * a - 3 * a = a := by
  sorry

end subtract_like_terms_l2794_279415


namespace function_value_2008_l2794_279470

theorem function_value_2008 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (4 - x)) 
  (h2 : ∀ x, f (2 - x) + f (x - 2) = 0) : 
  f 2008 = 0 := by
  sorry

end function_value_2008_l2794_279470


namespace sqrt_equation_solutions_l2794_279498

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 8) ^ x) + Real.sqrt ((3 - Real.sqrt 8) ^ x) = 6) ↔ (x = 2 ∨ x = -2) :=
by sorry

end sqrt_equation_solutions_l2794_279498


namespace median_salary_is_28000_l2794_279459

/-- Represents a position in the company with its title, number of employees, and salary -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company -/
def companyPositions : List Position := [
  { title := "CEO", count := 1, salary := 150000 },
  { title := "Senior Vice-President", count := 4, salary := 105000 },
  { title := "Manager", count := 15, salary := 80000 },
  { title := "Team Leader", count := 8, salary := 60000 },
  { title := "Office Assistant", count := 39, salary := 28000 }
]

/-- The total number of employees in the company -/
def totalEmployees : Nat := 67

/-- Calculates the median salary of the company -/
def medianSalary (positions : List Position) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary of the company is $28,000 -/
theorem median_salary_is_28000 :
  medianSalary companyPositions totalEmployees = 28000 := by
  sorry

end median_salary_is_28000_l2794_279459


namespace cereal_cost_l2794_279448

/-- Represents the cost of cereal boxes for a year -/
def cereal_problem (boxes_per_week : ℕ) (weeks_per_year : ℕ) (total_cost : ℕ) : Prop :=
  let total_boxes := boxes_per_week * weeks_per_year
  total_cost / total_boxes = 3

/-- Proves that each box of cereal costs $3 given the problem conditions -/
theorem cereal_cost : cereal_problem 2 52 312 := by
  sorry

end cereal_cost_l2794_279448


namespace min_a_value_l2794_279436

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

noncomputable def g (a e : ℝ) (x : ℝ) : ℝ := -e * x^2 + a * x

theorem min_a_value (e : ℝ) (he : e = Real.exp 1) :
  (∀ x₁ : ℝ, ∃ x₂ ∈ Set.Icc (1/3 : ℝ) 2, f x₁ ≤ g 2 e x₂) ∧
  (∀ ε > 0, ∃ x₁ : ℝ, ∀ x₂ ∈ Set.Icc (1/3 : ℝ) 2, f x₁ > g (2 - ε) e x₂) :=
sorry

end min_a_value_l2794_279436


namespace defective_probability_l2794_279410

/-- The probability of an item being produced by Machine 1 -/
def prob_machine1 : ℝ := 0.4

/-- The probability of an item being produced by Machine 2 -/
def prob_machine2 : ℝ := 0.6

/-- The probability of a defective item from Machine 1 -/
def defect_rate1 : ℝ := 0.03

/-- The probability of a defective item from Machine 2 -/
def defect_rate2 : ℝ := 0.02

/-- The probability of a randomly selected item being defective -/
def prob_defective : ℝ := prob_machine1 * defect_rate1 + prob_machine2 * defect_rate2

theorem defective_probability : prob_defective = 0.024 := by
  sorry

end defective_probability_l2794_279410
