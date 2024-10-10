import Mathlib

namespace percentage_in_accounting_l1546_154633

def accountant_years : ℕ := 25
def manager_years : ℕ := 15
def total_lifespan : ℕ := 80

def accounting_years : ℕ := accountant_years + manager_years

theorem percentage_in_accounting : 
  (accounting_years : ℚ) / total_lifespan * 100 = 50 := by
  sorry

end percentage_in_accounting_l1546_154633


namespace apples_to_eat_raw_l1546_154674

theorem apples_to_eat_raw (total : ℕ) (wormy : ℕ) (bruised : ℕ) : 
  total = 85 →
  wormy = total / 5 →
  bruised = wormy + 9 →
  total - (wormy + bruised) = 42 := by
sorry

end apples_to_eat_raw_l1546_154674


namespace cards_left_l1546_154684

def basketball_boxes : ℕ := 4
def basketball_cards_per_box : ℕ := 10
def baseball_boxes : ℕ := 5
def baseball_cards_per_box : ℕ := 8
def cards_given_away : ℕ := 58

theorem cards_left : 
  basketball_boxes * basketball_cards_per_box + 
  baseball_boxes * baseball_cards_per_box - 
  cards_given_away = 22 := by sorry

end cards_left_l1546_154684


namespace sum_of_squares_of_quadratic_solutions_l1546_154653

theorem sum_of_squares_of_quadratic_solutions : 
  let a : ℝ := -2
  let b : ℝ := -4
  let c : ℝ := -42
  let α : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let β : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  α^2 + β^2 = 46 := by sorry

end sum_of_squares_of_quadratic_solutions_l1546_154653


namespace auto_finance_fraction_l1546_154686

theorem auto_finance_fraction (total_credit : ℝ) (auto_credit_percentage : ℝ) (finance_company_credit : ℝ) :
  total_credit = 475 →
  auto_credit_percentage = 0.36 →
  finance_company_credit = 57 →
  finance_company_credit / (auto_credit_percentage * total_credit) = 1 / 3 :=
by sorry

end auto_finance_fraction_l1546_154686


namespace exactly_one_zero_two_zeros_greater_than_neg_one_l1546_154694

-- Define the function f(x) in terms of m
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 3*m + 4

-- Theorem for condition 1
theorem exactly_one_zero (m : ℝ) :
  (∃! x, f m x = 0) ↔ (m = 4 ∨ m = -1) :=
sorry

-- Theorem for condition 2
theorem two_zeros_greater_than_neg_one (m : ℝ) :
  (∃ x y, x > -1 ∧ y > -1 ∧ x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ 
  (m > -5 ∧ m < -1) :=
sorry

end exactly_one_zero_two_zeros_greater_than_neg_one_l1546_154694


namespace snake_sale_amount_l1546_154643

/-- Given Gary's initial and final amounts, calculate the amount he received from selling his pet snake. -/
theorem snake_sale_amount (initial_amount final_amount : ℝ) 
  (h1 : initial_amount = 73.0)
  (h2 : final_amount = 128) :
  final_amount - initial_amount = 55 := by
  sorry

end snake_sale_amount_l1546_154643


namespace function_domain_l1546_154648

/-- The domain of the function y = ln(x+1) / sqrt(-x^2 - 3x + 4) -/
theorem function_domain (x : ℝ) : 
  (x + 1 > 0 ∧ -x^2 - 3*x + 4 > 0) ↔ -1 < x ∧ x < 1 := by sorry

end function_domain_l1546_154648


namespace lcm_bound_implies_lower_bound_l1546_154608

theorem lcm_bound_implies_lower_bound (a : Fin 2000 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_order : ∀ i j, i < j → a i < a j)
  (h_upper_bound : ∀ i, a i < 4000)
  (h_lcm : ∀ i j, i ≠ j → Nat.lcm (a i) (a j) ≥ 4000) :
  a 0 ≥ 1334 := by
sorry

end lcm_bound_implies_lower_bound_l1546_154608


namespace de_morgan_laws_l1546_154660

universe u

theorem de_morgan_laws {α : Type u} (A B : Set α) : 
  ((A ∪ B)ᶜ = Aᶜ ∩ Bᶜ) ∧ ((A ∩ B)ᶜ = Aᶜ ∪ Bᶜ) := by
  sorry

end de_morgan_laws_l1546_154660


namespace gunther_free_time_l1546_154628

/-- Represents the time in minutes for each cleaning task -/
structure CleaningTasks where
  vacuum : ℕ
  dust : ℕ
  mop : ℕ
  brush_cat : ℕ

/-- Calculates the total cleaning time in hours -/
def total_cleaning_time (tasks : CleaningTasks) (num_cats : ℕ) : ℚ :=
  (tasks.vacuum + tasks.dust + tasks.mop + tasks.brush_cat * num_cats) / 60

/-- Theorem: If Gunther has no cats and 30 minutes of free time left after cleaning,
    his initial free time was 2.75 hours -/
theorem gunther_free_time (tasks : CleaningTasks) 
    (h1 : tasks.vacuum = 45)
    (h2 : tasks.dust = 60)
    (h3 : tasks.mop = 30)
    (h4 : tasks.brush_cat = 5)
    (h5 : total_cleaning_time tasks 0 + 0.5 = 2.75) : True :=
  sorry

end gunther_free_time_l1546_154628


namespace square_sum_xy_l1546_154683

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : 1 / x^2 + 1 / y^2 = 7)
  (h3 : x - y = Real.sqrt 10) :
  (x + y)^2 = 264 := by
  sorry

end square_sum_xy_l1546_154683


namespace sheep_barn_problem_l1546_154659

/-- Given a number of sheep between 2000 and 2100, if the probability of selecting
    two different sheep from different barns is exactly 1/2, then the number of
    sheep must be 2025. -/
theorem sheep_barn_problem (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2100) :
  (∃ k : ℕ, k < n ∧ 2 * k * (n - k) = n * (n - 1)) → n = 2025 :=
by sorry

end sheep_barn_problem_l1546_154659


namespace dog_tether_area_l1546_154646

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem dog_tether_area (side_length : Real) (rope_length : Real) :
  side_length = 1 ∧ rope_length = 3 →
  let hexagon_area := 3 * Real.sqrt 3 / 2 * side_length^2
  let tether_area := 2 * Real.pi * rope_length^2 / 3 + Real.pi * (rope_length - side_length)^2 / 3
  tether_area - hexagon_area = 22 * Real.pi / 3 := by
sorry

end dog_tether_area_l1546_154646


namespace third_runner_time_l1546_154672

/-- A relay race with 4 runners -/
structure RelayRace where
  mary : ℝ
  susan : ℝ
  third : ℝ
  tiffany : ℝ

/-- The conditions of the relay race -/
def validRelayRace (race : RelayRace) : Prop :=
  race.mary = 2 * race.susan ∧
  race.susan = race.third + 10 ∧
  race.tiffany = race.mary - 7 ∧
  race.mary + race.susan + race.third + race.tiffany = 223

/-- The theorem stating that the third runner's time is 30 seconds -/
theorem third_runner_time (race : RelayRace) 
  (h : validRelayRace race) : race.third = 30 := by
  sorry

end third_runner_time_l1546_154672


namespace shooting_probabilities_l1546_154605

/-- The probability of shooter A hitting the target -/
def P_A : ℝ := 0.9

/-- The probability of shooter B hitting the target -/
def P_B : ℝ := 0.8

/-- The probability of both A and B hitting the target -/
def P_both : ℝ := P_A * P_B

/-- The probability of at least one of A and B hitting the target -/
def P_at_least_one : ℝ := 1 - (1 - P_A) * (1 - P_B)

theorem shooting_probabilities :
  P_both = 0.72 ∧ P_at_least_one = 0.98 := by
  sorry

end shooting_probabilities_l1546_154605


namespace work_completion_time_l1546_154682

theorem work_completion_time (b a_and_b : ℚ) (hb : b = 35) (hab : a_and_b = 20 / 11) :
  let a : ℚ := (1 / a_and_b - 1 / b)⁻¹
  a = 700 / 365 := by sorry

end work_completion_time_l1546_154682


namespace new_students_admitted_l1546_154602

theorem new_students_admitted (initial_students_per_section : ℕ) 
                               (new_sections : ℕ)
                               (final_total_sections : ℕ)
                               (final_students_per_section : ℕ) :
  initial_students_per_section = 23 →
  new_sections = 5 →
  final_total_sections = 20 →
  final_students_per_section = 19 →
  (final_total_sections * final_students_per_section) - 
  ((final_total_sections - new_sections) * initial_students_per_section) = 35 :=
by sorry

end new_students_admitted_l1546_154602


namespace foreign_language_speakers_l1546_154693

theorem foreign_language_speakers (total_students : ℕ) (male_students : ℕ) (female_students : ℕ) :
  male_students = female_students →
  (3 : ℚ) / 5 * male_students + (2 : ℚ) / 3 * female_students = (19 : ℚ) / 30 * (male_students + female_students) :=
by sorry

end foreign_language_speakers_l1546_154693


namespace consecutive_integers_product_l1546_154658

/-- Given 5 consecutive integers whose sum is 120, their product is 7893600 -/
theorem consecutive_integers_product (x : ℤ) 
  (h1 : (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 120) : 
  (x - 2) * (x - 1) * x * (x + 1) * (x + 2) = 7893600 := by
  sorry

end consecutive_integers_product_l1546_154658


namespace train_length_l1546_154657

/-- Given a train that crosses a bridge and passes a lamp post, calculate its length -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ)
  (h1 : bridge_length = 800)
  (h2 : bridge_time = 45)
  (h3 : post_time = 15) :
  ∃ train_length : ℝ, train_length = 400 ∧ 
  train_length / post_time = (train_length + bridge_length) / bridge_time :=
by sorry

end train_length_l1546_154657


namespace base_k_conversion_l1546_154609

/-- Given a base k, convert a list of digits to its decimal representation -/
def toDecimal (k : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + k * acc) 0

/-- The problem statement -/
theorem base_k_conversion :
  ∃ (k : ℕ), k > 0 ∧ toDecimal k [2, 3, 1] = 30 :=
by
  sorry

end base_k_conversion_l1546_154609


namespace distance_between_points_l1546_154671

/-- The curve equation -/
def curve_equation (x y : ℝ) : Prop := y^2 + x^4 = 2*x^2*y + 1

/-- Theorem stating that for any real number e, if (e, a) and (e, b) are points on the curve y^2 + x^4 = 2x^2y + 1, then |a-b| = 2 -/
theorem distance_between_points (e a b : ℝ) 
  (ha : curve_equation e a) 
  (hb : curve_equation e b) : 
  |a - b| = 2 := by
  sorry

end distance_between_points_l1546_154671


namespace graceGardenTopBedRows_l1546_154649

/-- Represents the garden structure and seed distribution --/
structure Garden where
  totalSeeds : ℕ
  topBedSeedsPerRow : ℕ
  mediumBedRows : ℕ
  mediumBedSeedsPerRow : ℕ
  numMediumBeds : ℕ

/-- Calculates the number of rows in the top bed --/
def topBedRows (g : Garden) : ℕ :=
  (g.totalSeeds - g.numMediumBeds * g.mediumBedRows * g.mediumBedSeedsPerRow) / g.topBedSeedsPerRow

/-- Theorem stating that for Grace's garden, the top bed can hold 8 rows --/
theorem graceGardenTopBedRows :
  let g : Garden := {
    totalSeeds := 320,
    topBedSeedsPerRow := 25,
    mediumBedRows := 3,
    mediumBedSeedsPerRow := 20,
    numMediumBeds := 2
  }
  topBedRows g = 8 := by
  sorry

end graceGardenTopBedRows_l1546_154649


namespace modified_system_solution_l1546_154669

theorem modified_system_solution
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)
  (h₁ : a₁ * 4 + b₁ * 6 = c₁)
  (h₂ : a₂ * 4 + b₂ * 6 = c₂) :
  ∃ (x y : ℝ), x = 5 ∧ y = 10 ∧ 4 * a₁ * x + 3 * b₁ * y = 5 * c₁ ∧ 4 * a₂ * x + 3 * b₂ * y = 5 * c₂ :=
by sorry

end modified_system_solution_l1546_154669


namespace equal_spacing_theorem_l1546_154616

/-- The width of the wall in millimeters -/
def wall_width : ℕ := 4800

/-- The width of each picture in millimeters -/
def picture_width : ℕ := 420

/-- The number of pictures -/
def num_pictures : ℕ := 4

/-- The distance from the center of each middle picture to the center of the wall -/
def middle_picture_distance : ℕ := 730

/-- Theorem stating that the distance from the center of each middle picture
    to the center of the wall is 730 mm when all pictures are equally spaced -/
theorem equal_spacing_theorem :
  let total_space := wall_width - picture_width
  let spacing := total_space / (num_pictures - 1)
  spacing / 2 = middle_picture_distance := by sorry

end equal_spacing_theorem_l1546_154616


namespace units_digit_of_F_F8_l1546_154636

def modifiedFibonacci : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | (n + 2) => modifiedFibonacci (n + 1) + modifiedFibonacci n

theorem units_digit_of_F_F8 : 
  (modifiedFibonacci (modifiedFibonacci 8)) % 10 = 1 := by sorry

end units_digit_of_F_F8_l1546_154636


namespace partial_fraction_decomposition_l1546_154618

theorem partial_fraction_decomposition 
  (a b c d : ℤ) (h : a * d ≠ b * c) :
  ∃ (r s : ℝ), ∀ (x : ℝ), 
    1 / ((a * x + b) * (c * x + d)) = 
    r / (a * x + b) + s / (c * x + d) := by
  sorry

end partial_fraction_decomposition_l1546_154618


namespace train_speed_fraction_l1546_154670

theorem train_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 50.000000000000014 →
  delay = 10 →
  (usual_time / (usual_time + delay)) = (5 : ℝ) / 6 := by
sorry

end train_speed_fraction_l1546_154670


namespace total_distance_traveled_l1546_154621

/-- Calculates the total distance traveled by a man rowing upstream and downstream in a river. -/
theorem total_distance_traveled
  (man_speed : ℝ)
  (river_speed : ℝ)
  (total_time : ℝ)
  (h1 : man_speed = 6)
  (h2 : river_speed = 1.2)
  (h3 : total_time = 1)
  : ∃ (distance : ℝ), distance = 5.76 ∧ 
    (distance / (man_speed - river_speed) + distance / (man_speed + river_speed) = total_time) :=
by sorry

end total_distance_traveled_l1546_154621


namespace probability_is_correct_l1546_154619

def total_vehicles : ℕ := 20000
def shattered_windshields : ℕ := 600

def probability_shattered_windshield : ℚ :=
  shattered_windshields / total_vehicles

theorem probability_is_correct : 
  probability_shattered_windshield = 3 / 100 := by sorry

end probability_is_correct_l1546_154619


namespace merchant_discount_l1546_154620

theorem merchant_discount (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.2
  let final_price := increased_price * 0.8
  let actual_discount := (original_price - final_price) / original_price
  actual_discount = 0.04 := by
sorry

end merchant_discount_l1546_154620


namespace oscar_swag_bag_scarf_cost_l1546_154692

/-- The cost of each designer scarf in the Oscar swag bag -/
def scarf_cost (total_value earring_cost iphone_cost num_earrings num_scarves : ℕ) : ℕ :=
  (total_value - (num_earrings * earring_cost + iphone_cost)) / num_scarves

/-- Theorem: The cost of each designer scarf in the Oscar swag bag is $1,500 -/
theorem oscar_swag_bag_scarf_cost :
  scarf_cost 20000 6000 2000 2 4 = 1500 := by
  sorry

end oscar_swag_bag_scarf_cost_l1546_154692


namespace rhombus_perimeter_l1546_154603

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
sorry

end rhombus_perimeter_l1546_154603


namespace amy_music_files_l1546_154680

theorem amy_music_files :
  ∀ (initial_music_files : ℕ),
    initial_music_files + 21 - 23 = 2 →
    initial_music_files = 4 :=
by
  sorry

end amy_music_files_l1546_154680


namespace complex_number_property_l1546_154696

theorem complex_number_property (w : ℂ) (h : w + 1 / w = 2 * Real.cos (π / 4)) :
  w^12 + 1 / w^12 = -2 := by sorry

end complex_number_property_l1546_154696


namespace tabitha_current_age_l1546_154667

def tabitha_hair_colors (age : ℕ) : ℕ :=
  age - 13

theorem tabitha_current_age : 
  ∃ (current_age : ℕ), 
    tabitha_hair_colors current_age = 5 ∧ 
    tabitha_hair_colors (current_age + 3) = 8 ∧ 
    current_age = 18 := by
  sorry

end tabitha_current_age_l1546_154667


namespace ages_sum_l1546_154638

theorem ages_sum (a b c : ℕ+) : 
  b = c →                 -- twins have the same age
  b > a →                 -- twins are older than Kiana
  a * b * c = 144 →       -- product of ages is 144
  a + b + c = 16 :=       -- sum of ages is 16
by sorry

end ages_sum_l1546_154638


namespace angle_problem_l1546_154679

theorem angle_problem (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π)  -- θ is in the second quadrant
  (h2 : Real.tan (θ + π/4) = 1/2) :
  Real.tan θ = -1/3 ∧ 
  Real.sin (π/2 - 2*θ) + Real.sin (π + 2*θ) = 7/5 := by
  sorry

end angle_problem_l1546_154679


namespace ceiling_floor_sum_l1546_154647

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_l1546_154647


namespace cake_sugar_calculation_l1546_154644

theorem cake_sugar_calculation (frosting_sugar cake_sugar : ℝ) 
  (h1 : frosting_sugar = 0.6) 
  (h2 : cake_sugar = 0.2) : 
  frosting_sugar + cake_sugar = 0.8 := by
sorry

end cake_sugar_calculation_l1546_154644


namespace sin_theta_value_l1546_154663

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 8 * (Real.cos (x / 2))^2

theorem sin_theta_value (θ : ℝ) :
  (∀ x, f x ≤ f θ) → Real.sin θ = 3/5 :=
by sorry

end sin_theta_value_l1546_154663


namespace total_notes_count_l1546_154613

/-- Proves that given a total amount of Rs. 10350 in Rs. 50 and Rs. 500 notes, 
    with 57 notes of Rs. 50 denomination, the total number of notes is 72. -/
theorem total_notes_count (total_amount : ℕ) (fifty_note_count : ℕ) : 
  total_amount = 10350 →
  fifty_note_count = 57 →
  ∃ (five_hundred_note_count : ℕ),
    total_amount = fifty_note_count * 50 + five_hundred_note_count * 500 ∧
    fifty_note_count + five_hundred_note_count = 72 :=
by sorry

end total_notes_count_l1546_154613


namespace die_roll_probabilities_l1546_154676

def DieFaces := Finset.range 6

def roll_twice : Finset (ℕ × ℕ) :=
  DieFaces.product DieFaces

theorem die_roll_probabilities :
  let total_outcomes := (roll_twice.card : ℚ)
  let sum_at_least_nine := (roll_twice.filter (fun (a, b) => a + b ≥ 9)).card
  let tangent_to_circle := (roll_twice.filter (fun (a, b) => a^2 + b^2 = 25)).card
  let isosceles_triangle := (roll_twice.filter (fun (a, b) => 
    a = b ∨ a = 5 ∨ b = 5)).card
  (sum_at_least_nine : ℚ) / total_outcomes = 5 / 18 ∧
  (tangent_to_circle : ℚ) / total_outcomes = 1 / 18 ∧
  (isosceles_triangle : ℚ) / total_outcomes = 7 / 18 :=
by sorry

end die_roll_probabilities_l1546_154676


namespace arithmetic_sequence_ratio_l1546_154645

/-- Given two arithmetic sequences {a_n} and {b_n} with S_n and T_n as the sum of their first n terms respectively -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2 ∧ T n = (n * (b 1 + b n)) / 2

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequences a b S T)
  (h_ratio : ∀ n, S n / T n = (7 * n + 1) / (n + 3)) :
  (a 2 + a 5 + a 17 + a 22) / (b 8 + b 10 + b 12 + b 16) = 31 / 5 ∧
  a 5 / b 5 = 16 / 3 := by
sorry

end arithmetic_sequence_ratio_l1546_154645


namespace family_weight_ratio_l1546_154655

/-- Given the weights of three generations in a family, prove the ratio of the child's weight to the grandmother's weight -/
theorem family_weight_ratio :
  ∀ (grandmother daughter child : ℝ),
  grandmother + daughter + child = 110 →
  daughter + child = 60 →
  daughter = 50 →
  child / grandmother = 1 / 5 := by
sorry

end family_weight_ratio_l1546_154655


namespace union_A_M_eq_real_union_B_complement_M_eq_B_l1546_154625

-- Define the sets A, B, and M
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 9}
def B (b : ℝ) : Set ℝ := {x | 8 - b < x ∧ x < b}
def M : Set ℝ := {x | x < -1 ∨ x > 5}

-- Statement for the first part of the problem
theorem union_A_M_eq_real (a : ℝ) :
  A a ∪ M = Set.univ ↔ -4 ≤ a ∧ a ≤ -1 :=
sorry

-- Statement for the second part of the problem
theorem union_B_complement_M_eq_B (b : ℝ) :
  B b ∪ (Set.univ \ M) = B b ↔ b > 9 :=
sorry

end union_A_M_eq_real_union_B_complement_M_eq_B_l1546_154625


namespace ratio_of_sums_is_301_480_l1546_154629

/-- Calculate the sum of an arithmetic sequence -/
def sum_arithmetic (a1 : ℚ) (d : ℚ) (an : ℚ) : ℚ :=
  let n : ℚ := (an - a1) / d + 1
  n * (a1 + an) / 2

/-- The ratio of sums of two specific arithmetic sequences -/
def ratio_of_sums : ℚ :=
  (sum_arithmetic 2 3 41) / (sum_arithmetic 4 4 60)

theorem ratio_of_sums_is_301_480 : ratio_of_sums = 301 / 480 := by
  sorry

end ratio_of_sums_is_301_480_l1546_154629


namespace three_digit_reversal_difference_l1546_154673

theorem three_digit_reversal_difference (A B C : ℕ) 
  (h1 : A ≠ C) 
  (h2 : A ≥ 1 ∧ A ≤ 9) 
  (h3 : B ≥ 0 ∧ B ≤ 9) 
  (h4 : C ≥ 0 ∧ C ≤ 9) : 
  ∃ k : ℤ, (100 * A + 10 * B + C) - (100 * C + 10 * B + A) = 3 * k := by
  sorry

end three_digit_reversal_difference_l1546_154673


namespace carwash_problem_l1546_154656

theorem carwash_problem (car_price truck_price suv_price : ℕ) 
                        (total_raised : ℕ) 
                        (num_cars num_trucks : ℕ) : 
  car_price = 5 →
  truck_price = 6 →
  suv_price = 7 →
  total_raised = 100 →
  num_cars = 7 →
  num_trucks = 5 →
  ∃ (num_suvs : ℕ), 
    num_suvs * suv_price + num_cars * car_price + num_trucks * truck_price = total_raised ∧
    num_suvs = 5 := by
  sorry

end carwash_problem_l1546_154656


namespace keystone_arch_angle_l1546_154630

/-- A keystone arch composed of congruent isosceles trapezoids -/
structure KeystoneArch where
  /-- The number of trapezoids in the arch -/
  num_trapezoids : ℕ
  /-- The measure of the central angle between two adjacent trapezoids in degrees -/
  central_angle : ℝ
  /-- The measure of the smaller interior angle of each trapezoid in degrees -/
  smaller_angle : ℝ
  /-- The measure of the larger interior angle of each trapezoid in degrees -/
  larger_angle : ℝ
  /-- The sum of interior angles of a trapezoid is 360° -/
  angle_sum : smaller_angle + larger_angle = 180
  /-- The central angle is related to the number of trapezoids -/
  central_angle_def : central_angle = 360 / num_trapezoids
  /-- The smaller angle plus half the central angle is 90° -/
  smaller_angle_def : smaller_angle + central_angle / 2 = 90

/-- Theorem: In a keystone arch with 10 congruent isosceles trapezoids, 
    the larger interior angle of each trapezoid is 99° -/
theorem keystone_arch_angle (arch : KeystoneArch) 
    (h : arch.num_trapezoids = 10) : arch.larger_angle = 99 := by
  sorry

end keystone_arch_angle_l1546_154630


namespace train_speed_calculation_l1546_154627

/-- Calculates the speed of a train in km/hr given its length in meters and time to cross a pole in seconds. -/
def trainSpeed (length : Float) (time : Float) : Float :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a train with a length of 450.00000000000006 meters 
    crossing a pole in 27 seconds has a speed of 60 km/hr. -/
theorem train_speed_calculation :
  let length : Float := 450.00000000000006
  let time : Float := 27
  trainSpeed length time = 60 := by
  sorry

#eval trainSpeed 450.00000000000006 27

end train_speed_calculation_l1546_154627


namespace program_result_l1546_154677

/-- The program's operation on input n -/
def program (n : ℝ) : ℝ := n^2 + 3*n - (2*n^2 - n)

/-- Theorem stating that the program's result equals -n^2 + 4n for any real n -/
theorem program_result (n : ℝ) : program n = -n^2 + 4*n := by
  sorry

end program_result_l1546_154677


namespace molecular_weight_Al2S3_proof_l1546_154661

/-- The molecular weight of Al2S3 in g/mol -/
def molecular_weight_Al2S3 : ℝ := 150

/-- The number of moles used in the given condition -/
def moles : ℝ := 3

/-- The total weight of the given number of moles in grams -/
def total_weight : ℝ := 450

/-- Theorem: The molecular weight of Al2S3 is 150 g/mol, given that 3 moles weigh 450 grams -/
theorem molecular_weight_Al2S3_proof : 
  molecular_weight_Al2S3 = total_weight / moles := by
  sorry

end molecular_weight_Al2S3_proof_l1546_154661


namespace total_time_wasted_l1546_154691

def traffic_wait_time : ℝ := 2
def freeway_exit_time_multiplier : ℝ := 4

theorem total_time_wasted : 
  traffic_wait_time + freeway_exit_time_multiplier * traffic_wait_time = 10 := by
  sorry

end total_time_wasted_l1546_154691


namespace inequalities_given_negative_order_l1546_154642

theorem inequalities_given_negative_order (a b : ℝ) (h : b < a ∧ a < 0) :
  a^2 < b^2 ∧ 
  a * b > b^2 ∧ 
  (1/2 : ℝ)^b > (1/2 : ℝ)^a ∧ 
  a / b + b / a > 2 := by
sorry

end inequalities_given_negative_order_l1546_154642


namespace two_primes_between_lower_limit_and_14_l1546_154622

theorem two_primes_between_lower_limit_and_14 : 
  ∃ (x : ℕ), x ≤ 7 ∧ 
  (∃ (p q : ℕ), Prime p ∧ Prime q ∧ x < p ∧ p < q ∧ q < 14) ∧
  (∀ (y : ℕ), y > 7 → ¬(∃ (p q : ℕ), Prime p ∧ Prime q ∧ y < p ∧ p < q ∧ q < 14)) :=
sorry

end two_primes_between_lower_limit_and_14_l1546_154622


namespace birds_on_fence_l1546_154626

theorem birds_on_fence (initial_birds : ℕ) (total_birds : ℕ) (new_birds : ℕ) : 
  initial_birds = 2 → total_birds = 6 → new_birds = total_birds - initial_birds → new_birds = 4 := by
  sorry

end birds_on_fence_l1546_154626


namespace modular_inverse_35_mod_36_l1546_154651

theorem modular_inverse_35_mod_36 : ∃ x : ℤ, (35 * x) % 36 = 1 ∧ x % 36 = 35 := by
  sorry

end modular_inverse_35_mod_36_l1546_154651


namespace pet_store_combinations_l1546_154610

theorem pet_store_combinations (puppies kittens hamsters : ℕ) 
  (h1 : puppies = 20) (h2 : kittens = 9) (h3 : hamsters = 12) :
  (puppies * kittens * hamsters) * 6 = 12960 := by
  sorry

end pet_store_combinations_l1546_154610


namespace sqrt_2x_plus_1_domain_l1546_154681

theorem sqrt_2x_plus_1_domain (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x + 1) ↔ x ≥ -1/2 := by
  sorry

end sqrt_2x_plus_1_domain_l1546_154681


namespace other_number_proof_l1546_154637

theorem other_number_proof (x y : ℕ+) 
  (h1 : Nat.lcm x y = 2640)
  (h2 : Nat.gcd x y = 24)
  (h3 : x = 240) :
  y = 264 := by
  sorry

end other_number_proof_l1546_154637


namespace dice_puzzle_l1546_154678

/-- Given five dice with 21 dots each and 43 visible dots, prove that 62 dots are not visible -/
theorem dice_puzzle (num_dice : ℕ) (dots_per_die : ℕ) (visible_dots : ℕ) : 
  num_dice = 5 → dots_per_die = 21 → visible_dots = 43 → 
  num_dice * dots_per_die - visible_dots = 62 := by
  sorry

end dice_puzzle_l1546_154678


namespace dice_product_nonzero_probability_l1546_154614

/-- The probability of getting a specific outcome when rolling a standard die -/
def roll_probability : ℚ := 1 / 6

/-- The number of faces on a standard die -/
def die_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability that a single die roll is not 1 -/
def prob_not_one : ℚ := (die_faces - 1) / die_faces

theorem dice_product_nonzero_probability :
  (prob_not_one ^ num_dice : ℚ) = 625 / 1296 := by sorry

end dice_product_nonzero_probability_l1546_154614


namespace equation_solution_l1546_154675

theorem equation_solution : ∃! x : ℝ, (x^2 + x)^2 + Real.sqrt (x^2 - 1) = 0 ∧ x = -1 := by
  sorry

end equation_solution_l1546_154675


namespace floor_sum_difference_l1546_154601

theorem floor_sum_difference (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ⌊a + b⌋ - (⌊a⌋ + ⌊b⌋) = 0 ∨ ⌊a + b⌋ - (⌊a⌋ + ⌊b⌋) = 1 :=
by sorry

end floor_sum_difference_l1546_154601


namespace koi_fish_count_l1546_154687

/-- Calculates the number of koi fish after three weeks given the initial conditions and final number of goldfish --/
theorem koi_fish_count (initial_total : ℕ) (koi_added_per_day : ℕ) (goldfish_added_per_day : ℕ) (days : ℕ) (final_goldfish : ℕ) : 
  initial_total = 280 →
  days = 21 →
  koi_added_per_day = 2 →
  goldfish_added_per_day = 5 →
  final_goldfish = 200 →
  initial_total + days * (koi_added_per_day + goldfish_added_per_day) - final_goldfish = 227 :=
by
  sorry

#check koi_fish_count

end koi_fish_count_l1546_154687


namespace intersection_line_of_circles_l1546_154600

/-- Given two circles in the plane, this theorem states that the line passing through their
    intersection points has a specific equation. -/
theorem intersection_line_of_circles
  (circle1 : Set (ℝ × ℝ))
  (circle2 : Set (ℝ × ℝ))
  (h1 : circle1 = {(x, y) | x^2 + y^2 - 4*x + 6*y = 0})
  (h2 : circle2 = {(x, y) | x^2 + y^2 - 6*x = 0})
  (h3 : (circle1 ∩ circle2).Nonempty) :
  ∃ (A B : ℝ × ℝ),
    A ∈ circle1 ∧ A ∈ circle2 ∧
    B ∈ circle1 ∧ B ∈ circle2 ∧
    A ≠ B ∧
    (∀ (x y : ℝ), (x, y) ∈ Set.Icc A B → x + 3*y = 0) :=
sorry

end intersection_line_of_circles_l1546_154600


namespace christopher_alexander_difference_l1546_154612

/-- Represents the number of joggers bought by each person -/
structure JoggerPurchases where
  christopher : Nat
  tyson : Nat
  alexander : Nat

/-- The conditions of the jogger purchase problem -/
def jogger_problem (purchases : JoggerPurchases) : Prop :=
  purchases.christopher = 80 ∧
  purchases.christopher = 20 * purchases.tyson ∧
  purchases.alexander = purchases.tyson + 22

/-- The theorem to be proved -/
theorem christopher_alexander_difference 
  (purchases : JoggerPurchases) 
  (h : jogger_problem purchases) : 
  purchases.christopher - purchases.alexander = 54 := by
  sorry

end christopher_alexander_difference_l1546_154612


namespace f_max_value_l1546_154617

/-- The quadratic function f(y) = -3y^2 + 18y - 7 -/
def f (y : ℝ) : ℝ := -3 * y^2 + 18 * y - 7

/-- The maximum value of f(y) is 20 -/
theorem f_max_value : ∃ (M : ℝ), M = 20 ∧ ∀ (y : ℝ), f y ≤ M := by
  sorry

end f_max_value_l1546_154617


namespace simplest_form_count_l1546_154699

-- Define the fractions
def fraction1 (a b : ℚ) : ℚ := b / (8 * a)
def fraction2 (a b : ℚ) : ℚ := (a + b) / (a - b)
def fraction3 (x y : ℚ) : ℚ := (x - y) / (x^2 - y^2)
def fraction4 (x y : ℚ) : ℚ := (x - y) / (x^2 + 2*x*y + y^2)

-- Define a function to check if a fraction is in simplest form
def isSimplestForm (f : ℚ → ℚ → ℚ) : Prop := 
  ∀ a b, a ≠ 0 → b ≠ 0 → (∃ c, f a b = c) → 
    ¬∃ d e, d ≠ 0 ∧ e ≠ 0 ∧ f (a*d) (b*e) = f a b

-- Theorem statement
theorem simplest_form_count : 
  (isSimplestForm fraction1) ∧ 
  (isSimplestForm fraction2) ∧ 
  ¬(isSimplestForm fraction3) ∧
  (isSimplestForm fraction4) := by sorry

end simplest_form_count_l1546_154699


namespace fan_ratio_theorem_l1546_154604

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of NY Yankees fans to NY Mets fans is 3:2 -/
def yankees_mets_ratio (fc : FanCounts) : Prop :=
  fc.yankees * 2 = fc.mets * 3

/-- The total number of fans is 390 -/
def total_fans (fc : FanCounts) : Prop :=
  fc.yankees + fc.mets + fc.red_sox = 390

/-- There are 104 NY Mets fans -/
def mets_fans_count (fc : FanCounts) : Prop :=
  fc.mets = 104

/-- The ratio of NY Mets fans to Boston Red Sox fans is 4:5 -/
def mets_red_sox_ratio (fc : FanCounts) : Prop :=
  fc.mets * 5 = fc.red_sox * 4

theorem fan_ratio_theorem (fc : FanCounts) :
  yankees_mets_ratio fc → total_fans fc → mets_fans_count fc → mets_red_sox_ratio fc := by
  sorry

end fan_ratio_theorem_l1546_154604


namespace cat_meow_ratio_l1546_154662

/-- Given three cats meowing, prove the ratio of meows per minute of the third cat to the second cat -/
theorem cat_meow_ratio :
  ∀ (cat1_rate cat2_rate cat3_rate : ℚ),
  cat1_rate = 3 →
  cat2_rate = 2 * cat1_rate →
  5 * (cat1_rate + cat2_rate + cat3_rate) = 55 →
  cat3_rate / cat2_rate = 1 / 3 := by
sorry

end cat_meow_ratio_l1546_154662


namespace f_min_max_l1546_154615

def f (x : ℝ) : ℝ := 2 * x^2 - 6 * x + 1

theorem f_min_max :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≥ -3) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = -3) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 9) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 9) :=
by sorry

end f_min_max_l1546_154615


namespace sum_of_products_l1546_154611

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a + b + c = 20) : 
  a*b + b*c + c*a = 5 := by
sorry

end sum_of_products_l1546_154611


namespace cost_per_person_is_correct_l1546_154639

def item1_base_cost : ℝ := 40
def item1_tax_rate : ℝ := 0.05
def item1_discount_rate : ℝ := 0.10

def item2_base_cost : ℝ := 70
def item2_tax_rate : ℝ := 0.08
def item2_coupon : ℝ := 5

def item3_base_cost : ℝ := 100
def item3_tax_rate : ℝ := 0.06
def item3_discount_rate : ℝ := 0.10

def num_people : ℕ := 3

def calculate_item1_cost : ℝ := 
  let cost_after_tax := item1_base_cost * (1 + item1_tax_rate)
  cost_after_tax * (1 - item1_discount_rate)

def calculate_item2_cost : ℝ := 
  item2_base_cost * (1 + item2_tax_rate) - item2_coupon

def calculate_item3_cost : ℝ := 
  let cost_after_tax := item3_base_cost * (1 + item3_tax_rate)
  cost_after_tax * (1 - item3_discount_rate)

def total_cost : ℝ := 
  calculate_item1_cost + calculate_item2_cost + calculate_item3_cost

theorem cost_per_person_is_correct : 
  total_cost / num_people = 67.93 := by sorry

end cost_per_person_is_correct_l1546_154639


namespace power_sum_sequence_l1546_154666

theorem power_sum_sequence (a b : ℝ) : 
  a + b = 1 → 
  a^2 + b^2 = 3 → 
  a^3 + b^3 = 4 → 
  a^4 + b^4 = 7 → 
  a^5 + b^5 = 11 → 
  a^10 + b^10 = 123 := by
sorry

end power_sum_sequence_l1546_154666


namespace q_is_false_l1546_154631

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
by sorry

end q_is_false_l1546_154631


namespace art_book_cost_is_two_l1546_154698

/-- The cost of each art book given the number of books and their prices --/
def cost_of_art_book (math_books science_books art_books : ℕ) 
                     (total_cost : ℚ) (math_science_cost : ℚ) : ℚ :=
  (total_cost - (math_books + science_books : ℚ) * math_science_cost) / art_books

/-- Theorem stating that the cost of each art book is $2 --/
theorem art_book_cost_is_two :
  cost_of_art_book 2 6 3 30 3 = 2 := by
  sorry

end art_book_cost_is_two_l1546_154698


namespace intersecting_quadratic_properties_l1546_154688

/-- A quadratic function that intersects both coordinate axes at three points -/
structure IntersectingQuadratic where
  b : ℝ
  intersects_axes : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ -x₁^2 - 2*x₁ + b = 0 ∧ -x₂^2 - 2*x₂ + b = 0
  intersects_y : b ≠ 0

/-- The range of possible values for b -/
def valid_b_range (q : IntersectingQuadratic) : Prop :=
  q.b > -1 ∧ q.b ≠ 0

/-- The equation of the circle passing through the three intersection points -/
def circle_equation (q : IntersectingQuadratic) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + (1 - q.b)*y - q.b = 0

theorem intersecting_quadratic_properties (q : IntersectingQuadratic) :
  valid_b_range q ∧
  ∀ (x y : ℝ), circle_equation q x y ↔ 
    (x = 0 ∧ y = q.b) ∨ 
    (y = 0 ∧ -x^2 - 2*x + q.b = 0) :=
sorry

end intersecting_quadratic_properties_l1546_154688


namespace hexagon_circle_area_ratio_l1546_154640

/-- Given a regular hexagon and a circle with equal perimeter/circumference,
    the ratio of the area of the hexagon to the area of the circle is π√3/6 -/
theorem hexagon_circle_area_ratio :
  ∀ (s r : ℝ),
  s > 0 → r > 0 →
  6 * s = 2 * Real.pi * r →
  (3 * Real.sqrt 3 / 2 * s^2) / (Real.pi * r^2) = Real.pi * Real.sqrt 3 / 6 := by
sorry

end hexagon_circle_area_ratio_l1546_154640


namespace eighty_seventh_odd_integer_l1546_154635

theorem eighty_seventh_odd_integer : ∀ n : ℕ, n > 0 → (2 * n - 1) = 173 ↔ n = 87 := by
  sorry

end eighty_seventh_odd_integer_l1546_154635


namespace complex_modulus_theorem_l1546_154652

theorem complex_modulus_theorem (z : ℂ) (h : z * Complex.I = 1 - Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_theorem_l1546_154652


namespace delta_value_l1546_154634

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ - 3 → Δ = -9 := by
  sorry

end delta_value_l1546_154634


namespace units_digit_problem_l1546_154668

theorem units_digit_problem : (25^3 + 17^3) * 12^2 % 10 = 2 := by
  sorry

end units_digit_problem_l1546_154668


namespace expression_factorization_l1546_154632

theorem expression_factorization (x : ℝ) : 
  (16 * x^4 + 36 * x^2 - 9) - (4 * x^4 - 6 * x^2 - 9) = 6 * x^2 * (2 * x^2 + 7) :=
by sorry

end expression_factorization_l1546_154632


namespace jellybean_box_scaling_l1546_154664

theorem jellybean_box_scaling (bert_jellybeans : ℕ) (scale_factor : ℕ) : 
  bert_jellybeans = 150 → scale_factor = 3 →
  (scale_factor ^ 3 : ℕ) * bert_jellybeans = 4050 := by
  sorry

end jellybean_box_scaling_l1546_154664


namespace f_min_value_inequality_solution_l1546_154689

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 5|

-- Theorem 1: The minimum value of f(x) is 6
theorem f_min_value : ∀ x : ℝ, f x ≥ 6 := by sorry

-- Theorem 2: Solution to the inequality when m = 6
theorem inequality_solution :
  ∀ x : ℝ, (|x - 3| - 2*x ≤ 4) ↔ (x ≥ -1/3) := by sorry

end f_min_value_inequality_solution_l1546_154689


namespace jane_usable_sheets_l1546_154624

/-- Represents the total number of sheets Jane has for each type and size --/
structure TotalSheets where
  brownA4 : ℕ
  yellowA4 : ℕ
  yellowA3 : ℕ

/-- Represents the number of damaged sheets (less than 70% intact) for each type and size --/
structure DamagedSheets where
  brownA4 : ℕ
  yellowA4 : ℕ
  yellowA3 : ℕ

/-- Calculates the number of usable sheets given the total and damaged sheets --/
def usableSheets (total : TotalSheets) (damaged : DamagedSheets) : ℕ :=
  (total.brownA4 - damaged.brownA4) + (total.yellowA4 - damaged.yellowA4) + (total.yellowA3 - damaged.yellowA3)

theorem jane_usable_sheets :
  let total := TotalSheets.mk 28 18 9
  let damaged := DamagedSheets.mk 3 5 2
  usableSheets total damaged = 45 := by
  sorry

end jane_usable_sheets_l1546_154624


namespace max_sum_hexagonal_prism_with_pyramid_l1546_154697

/-- Represents a three-dimensional geometric shape --/
structure Shape3D where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- A hexagonal prism --/
def hexagonal_prism : Shape3D :=
  { faces := 8, vertices := 12, edges := 18 }

/-- Adds a pyramid to a hexagonal face of the prism --/
def add_pyramid_to_hexagonal_face (s : Shape3D) : Shape3D :=
  { faces := s.faces + 5,
    vertices := s.vertices + 1,
    edges := s.edges + 6 }

/-- Adds a pyramid to a rectangular face of the prism --/
def add_pyramid_to_rectangular_face (s : Shape3D) : Shape3D :=
  { faces := s.faces + 3,
    vertices := s.vertices + 1,
    edges := s.edges + 4 }

/-- Calculates the sum of faces, vertices, and edges --/
def sum_features (s : Shape3D) : ℕ :=
  s.faces + s.vertices + s.edges

/-- Theorem: The maximum sum of exterior faces, vertices, and edges 
    when adding a pyramid to a hexagonal prism is 50 --/
theorem max_sum_hexagonal_prism_with_pyramid : 
  max 
    (sum_features (add_pyramid_to_hexagonal_face hexagonal_prism))
    (sum_features (add_pyramid_to_rectangular_face hexagonal_prism)) = 50 := by
  sorry

end max_sum_hexagonal_prism_with_pyramid_l1546_154697


namespace product_less_than_square_l1546_154654

theorem product_less_than_square : 1234567 * 1234569 < 1234568^2 := by
  sorry

end product_less_than_square_l1546_154654


namespace karting_routes_count_l1546_154665

/-- Represents the number of routes ending at point A after n minutes -/
def M_n_A : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 1
| (n+3) => M_n_A (n+1) + M_n_A n

/-- The race duration in minutes -/
def race_duration : ℕ := 10

/-- Theorem stating that the number of routes ending at A after 10 minutes
    is equal to the 10th number in the defined Fibonacci-like sequence -/
theorem karting_routes_count : M_n_A race_duration = 34 := by
  sorry

end karting_routes_count_l1546_154665


namespace cone_ratio_l1546_154606

/-- For a cone with a central angle of 120° in its unfolded lateral surface,
    the ratio of its base radius to its slant height is 1/3 -/
theorem cone_ratio (r : ℝ) (l : ℝ) (h : r > 0) (h' : l > 0) :
  2 * Real.pi * r = 2 * Real.pi / 3 * l → r / l = 1 / 3 := by
sorry

end cone_ratio_l1546_154606


namespace function_value_range_l1546_154623

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * a + 1

-- State the theorem
theorem function_value_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ 1 ∧ -1 ≤ x₂ ∧ x₂ ≤ 1 ∧ f a x₁ < 0 ∧ 0 < f a x₂) →
  -1 < a ∧ a < -1/3 := by
  sorry


end function_value_range_l1546_154623


namespace meaningful_zero_power_l1546_154607

theorem meaningful_zero_power (m : ℝ) (h : m ≠ -1) : (m + 1) ^ (0 : ℕ) = 1 := by
  sorry

end meaningful_zero_power_l1546_154607


namespace factorization_d_is_valid_l1546_154695

/-- Represents a polynomial factorization -/
def IsFactorization (left right : ℝ → ℝ) : Prop :=
  ∀ x, left x = right x ∧ 
       ∃ p q : ℝ → ℝ, right = fun y ↦ p y * q y

/-- The specific factorization we want to prove -/
def FactorizationD (x : ℝ) : ℝ := x^2 + 4*x + 4

/-- The factored form -/
def FactoredFormD (x : ℝ) : ℝ := (x + 2)^2

/-- Theorem stating that FactorizationD is a valid factorization -/
theorem factorization_d_is_valid : IsFactorization FactorizationD FactoredFormD := by
  sorry

end factorization_d_is_valid_l1546_154695


namespace pair_op_theorem_l1546_154641

/-- Definition of the custom operation ⊗ for pairs of real numbers -/
def pair_op (a b c d : ℝ) : ℝ × ℝ := (a * c - b * d, a * d + b * c)

/-- Theorem stating that if (1, 2) ⊗ (p, q) = (5, 0), then p + q equals some real number -/
theorem pair_op_theorem (p q : ℝ) :
  pair_op 1 2 p q = (5, 0) → ∃ r : ℝ, p + q = r := by
  sorry

end pair_op_theorem_l1546_154641


namespace two_thirds_of_fifteen_fourths_l1546_154690

theorem two_thirds_of_fifteen_fourths (x : ℚ) : x = 15 / 4 → (2 / 3) * x = 5 / 2 := by
  sorry

end two_thirds_of_fifteen_fourths_l1546_154690


namespace number_equation_l1546_154650

theorem number_equation (x : ℝ) : 2 * x + 5 = 17 ↔ x = 6 := by
  sorry

end number_equation_l1546_154650


namespace marbles_started_with_l1546_154685

def marbles_bought : Real := 489.0
def total_marbles : Real := 2778.0

theorem marbles_started_with : total_marbles - marbles_bought = 2289.0 := by
  sorry

end marbles_started_with_l1546_154685
