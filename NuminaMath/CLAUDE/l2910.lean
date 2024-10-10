import Mathlib

namespace candy_count_proof_l2910_291097

/-- Calculates the total number of candy pieces given the number of packages and pieces per package -/
def total_candy_pieces (packages : ℕ) (pieces_per_package : ℕ) : ℕ :=
  packages * pieces_per_package

/-- Proves that 45 packages of candy with 9 pieces each results in 405 total pieces -/
theorem candy_count_proof :
  total_candy_pieces 45 9 = 405 := by
  sorry

end candy_count_proof_l2910_291097


namespace alternate_shading_six_by_six_l2910_291021

theorem alternate_shading_six_by_six (grid_size : Nat) (shaded_squares : Nat) :
  grid_size = 6 → shaded_squares = 18 → (shaded_squares : ℚ) / (grid_size * grid_size) = 1/2 := by
  sorry

end alternate_shading_six_by_six_l2910_291021


namespace vector_properties_l2910_291005

def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (1, 2)

theorem vector_properties :
  let cos_angle := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  let projection := ((a.1 + b.1) * a.1 + (a.2 + b.2) * a.2) / Real.sqrt (a.1^2 + a.2^2)
  cos_angle = 4/5 ∧ projection = 14 * Real.sqrt 5 / 5 := by
  sorry

end vector_properties_l2910_291005


namespace factor_theorem_application_l2910_291037

theorem factor_theorem_application (c : ℝ) : 
  (∀ x : ℝ, (x + 7) ∣ (c * x^3 + 19 * x^2 - 3 * c * x + 35)) → c = 3 := by
  sorry

end factor_theorem_application_l2910_291037


namespace pen_profit_percentage_pen_profit_percentage_result_l2910_291079

/-- Given a purchase of pens with specific pricing and discount conditions, 
    calculate the profit percentage. -/
theorem pen_profit_percentage 
  (num_pens : ℕ) 
  (marked_price_ratio : ℚ) 
  (discount_percent : ℚ) : ℚ :=
  let cost_price := marked_price_ratio
  let selling_price := num_pens * (1 - discount_percent / 100)
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  by
    -- Assuming num_pens = 50, marked_price_ratio = 46/50, discount_percent = 1
    sorry

/-- The profit percentage for the given pen sale scenario is 7.61%. -/
theorem pen_profit_percentage_result : 
  pen_profit_percentage 50 (46/50) 1 = 761/100 :=
by sorry

end pen_profit_percentage_pen_profit_percentage_result_l2910_291079


namespace previous_largest_spider_weight_l2910_291048

/-- Proves the weight of the previous largest spider given the characteristics of a giant spider. -/
theorem previous_largest_spider_weight
  (weight_ratio : ℝ)
  (leg_count : ℕ)
  (leg_area : ℝ)
  (leg_pressure : ℝ)
  (h1 : weight_ratio = 2.5)
  (h2 : leg_count = 8)
  (h3 : leg_area = 0.5)
  (h4 : leg_pressure = 4) :
  let giant_spider_weight := leg_count * leg_area * leg_pressure
  giant_spider_weight / weight_ratio = 6.4 := by
sorry

end previous_largest_spider_weight_l2910_291048


namespace Q_roots_nature_l2910_291095

def Q (x : ℝ) : ℝ := x^7 - 2*x^6 - 6*x^4 - 4*x + 16

theorem Q_roots_nature :
  (∀ x < 0, Q x > 0) ∧ 
  (∃ x > 0, Q x < 0) ∧ 
  (∃ x > 0, Q x > 0) :=
by sorry

end Q_roots_nature_l2910_291095


namespace monotonically_decreasing_implies_a_leq_1_l2910_291068

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x

-- State the theorem
theorem monotonically_decreasing_implies_a_leq_1 :
  ∀ a : ℝ, (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x > f a y) → a ≤ 1 := by
  sorry

end monotonically_decreasing_implies_a_leq_1_l2910_291068


namespace area_between_tangent_circles_l2910_291081

/-- The area of the region between two tangent circles -/
theorem area_between_tangent_circles
  (r₁ : ℝ) (r₂ : ℝ) (d : ℝ)
  (h₁ : r₁ = 4)
  (h₂ : r₂ = 7)
  (h₃ : d = 3)
  (h₄ : d = r₂ - r₁) :
  π * (r₂^2 - r₁^2) = 33 * π :=
sorry

end area_between_tangent_circles_l2910_291081


namespace same_solution_implies_k_equals_four_l2910_291060

theorem same_solution_implies_k_equals_four (x k : ℝ) :
  (8 * x - k = 2 * (x + 1)) ∧ 
  (2 * (2 * x - 3) = 1 - 3 * x) ∧ 
  (∃ x, (8 * x - k = 2 * (x + 1)) ∧ (2 * (2 * x - 3) = 1 - 3 * x)) →
  k = 4 :=
by sorry

end same_solution_implies_k_equals_four_l2910_291060


namespace quadrilateral_prism_properties_l2910_291032

structure QuadrilateralPrism where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

theorem quadrilateral_prism_properties :
  ∃ (qp : QuadrilateralPrism), qp.vertices = 8 ∧ qp.edges = 12 ∧ qp.faces = 6 := by
  sorry

end quadrilateral_prism_properties_l2910_291032


namespace cube_sum_given_sum_and_product_l2910_291080

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 9) (h2 : x * y = 10) : x^3 + y^3 = 459 := by
  sorry

end cube_sum_given_sum_and_product_l2910_291080


namespace fathers_cookies_l2910_291073

theorem fathers_cookies (total cookies_charlie cookies_mother : ℕ) 
  (h1 : total = 30)
  (h2 : cookies_charlie = 15)
  (h3 : cookies_mother = 5) :
  total - cookies_charlie - cookies_mother = 10 := by
sorry

end fathers_cookies_l2910_291073


namespace arithmetic_sequence_a3_equals_5_l2910_291099

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that in an arithmetic sequence, a_3 = 5 given the conditions -/
theorem arithmetic_sequence_a3_equals_5 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 3 + a 5 = 15) : 
  a 3 = 5 := by sorry

end arithmetic_sequence_a3_equals_5_l2910_291099


namespace mess_expense_increase_l2910_291062

theorem mess_expense_increase
  (initial_students : ℕ)
  (new_students : ℕ)
  (original_expenditure : ℕ)
  (average_decrease : ℕ)
  (h1 : initial_students = 35)
  (h2 : new_students = 7)
  (h3 : original_expenditure = 420)
  (h4 : average_decrease = 1)
  : (initial_students + new_students) * 
    (original_expenditure / initial_students - average_decrease) - 
    original_expenditure = 42 := by
  sorry

end mess_expense_increase_l2910_291062


namespace right_triangle_angle_bisector_segments_l2910_291096

/-- Given a right triangle where an acute angle bisector divides the adjacent leg into segments m and n,
    prove the lengths of the other leg and hypotenuse. -/
theorem right_triangle_angle_bisector_segments (m n : ℝ) (h : m > n) :
  ∃ (other_leg hypotenuse : ℝ),
    other_leg = n * Real.sqrt ((m + n) / (m - n)) ∧
    hypotenuse = m * Real.sqrt ((m + n) / (m - n)) := by
  sorry

end right_triangle_angle_bisector_segments_l2910_291096


namespace functional_equation_implies_constant_l2910_291040

theorem functional_equation_implies_constant (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_eq : ∀ x y : ℝ, f (x + 2*y) = 2 * f x * f y) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end functional_equation_implies_constant_l2910_291040


namespace homework_difference_is_two_l2910_291035

/-- The number of pages of reading homework Rachel has to complete -/
def reading_pages : ℕ := 2

/-- The number of pages of math homework Rachel has to complete -/
def math_pages : ℕ := 4

/-- The difference in pages between math and reading homework -/
def homework_difference : ℕ := math_pages - reading_pages

theorem homework_difference_is_two : homework_difference = 2 := by
  sorry

end homework_difference_is_two_l2910_291035


namespace inequality_for_positive_integers_l2910_291066

theorem inequality_for_positive_integers (n : ℕ+) :
  (n : ℝ)^(n : ℕ) ≤ ((n : ℕ).factorial : ℝ)^2 ∧ 
  ((n : ℕ).factorial : ℝ)^2 ≤ (((n + 1) * (n + 2) : ℝ) / 6)^(n : ℕ) := by
  sorry

end inequality_for_positive_integers_l2910_291066


namespace hash_one_two_three_l2910_291098

/-- The operation # defined for real numbers a, b, and c -/
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem stating that #(1, 2, 3) = -8 -/
theorem hash_one_two_three : hash 1 2 3 = -8 := by
  sorry

end hash_one_two_three_l2910_291098


namespace circle_center_l2910_291083

/-- The center of the circle with equation x^2 - 10x + y^2 - 4y = -4 is (5, 2) -/
theorem circle_center (x y : ℝ) :
  (x^2 - 10*x + y^2 - 4*y = -4) ↔ ((x - 5)^2 + (y - 2)^2 = 25) :=
by sorry

end circle_center_l2910_291083


namespace problem_1_problem_2_problem_3_problem_4_l2910_291078

-- Problem 1
theorem problem_1 : (-12) + 13 + (-18) + 16 = -1 := by sorry

-- Problem 2
theorem problem_2 : 19.5 + (-6.9) + (-3.1) + (-9.5) = 0 := by sorry

-- Problem 3
theorem problem_3 : (6/5 : ℚ) * (-1/3 - 1/2) / (5/4 : ℚ) = -4/5 := by sorry

-- Problem 4
theorem problem_4 : 18 + 32 * (-1/2)^5 - (1/2)^4 * (-2)^5 = 19 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2910_291078


namespace prism_length_l2910_291077

/-- A regular rectangular prism with given edge sum and proportions -/
structure RegularPrism where
  width : ℝ
  length : ℝ
  height : ℝ
  edge_sum : ℝ
  length_prop : length = 4 * width
  height_prop : height = 3 * width
  sum_prop : 4 * length + 4 * width + 4 * height = edge_sum

/-- The length of a regular rectangular prism with edge sum 256 cm is 32 cm -/
theorem prism_length (p : RegularPrism) (h : p.edge_sum = 256) : p.length = 32 := by
  sorry

end prism_length_l2910_291077


namespace cat_dog_ratio_l2910_291017

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
theorem cat_dog_ratio (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) (num_dogs : ℕ) :
  cat_ratio ≠ 0 ∧ dog_ratio ≠ 0 →
  cat_ratio * num_dogs = dog_ratio * num_cats →
  cat_ratio = 4 ∧ dog_ratio = 5 ∧ num_cats = 24 →
  num_dogs = 30 := by
  sorry

#check cat_dog_ratio

end cat_dog_ratio_l2910_291017


namespace eighth_term_is_25_5_l2910_291031

/-- An arithmetic sequence with 15 terms, first term 3, and last term 48 -/
structure ArithmeticSequence where
  n : ℕ
  a₁ : ℚ
  a₁₅ : ℚ
  h_n : n = 15
  h_a₁ : a₁ = 3
  h_a₁₅ : a₁₅ = 48

/-- The 8th term of the arithmetic sequence is 25.5 -/
theorem eighth_term_is_25_5 (seq : ArithmeticSequence) : 
  let d := (seq.a₁₅ - seq.a₁) / (seq.n - 1)
  seq.a₁ + 7 * d = 25.5 := by
  sorry

end eighth_term_is_25_5_l2910_291031


namespace bridge_length_l2910_291085

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 225 := by
  sorry

end bridge_length_l2910_291085


namespace john_soap_cost_l2910_291009

/-- The amount of money spent on soap given the number of bars, weight per bar, and price per pound -/
def soap_cost (num_bars : ℕ) (weight_per_bar : ℚ) (price_per_pound : ℚ) : ℚ :=
  num_bars * weight_per_bar * price_per_pound

/-- Theorem stating that John spent $15 on soap -/
theorem john_soap_cost : soap_cost 20 (3/2) (1/2) = 15 := by
  sorry

end john_soap_cost_l2910_291009


namespace scalper_ticket_percentage_l2910_291059

theorem scalper_ticket_percentage :
  let normal_price : ℝ := 50
  let website_tickets : ℕ := 2
  let scalper_tickets : ℕ := 2
  let discounted_tickets : ℕ := 1
  let discounted_percentage : ℝ := 60
  let total_paid : ℝ := 360
  let scalper_discount : ℝ := 10

  ∃ P : ℝ,
    website_tickets * normal_price +
    scalper_tickets * (P / 100 * normal_price) - scalper_discount +
    discounted_tickets * (discounted_percentage / 100 * normal_price) = total_paid ∧
    P = 480 :=
by sorry

end scalper_ticket_percentage_l2910_291059


namespace permutation_combination_relation_l2910_291026

-- Define permutation function
def p (n r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

-- Define combination function
def c (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem permutation_combination_relation :
  ∃ k : ℕ, p 32 6 = k * c 32 6 ∧ k = 720 := by
  sorry

end permutation_combination_relation_l2910_291026


namespace temperature_function_correct_and_linear_l2910_291075

/-- Represents the temperature change per kilometer of altitude increase -/
def temperature_change_per_km : ℝ := -6

/-- Represents the ground temperature in Celsius -/
def ground_temperature : ℝ := 20

/-- Represents the temperature y in Celsius at a height of x kilometers above the ground -/
def temperature (x : ℝ) : ℝ := temperature_change_per_km * x + ground_temperature

theorem temperature_function_correct_and_linear :
  (∀ x : ℝ, temperature x = temperature_change_per_km * x + ground_temperature) ∧
  (∃ m b : ℝ, ∀ x : ℝ, temperature x = m * x + b) :=
by sorry

end temperature_function_correct_and_linear_l2910_291075


namespace contractor_absent_days_l2910_291013

/-- Proves that given the specified contract conditions, the number of days absent is 10 -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (payment_per_day : ℚ) 
  (fine_per_day : ℚ) 
  (total_amount : ℚ) : 
  total_days = 30 ∧ 
  payment_per_day = 25 ∧ 
  fine_per_day = 7.5 ∧ 
  total_amount = 425 → 
  ∃ (days_worked : ℕ) (days_absent : ℕ), 
    days_worked + days_absent = total_days ∧ 
    days_absent = 10 ∧
    (payment_per_day * days_worked : ℚ) - (fine_per_day * days_absent : ℚ) = total_amount :=
by sorry

end contractor_absent_days_l2910_291013


namespace ball_probability_after_swap_l2910_291093

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the probability of drawing a ball of a specific color -/
def probability (bag : BagContents) (color : ℕ) : ℚ :=
  color / (bag.red + bag.yellow + bag.blue)

/-- The initial contents of the bag -/
def initialBag : BagContents :=
  { red := 10, yellow := 2, blue := 8 }

/-- The contents of the bag after removing red balls and adding yellow balls -/
def finalBag (n : ℕ) : BagContents :=
  { red := initialBag.red - n, yellow := initialBag.yellow + n, blue := initialBag.blue }

theorem ball_probability_after_swap :
  probability (finalBag 6) (finalBag 6).yellow = 2/5 :=
sorry

end ball_probability_after_swap_l2910_291093


namespace weight_of_BaO_l2910_291043

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of barium oxide (BaO) in g/mol -/
def molecular_weight_BaO : ℝ := atomic_weight_Ba + atomic_weight_O

/-- The number of moles of barium oxide -/
def moles_BaO : ℝ := 6

/-- Theorem: The weight of 6 moles of barium oxide (BaO) is 919.98 grams -/
theorem weight_of_BaO : moles_BaO * molecular_weight_BaO = 919.98 := by
  sorry

end weight_of_BaO_l2910_291043


namespace ceiling_floor_difference_l2910_291010

theorem ceiling_floor_difference : ⌈(15 : ℚ) / 8 * (-34 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-34 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end ceiling_floor_difference_l2910_291010


namespace no_solution_quadratic_inequality_l2910_291065

theorem no_solution_quadratic_inequality :
  ∀ x : ℝ, ¬(x^2 + x ≤ -8) := by
  sorry

end no_solution_quadratic_inequality_l2910_291065


namespace bird_population_theorem_l2910_291053

/-- Represents the bird population in the nature reserve -/
structure BirdPopulation where
  total : ℝ
  hawks : ℝ
  paddyfield_warblers : ℝ
  kingfishers : ℝ

/-- Conditions for the bird population -/
def ValidBirdPopulation (bp : BirdPopulation) : Prop :=
  bp.total > 0 ∧
  bp.hawks = 0.3 * bp.total ∧
  bp.paddyfield_warblers = 0.4 * (bp.total - bp.hawks) ∧
  bp.kingfishers = 0.25 * bp.paddyfield_warblers

/-- Theorem stating the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
theorem bird_population_theorem (bp : BirdPopulation) 
  (h : ValidBirdPopulation bp) : 
  (bp.total - (bp.hawks + bp.paddyfield_warblers + bp.kingfishers)) / bp.total = 0.35 := by
  sorry

end bird_population_theorem_l2910_291053


namespace field_trip_total_l2910_291000

/-- Field trip problem -/
theorem field_trip_total (
  num_vans : ℕ) (num_minibusses : ℕ) (num_coach_buses : ℕ)
  (students_per_van : ℕ) (teachers_per_van : ℕ) (parents_per_van : ℕ)
  (students_per_minibus : ℕ) (teachers_per_minibus : ℕ) (parents_per_minibus : ℕ)
  (students_per_coach : ℕ) (teachers_per_coach : ℕ) (parents_per_coach : ℕ)
  (h1 : num_vans = 6)
  (h2 : num_minibusses = 4)
  (h3 : num_coach_buses = 2)
  (h4 : students_per_van = 10)
  (h5 : teachers_per_van = 2)
  (h6 : parents_per_van = 1)
  (h7 : students_per_minibus = 24)
  (h8 : teachers_per_minibus = 3)
  (h9 : parents_per_minibus = 2)
  (h10 : students_per_coach = 48)
  (h11 : teachers_per_coach = 4)
  (h12 : parents_per_coach = 4) :
  (num_vans * (students_per_van + teachers_per_van + parents_per_van) +
   num_minibusses * (students_per_minibus + teachers_per_minibus + parents_per_minibus) +
   num_coach_buses * (students_per_coach + teachers_per_coach + parents_per_coach)) = 306 := by
  sorry

end field_trip_total_l2910_291000


namespace family_travel_info_l2910_291094

structure FamilyMember where
  name : String
  statement : String

structure TravelInfo where
  origin : String
  destination : String
  stopover : Option String

def father : FamilyMember :=
  { name := "Father", statement := "We are going to Spain (we are coming from Newcastle)." }

def mother : FamilyMember :=
  { name := "Mother", statement := "We are not going to Spain but are coming from Newcastle (we stopped in Paris and are not going to Spain)." }

def daughter : FamilyMember :=
  { name := "Daughter", statement := "We are not coming from Newcastle (we stopped in Paris)." }

def family : List FamilyMember := [father, mother, daughter]

def interpretStatements (family : List FamilyMember) : TravelInfo :=
  { origin := "Newcastle", destination := "", stopover := some "Paris" }

theorem family_travel_info (family : List FamilyMember) :
  interpretStatements family = { origin := "Newcastle", destination := "", stopover := some "Paris" } :=
sorry

end family_travel_info_l2910_291094


namespace curve_transformation_l2910_291076

theorem curve_transformation (x : ℝ) : 
  Real.sin (x + π / 2) = Real.sin (2 * (x + π / 12) + 2 * π / 3) := by
  sorry

end curve_transformation_l2910_291076


namespace modular_inverse_31_mod_45_l2910_291025

theorem modular_inverse_31_mod_45 : ∃ x : ℤ, 0 ≤ x ∧ x < 45 ∧ (31 * x) % 45 = 1 := by
  use 15
  sorry

end modular_inverse_31_mod_45_l2910_291025


namespace curry_house_spicy_curries_l2910_291024

/-- Represents the curry house's pepper buying strategy -/
structure CurryHouse where
  very_spicy_peppers : ℕ := 3
  spicy_peppers : ℕ := 2
  mild_peppers : ℕ := 1
  prev_very_spicy : ℕ := 30
  prev_spicy : ℕ := 30
  prev_mild : ℕ := 10
  new_mild : ℕ := 90
  pepper_reduction : ℕ := 40

/-- Calculates the number of spicy curries the curry house now buys peppers for -/
def calculate_new_spicy_curries (ch : CurryHouse) : ℕ :=
  let prev_total := ch.very_spicy_peppers * ch.prev_very_spicy + 
                    ch.spicy_peppers * ch.prev_spicy + 
                    ch.mild_peppers * ch.prev_mild
  let new_total := prev_total - ch.pepper_reduction
  (new_total - ch.mild_peppers * ch.new_mild) / ch.spicy_peppers

/-- Proves that the curry house now buys peppers for 15 spicy curries -/
theorem curry_house_spicy_curries (ch : CurryHouse) : 
  calculate_new_spicy_curries ch = 15 := by
  sorry

end curry_house_spicy_curries_l2910_291024


namespace big_n_conference_teams_l2910_291061

theorem big_n_conference_teams (n : ℕ) : n * (n - 1) / 2 = 28 → n = 8 := by
  sorry

end big_n_conference_teams_l2910_291061


namespace k_value_l2910_291051

theorem k_value : ∃ k : ℝ, (24 / k = 4) ∧ (k = 6) := by
  sorry

end k_value_l2910_291051


namespace isosceles_triangle_fold_crease_length_l2910_291038

theorem isosceles_triangle_fold_crease_length 
  (a b c : ℝ) (h_isosceles : a = b) (h_sides : a = 5 ∧ c = 6) :
  let m := c / 2
  let crease_length := Real.sqrt (a^2 + m^2)
  crease_length = Real.sqrt 34 := by
sorry

end isosceles_triangle_fold_crease_length_l2910_291038


namespace no_intersection_l2910_291058

theorem no_intersection : ¬∃ x : ℝ, |3*x + 6| = -|4*x - 3| := by
  sorry

end no_intersection_l2910_291058


namespace mother_twice_lucy_age_year_l2910_291029

def lucy_age_2006 : ℕ := 10
def mother_age_2006 : ℕ := 5 * lucy_age_2006

def year_mother_twice_lucy (y : ℕ) : Prop :=
  mother_age_2006 + (y - 2006) = 2 * (lucy_age_2006 + (y - 2006))

theorem mother_twice_lucy_age_year :
  ∃ y : ℕ, y = 2036 ∧ year_mother_twice_lucy y := by sorry

end mother_twice_lucy_age_year_l2910_291029


namespace equal_population_in_17_years_l2910_291055

/-- The number of years it takes for two village populations to be equal -/
def years_to_equal_population (x_initial : ℕ) (x_rate : ℕ) (y_initial : ℕ) (y_rate : ℕ) : ℕ :=
  (x_initial - y_initial) / (y_rate + x_rate)

/-- Theorem: Given the initial populations and rates of change, it takes 17 years for the populations to be equal -/
theorem equal_population_in_17_years :
  years_to_equal_population 76000 1200 42000 800 = 17 := by
  sorry

end equal_population_in_17_years_l2910_291055


namespace screen_area_difference_screen_area_difference_is_152_l2910_291082

/-- The difference in area between two square screens with diagonal lengths 21 and 17 inches -/
theorem screen_area_difference : Int :=
  let screen1_diagonal : Int := 21
  let screen2_diagonal : Int := 17
  let screen1_area : Int := screen1_diagonal ^ 2
  let screen2_area : Int := screen2_diagonal ^ 2
  screen1_area - screen2_area

/-- Proof that the difference in area is 152 square inches -/
theorem screen_area_difference_is_152 : screen_area_difference = 152 := by
  sorry

end screen_area_difference_screen_area_difference_is_152_l2910_291082


namespace total_collection_l2910_291045

def friend_payment : ℕ := 5
def brother_payment : ℕ := 8
def cousin_payment : ℕ := 4
def num_days : ℕ := 7

theorem total_collection :
  (friend_payment * num_days) + (brother_payment * num_days) + (cousin_payment * num_days) = 119 := by
  sorry

end total_collection_l2910_291045


namespace dana_hourly_wage_l2910_291034

/-- Given a person who worked for a certain number of hours and earned a total amount,
    calculate their hourly wage. -/
def hourly_wage (hours_worked : ℕ) (total_earned : ℕ) : ℚ :=
  total_earned / hours_worked

theorem dana_hourly_wage :
  hourly_wage 22 286 = 13 := by sorry

end dana_hourly_wage_l2910_291034


namespace door_ticket_cost_l2910_291052

/-- Proves the cost of tickets purchased at the door given ticket sales information -/
theorem door_ticket_cost
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (advance_ticket_cost : ℕ)
  (advance_tickets_sold : ℕ)
  (h1 : total_tickets = 140)
  (h2 : total_revenue = 1720)
  (h3 : advance_ticket_cost = 8)
  (h4 : advance_tickets_sold = 100) :
  (total_revenue - advance_ticket_cost * advance_tickets_sold) / (total_tickets - advance_tickets_sold) = 23 := by
  sorry


end door_ticket_cost_l2910_291052


namespace complement_union_theorem_l2910_291063

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 4, 5} := by sorry

end complement_union_theorem_l2910_291063


namespace cityD_highest_increase_l2910_291018

structure City where
  name : String
  population1990 : ℕ
  population2000 : ℕ

def percentageIncrease (city : City) : ℚ :=
  (city.population2000 : ℚ) / (city.population1990 : ℚ)

def cityA : City := ⟨"A", 45, 60⟩
def cityB : City := ⟨"B", 65, 85⟩
def cityC : City := ⟨"C", 90, 120⟩
def cityD : City := ⟨"D", 115, 160⟩
def cityE : City := ⟨"E", 150, 200⟩
def cityF : City := ⟨"F", 130, 180⟩

def cities : List City := [cityA, cityB, cityC, cityD, cityE, cityF]

theorem cityD_highest_increase :
  ∀ city ∈ cities, percentageIncrease cityD ≥ percentageIncrease city :=
by sorry

end cityD_highest_increase_l2910_291018


namespace ordered_pairs_satisfying_equation_l2910_291004

theorem ordered_pairs_satisfying_equation : 
  ∃! (n : ℕ), n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      let a := p.1
      let b := p.2
      a > 0 ∧ b > 0 ∧ 
      a * b + 80 = 15 * Nat.lcm a b + 10 * Nat.gcd a b)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 :=
by sorry

end ordered_pairs_satisfying_equation_l2910_291004


namespace root_product_sum_l2910_291049

theorem root_product_sum (p q r : ℂ) : 
  (6 * p^3 - 5 * p^2 + 20 * p - 10 = 0) →
  (6 * q^3 - 5 * q^2 + 20 * q - 10 = 0) →
  (6 * r^3 - 5 * r^2 + 20 * r - 10 = 0) →
  p * q + p * r + q * r = 10 / 3 := by
sorry

end root_product_sum_l2910_291049


namespace ratio_equality_l2910_291056

theorem ratio_equality (x y z : ℝ) 
  (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (x + 4) / 2 = (x + 5) / (z - 5)) : 
  x / y = 1 / 2 := by
sorry

end ratio_equality_l2910_291056


namespace delivery_pay_difference_l2910_291047

/-- Calculate the difference in pay between two workers --/
theorem delivery_pay_difference (deliveries_worker1 : ℕ) 
  (pay_per_delivery : ℕ) : 
  deliveries_worker1 = 96 →
  pay_per_delivery = 100 →
  (deliveries_worker1 * pay_per_delivery : ℕ) - 
  ((deliveries_worker1 * 3 / 4) * pay_per_delivery : ℕ) = 2400 := by
sorry

end delivery_pay_difference_l2910_291047


namespace operation_result_l2910_291008

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def op : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem operation_result :
  op (op Element.three Element.one) (op Element.four Element.two) = Element.three := by
  sorry

end operation_result_l2910_291008


namespace square_sum_theorem_l2910_291001

theorem square_sum_theorem (p q : ℝ) 
  (h1 : p * q = 9)
  (h2 : p^2 * q + q^2 * p + p + q = 70) :
  p^2 + q^2 = 31 := by
  sorry

end square_sum_theorem_l2910_291001


namespace min_hypotenuse_right_triangle_perimeter_6_l2910_291086

/-- The minimum value of the hypotenuse of a right triangle with perimeter 6 -/
theorem min_hypotenuse_right_triangle_perimeter_6 :
  ∃ (c : ℝ), c > 0 ∧ c = 6 * (Real.sqrt 2 - 1) ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + b^2 = c^2 → a + b + c = 6 →
  c ≤ 6 * (Real.sqrt 2 - 1) :=
by sorry

end min_hypotenuse_right_triangle_perimeter_6_l2910_291086


namespace common_root_theorem_l2910_291011

theorem common_root_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0) ∧
  (∃ y : ℝ, b * y^11 + c * y^4 + a = 0 ∧ c * y^11 + a * y^4 + b = 0) ∧
  (∃ z : ℝ, c * z^11 + a * z^4 + b = 0 ∧ a * z^11 + b * z^4 + c = 0) →
  ∃ w : ℝ, a * w^11 + b * w^4 + c = 0 ∧
           b * w^11 + c * w^4 + a = 0 ∧
           c * w^11 + a * w^4 + b = 0 := by
  sorry

end common_root_theorem_l2910_291011


namespace divisor_problem_l2910_291089

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a number is divisible by another -/
def is_divisible_by (a b : ℕ) : Prop := sorry

theorem divisor_problem (n : ℕ+) (k : ℕ) :
  num_divisors n = 72 →
  num_divisors (5 * n) = 120 →
  (∀ m : ℕ, m > k → ¬ is_divisible_by n (5^m)) →
  is_divisible_by n (5^k) →
  k = 0 := by
  sorry

end divisor_problem_l2910_291089


namespace sour_candy_percentage_l2910_291042

theorem sour_candy_percentage (total_candies : ℕ) (num_people : ℕ) (good_candies_per_person : ℕ) :
  total_candies = 300 →
  num_people = 3 →
  good_candies_per_person = 60 →
  (total_candies - num_people * good_candies_per_person) / total_candies = 2/5 :=
by
  sorry

end sour_candy_percentage_l2910_291042


namespace same_value_point_m_two_distinct_same_value_points_l2910_291088

/-- Quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + m

theorem same_value_point_m (m : ℝ) :
  f m 2 = 2 → m = -8 := by sorry

theorem two_distinct_same_value_points (m : ℝ) (a b : ℝ) :
  (∃ (a b : ℝ), a < 1 ∧ 1 < b ∧ f m a = a ∧ f m b = b) →
  m < -3 := by sorry

end same_value_point_m_two_distinct_same_value_points_l2910_291088


namespace boat_current_speed_l2910_291069

/-- Proves that given a boat with a speed of 18 km/hr in still water, traveling downstream
    for 14 minutes and covering a distance of 5.133333333333334 km, the rate of the current
    is 4 km/hr. -/
theorem boat_current_speed
  (boat_speed : ℝ)
  (travel_time : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 18)
  (h2 : travel_time = 14 / 60)
  (h3 : distance = 5.133333333333334) :
  let current_speed := (distance / travel_time) - boat_speed
  current_speed = 4 := by
sorry

end boat_current_speed_l2910_291069


namespace quadratic_roots_and_k_l2910_291036

/-- The quadratic equation x^2 - (k+2)x + 2k - 1 = 0 -/
def quadratic (k x : ℝ) : Prop :=
  x^2 - (k+2)*x + 2*k - 1 = 0

theorem quadratic_roots_and_k :
  (∀ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ quadratic k x ∧ quadratic k y) ∧
  (∃ k : ℝ, quadratic k 3 ∧ quadratic k 1 ∧ k = 2) :=
sorry

end quadratic_roots_and_k_l2910_291036


namespace range_of_a_l2910_291046

open Set

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 1 2, x^2 - a ≥ 0) ∨ 
  (∃ x₀ : ℝ, ∀ x : ℝ, x + (a - 1) * x₀ + 1 < 0) ∧
  ¬((∀ x ∈ Icc 1 2, x^2 - a ≥ 0) ∧ 
    (∃ x₀ : ℝ, ∀ x : ℝ, x + (a - 1) * x₀ + 1 < 0)) →
  a > 3 ∨ a ∈ Icc (-1) 1 :=
by sorry

#check range_of_a

end range_of_a_l2910_291046


namespace t_plus_inverse_t_l2910_291030

theorem t_plus_inverse_t (t : ℝ) (h1 : t^2 - 3*t + 1 = 0) (h2 : t ≠ 0) : 
  t + 1/t = 3 := by
  sorry

end t_plus_inverse_t_l2910_291030


namespace equation_solution_l2910_291084

theorem equation_solution (n : ℤ) : 
  (5 : ℚ) / 4 * n + (5 : ℚ) / 4 = n ↔ ∃ k : ℤ, n = -5 + 1024 * k := by
  sorry

end equation_solution_l2910_291084


namespace polynomial_division_l2910_291020

/-- The dividend polynomial -/
def dividend (x : ℚ) : ℚ := 9*x^4 + 27*x^3 - 8*x^2 + 8*x + 5

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 3*x + 4

/-- The quotient polynomial -/
def quotient (x : ℚ) : ℚ := 3*x^3 + 5*x^2 - (28/3)*x + 136/9

/-- The remainder -/
def remainder : ℚ := 5 - 544/9

theorem polynomial_division :
  ∀ x, dividend x = divisor x * quotient x + remainder := by
  sorry

end polynomial_division_l2910_291020


namespace min_cubes_required_l2910_291012

-- Define the dimensions of the box
def box_length : ℕ := 9
def box_width : ℕ := 12
def box_height : ℕ := 3

-- Define the volume of a single cube
def cube_volume : ℕ := 3

-- Theorem: The minimum number of cubes required is 108
theorem min_cubes_required : 
  (box_length * box_width * box_height) / cube_volume = 108 := by
  sorry

end min_cubes_required_l2910_291012


namespace quadratic_root_zero_l2910_291002

/-- Given a quadratic equation (k+2)x^2 + 6x + k^2 + k - 2 = 0 where one of its roots is 0,
    prove that k = 1 -/
theorem quadratic_root_zero (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0) ∧ 
  ((k + 2) * 0^2 + 6 * 0 + k^2 + k - 2 = 0) →
  k = 1 :=
by sorry

end quadratic_root_zero_l2910_291002


namespace smallest_k_for_64_power_l2910_291019

theorem smallest_k_for_64_power (k : ℕ) (some_exponent : ℕ) : k = 6 → some_exponent < 18 → 64^k > 4^some_exponent := by
  sorry

end smallest_k_for_64_power_l2910_291019


namespace multiply_by_99999_l2910_291067

theorem multiply_by_99999 (x : ℝ) : x * 99999 = 58293485180 → x = 582.935 := by
  sorry

end multiply_by_99999_l2910_291067


namespace real_number_inequalities_l2910_291015

theorem real_number_inequalities (a b c : ℝ) : 
  (a > b → a > (a + b) / 2 ∧ (a + b) / 2 > b) ∧
  (a > b ∧ b > 0 → a > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > b) ∧
  (a > b ∧ b > 0 ∧ c > 0 → (b + c) / (a + c) > b / a) :=
by sorry

end real_number_inequalities_l2910_291015


namespace max_projection_area_parallelepiped_l2910_291022

/-- The maximum area of the projection of a rectangular parallelepiped with edge lengths √70, √99, and √126 onto any plane is 168. -/
theorem max_projection_area_parallelepiped :
  let a := Real.sqrt 70
  let b := Real.sqrt 99
  let c := Real.sqrt 126
  ∃ (proj : ℝ → ℝ → ℝ → ℝ), 
    (∀ x y z, proj x y z ≤ 168) ∧ 
    (∃ x y z, proj x y z = 168) :=
by sorry

end max_projection_area_parallelepiped_l2910_291022


namespace lobachevsky_angle_existence_l2910_291092

theorem lobachevsky_angle_existence (A B C : Real) 
  (hB : 0 < B ∧ B < Real.pi / 2) 
  (hC : 0 < C ∧ C < Real.pi / 2) : 
  ∃ X, Real.sin X = (Real.sin B * Real.sin C) / (1 - Real.cos A * Real.cos B * Real.cos C) := by
  sorry

end lobachevsky_angle_existence_l2910_291092


namespace square_formation_total_l2910_291016

/-- Given a square formation of people where one person is the 5th from each side,
    prove that the total number of people is 81. -/
theorem square_formation_total (n : ℕ) (h : n = 5) :
  (2 * n - 1) * (2 * n - 1) = 81 := by
  sorry

end square_formation_total_l2910_291016


namespace firefly_count_l2910_291074

/-- The number of fireflies remaining after a series of events --/
def remaining_fireflies (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the number of remaining fireflies in the given scenario --/
theorem firefly_count : remaining_fireflies 3 8 2 = 9 := by
  sorry

end firefly_count_l2910_291074


namespace solution_set_f_plus_x_squared_range_of_m_l2910_291033

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (m : ℝ) (x : ℝ) : ℝ := -|x + 3| + m

-- Theorem 1: Solution set of |x-1| + x^2 - 1 > 0
theorem solution_set_f_plus_x_squared (x : ℝ) : 
  (|x - 1| + x^2 - 1 > 0) ↔ (x > 1 ∨ x < 0) := by sorry

-- Theorem 2: If f(x) < g(x) has a non-empty solution set, then m > 4
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < g m x) → m > 4 := by sorry

end solution_set_f_plus_x_squared_range_of_m_l2910_291033


namespace simplify_sqrt_sum_l2910_291057

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 3) + Real.sqrt (8 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end simplify_sqrt_sum_l2910_291057


namespace quadratic_symmetry_l2910_291039

/-- A quadratic function with specific properties -/
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^2 + B * x + C

/-- Theorem: For a quadratic function p(x) with axis of symmetry at x = 3.5 and p(0) = 2, p(20) = 2 -/
theorem quadratic_symmetry (A B C : ℝ) :
  (∀ x : ℝ, p A B C (3.5 + x) = p A B C (3.5 - x)) →  -- Axis of symmetry at x = 3.5
  p A B C 0 = 2 →                                     -- p(0) = 2
  p A B C 20 = 2 :=                                   -- Conclusion: p(20) = 2
by
  sorry


end quadratic_symmetry_l2910_291039


namespace book_arrangement_l2910_291091

theorem book_arrangement (n m : ℕ) (hn : n = 3) (hm : m = 4) :
  (Nat.choose (n + m) n) = 35 := by
  sorry

end book_arrangement_l2910_291091


namespace hole_empty_time_l2910_291054

/-- Given a pipe that can fill a tank in 15 hours, and with a hole causing
    the tank to fill in 20 hours instead, prove that the time it takes for
    the hole to empty a full tank is 60 hours. -/
theorem hole_empty_time (fill_time_no_hole fill_time_with_hole : ℝ)
  (h1 : fill_time_no_hole = 15)
  (h2 : fill_time_with_hole = 20) :
  (fill_time_no_hole * fill_time_with_hole) /
    (fill_time_with_hole - fill_time_no_hole) = 60 := by
  sorry

end hole_empty_time_l2910_291054


namespace banana_arrangements_l2910_291044

def banana_length : ℕ := 6
def num_a : ℕ := 3
def num_n : ℕ := 2

theorem banana_arrangements : 
  (banana_length.factorial) / (num_a.factorial * num_n.factorial) = 60 := by
  sorry

end banana_arrangements_l2910_291044


namespace paper_folding_thickness_l2910_291070

/-- The thickness of the paper after n folds -/
def thickness (n : ℕ) : ℚ := (1 / 10) * 2^n

/-- The minimum number of folds required to exceed 12mm -/
def min_folds : ℕ := 7

theorem paper_folding_thickness :
  (∀ k < min_folds, thickness k ≤ 12) ∧ thickness min_folds > 12 := by
  sorry

end paper_folding_thickness_l2910_291070


namespace one_nonnegative_solution_for_quadratic_l2910_291028

theorem one_nonnegative_solution_for_quadratic :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -5*x := by sorry

end one_nonnegative_solution_for_quadratic_l2910_291028


namespace midpoint_ordinate_l2910_291027

theorem midpoint_ordinate (a : Real) (h1 : 0 < a) (h2 : a < π / 2) :
  let P : Real × Real := (a, Real.sin a)
  let Q : Real × Real := (a, Real.cos a)
  let distance := |P.2 - Q.2|
  let midpoint_y := (P.2 + Q.2) / 2
  distance = 1/4 → midpoint_y = Real.sqrt 31 / 8 := by
  sorry

end midpoint_ordinate_l2910_291027


namespace union_equal_iff_x_zero_l2910_291023

def A (x : ℝ) : Set ℝ := {0, Real.exp x}
def B : Set ℝ := {-1, 0, 1}

theorem union_equal_iff_x_zero (x : ℝ) : A x ∪ B = B ↔ x = 0 := by
  sorry

end union_equal_iff_x_zero_l2910_291023


namespace bucket_fill_lcm_l2910_291003

/-- Time to fill bucket A completely -/
def time_A : ℕ := 135

/-- Time to fill bucket B completely -/
def time_B : ℕ := 240

/-- Time to fill bucket C completely -/
def time_C : ℕ := 200

/-- Function to calculate the least common multiple of three natural numbers -/
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem bucket_fill_lcm :
  (2 * time_A = 3 * 90) ∧
  (time_B = 2 * 120) ∧
  (3 * time_C = 4 * 150) →
  lcm_three time_A time_B time_C = 1200 := by
  sorry

end bucket_fill_lcm_l2910_291003


namespace crayons_per_pack_l2910_291007

theorem crayons_per_pack (total_crayons : ℕ) (num_packs : ℕ) 
  (h1 : total_crayons = 615) (h2 : num_packs = 41) :
  total_crayons / num_packs = 15 := by
  sorry

end crayons_per_pack_l2910_291007


namespace yoongi_has_smallest_number_l2910_291064

def jungkook_number : ℕ := 6 * 3
def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5

theorem yoongi_has_smallest_number : 
  yoongi_number ≤ jungkook_number ∧ yoongi_number ≤ yuna_number :=
sorry

end yoongi_has_smallest_number_l2910_291064


namespace system_of_equations_solution_l2910_291041

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (4 * x - 7 * y = -20) ∧ 
    (9 * x + 3 * y = -21) ∧ 
    (x = -69/25) ∧ 
    (y = 32/25) := by
  sorry

end system_of_equations_solution_l2910_291041


namespace sqrt_529_squared_l2910_291087

theorem sqrt_529_squared : (Real.sqrt 529)^2 = 529 := by
  sorry

end sqrt_529_squared_l2910_291087


namespace unique_valid_number_l2910_291050

/-- A function that returns the digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a 3-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The set of 3-digit numbers with digit sum 25 that are even -/
def validNumbers : Set ℕ := {n : ℕ | isThreeDigit n ∧ digitSum n = 25 ∧ Even n}

theorem unique_valid_number : ∃! n : ℕ, n ∈ validNumbers := by sorry

end unique_valid_number_l2910_291050


namespace nails_per_plank_l2910_291014

theorem nails_per_plank (total_planks : ℕ) (total_nails : ℕ) 
  (h1 : total_planks = 16) (h2 : total_nails = 32) : 
  total_nails / total_planks = 2 := by
  sorry

end nails_per_plank_l2910_291014


namespace seed_germination_percentage_l2910_291072

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot2 overall_germination_rate : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot2 = 35 / 100 →
  overall_germination_rate = 26 / 100 →
  ∃ (germination_rate_plot1 : ℚ),
    germination_rate_plot1 = 20 / 100 ∧
    germination_rate_plot1 * seeds_plot1 + germination_rate_plot2 * seeds_plot2 = 
    overall_germination_rate * (seeds_plot1 + seeds_plot2) := by
  sorry

end seed_germination_percentage_l2910_291072


namespace debate_schedule_ways_l2910_291090

/-- Number of debaters from each school -/
def num_debaters : ℕ := 4

/-- Total number of debates -/
def total_debates : ℕ := num_debaters * num_debaters

/-- Maximum number of debates per session -/
def max_debates_per_session : ℕ := 3

/-- Number of ways to schedule debates -/
def schedule_ways : ℕ := 20922789888000

/-- Theorem stating the number of ways to schedule debates -/
theorem debate_schedule_ways :
  (total_debates.factorial) / (max_debates_per_session.factorial ^ 5 * 1) = schedule_ways := by
  sorry

end debate_schedule_ways_l2910_291090


namespace square_difference_equality_l2910_291071

theorem square_difference_equality : (45 + 18)^2 - (45^2 + 18^2 + 10) = 1610 := by
  sorry

end square_difference_equality_l2910_291071


namespace exponential_grows_faster_than_quadratic_l2910_291006

theorem exponential_grows_faster_than_quadratic : 
  ∀ ε > 0, ∃ x₀ > 0, ∀ x > x₀, (2:ℝ)^x > x^2 + ε := by
  sorry

end exponential_grows_faster_than_quadratic_l2910_291006
