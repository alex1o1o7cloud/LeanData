import Mathlib

namespace spring_compression_l447_44787

/-- The force-distance relationship for a spring -/
def spring_force (s : ℝ) : ℝ := 16 * s^2

/-- Theorem: When a force of 4 newtons is applied, the spring compresses by 0.5 meters -/
theorem spring_compression :
  spring_force 0.5 = 4 := by sorry

end spring_compression_l447_44787


namespace jamie_max_correct_answers_l447_44771

theorem jamie_max_correct_answers
  (total_questions : ℕ)
  (correct_points : ℤ)
  (blank_points : ℤ)
  (incorrect_points : ℤ)
  (total_score : ℤ)
  (h1 : total_questions = 60)
  (h2 : correct_points = 5)
  (h3 : blank_points = 0)
  (h4 : incorrect_points = -2)
  (h5 : total_score = 150) :
  ∃ (x : ℕ), x ≤ 38 ∧
    ∀ (y : ℕ), y > 38 →
      ¬∃ (blank incorrect : ℕ),
        y + blank + incorrect = total_questions ∧
        y * correct_points + blank * blank_points + incorrect * incorrect_points = total_score :=
by sorry

end jamie_max_correct_answers_l447_44771


namespace seating_arrangements_l447_44799

def total_people : ℕ := 10
def restricted_group : ℕ := 4

theorem seating_arrangements (total : ℕ) (restricted : ℕ) :
  total = total_people ∧ restricted = restricted_group →
  (total.factorial - (total - restricted + 1).factorial * restricted.factorial) = 3507840 :=
by sorry

end seating_arrangements_l447_44799


namespace range_of_a_part1_range_of_a_part2_l447_44711

def proposition_p (a : ℝ) : Prop := a^2 - 5*a - 6 > 0

def proposition_q (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^2 + a*x₁ + 1 = 0 ∧ x₂^2 + a*x₂ + 1 = 0

theorem range_of_a_part1 :
  {a : ℝ | proposition_p a} = {a | a < -1 ∨ a > 6} :=
sorry

theorem range_of_a_part2 :
  {a : ℝ | (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a)} =
  {a | a < -1 ∨ (2 < a ∧ a ≤ 6)} :=
sorry

end range_of_a_part1_range_of_a_part2_l447_44711


namespace largest_angle_obtuse_triangle_l447_44777

/-- Given an obtuse, scalene triangle ABC with angle A measuring 30 degrees and angle B measuring 55 degrees,
    the measure of the largest interior angle is 95 degrees. -/
theorem largest_angle_obtuse_triangle (A B C : ℝ) (h_obtuse : A + B + C = 180) 
  (h_A : A = 30) (h_B : B = 55) (h_scalene : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  max A (max B C) = 95 := by
  sorry

end largest_angle_obtuse_triangle_l447_44777


namespace first_group_size_l447_44747

/-- The number of people in the first group -/
def P : ℕ := sorry

/-- The amount of work that can be completed by the first group in 3 days -/
def W₁ : ℕ := 3

/-- The number of days it takes the first group to complete W₁ amount of work -/
def D₁ : ℕ := 3

/-- The amount of work that can be completed by 4 people in 3 days -/
def W₂ : ℕ := 4

/-- The number of people in the second group -/
def P₂ : ℕ := 4

/-- The number of days it takes the second group to complete W₂ amount of work -/
def D₂ : ℕ := 3

/-- The theorem stating that the number of people in the first group is 3 -/
theorem first_group_size :
  (P * W₂ * D₁ = P₂ * W₁ * D₂) → P = 3 := by sorry

end first_group_size_l447_44747


namespace age_ratio_change_l447_44784

/-- Proves the number of years it takes for a parent to become 2.5 times as old as their son -/
theorem age_ratio_change (parent_age son_age : ℕ) (x : ℕ) 
  (h1 : parent_age = 45)
  (h2 : son_age = 15)
  (h3 : parent_age = 3 * son_age) :
  (parent_age + x) = (5/2 : ℚ) * (son_age + x) ↔ x = 5 := by
  sorry

end age_ratio_change_l447_44784


namespace commission_percentage_problem_l447_44720

/-- Calculates the commission percentage given the commission amount and total sales. -/
def commission_percentage (commission : ℚ) (total_sales : ℚ) : ℚ :=
  (commission / total_sales) * 100

/-- Theorem stating that for the given commission and sales values, the commission percentage is 4%. -/
theorem commission_percentage_problem :
  let commission : ℚ := 25/2  -- Rs. 12.50
  let total_sales : ℚ := 625/2  -- Rs. 312.5
  commission_percentage commission total_sales = 4 := by
  sorry

end commission_percentage_problem_l447_44720


namespace inverse_proportion_inequality_l447_44729

theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < 0 → 0 < x₂ → y₁ = 6 / x₁ → y₂ = 6 / x₂ → y₁ < y₂ := by
  sorry

end inverse_proportion_inequality_l447_44729


namespace michael_money_ratio_l447_44782

/-- Given the initial conditions and final state of a money transfer between Michael and his brother,
    prove that the ratio of the money Michael gave to his brother to his initial amount is 1/2. -/
theorem michael_money_ratio :
  ∀ (michael_initial brother_initial michael_final brother_final transfer candy : ℕ),
    michael_initial = 42 →
    brother_initial = 17 →
    brother_final = 35 →
    candy = 3 →
    michael_final + transfer = michael_initial →
    brother_final + candy = brother_initial + transfer →
    (transfer : ℚ) / michael_initial = 1 / 2 := by
  sorry

end michael_money_ratio_l447_44782


namespace brownie_theorem_l447_44703

/-- The number of brownie pieces obtained from a rectangular tray -/
def brownie_pieces (tray_length tray_width piece_length piece_width : ℕ) : ℕ :=
  (tray_length * tray_width) / (piece_length * piece_width)

/-- Theorem stating that a 24-inch by 30-inch tray yields 60 brownie pieces of size 3 inches by 4 inches -/
theorem brownie_theorem :
  brownie_pieces 24 30 3 4 = 60 := by
  sorry

end brownie_theorem_l447_44703


namespace twelve_million_plus_twelve_thousand_l447_44794

theorem twelve_million_plus_twelve_thousand : 
  12000000 + 12000 = 12012000 := by
  sorry

end twelve_million_plus_twelve_thousand_l447_44794


namespace union_equals_interval_l447_44796

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

-- Define the interval [-1, 4]
def interval : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4}

-- Theorem statement
theorem union_equals_interval : A ∪ B = interval := by
  sorry

end union_equals_interval_l447_44796


namespace square_difference_of_integers_l447_44791

theorem square_difference_of_integers (a b : ℕ+) 
  (sum_eq : a + b = 70) 
  (diff_eq : a - b = 20) : 
  a ^ 2 - b ^ 2 = 1400 := by
sorry

end square_difference_of_integers_l447_44791


namespace add_fractions_l447_44709

theorem add_fractions : (3 : ℚ) / 4 + (5 : ℚ) / 6 = (19 : ℚ) / 12 := by sorry

end add_fractions_l447_44709


namespace labor_cost_calculation_l447_44795

def cost_of_seeds : ℝ := 50
def cost_of_fertilizers_and_pesticides : ℝ := 35
def number_of_bags : ℕ := 10
def price_per_bag : ℝ := 11
def profit_percentage : ℝ := 0.1

theorem labor_cost_calculation (labor_cost : ℝ) : 
  (cost_of_seeds + cost_of_fertilizers_and_pesticides + labor_cost) * (1 + profit_percentage) = 
  (number_of_bags : ℝ) * price_per_bag → 
  labor_cost = 15 := by
sorry

end labor_cost_calculation_l447_44795


namespace sci_fi_readers_l447_44786

theorem sci_fi_readers (total : ℕ) (literary : ℕ) (both : ℕ) : 
  total = 650 → literary = 550 → both = 150 → 
  total = literary + (total - literary + both) - both :=
by
  sorry

#check sci_fi_readers

end sci_fi_readers_l447_44786


namespace p_minus_q_value_l447_44713

theorem p_minus_q_value (p q : ℚ) (hp : 3 / p = 4) (hq : 3 / q = 18) : p - q = 7 / 12 := by
  sorry

end p_minus_q_value_l447_44713


namespace fifteen_guests_four_rooms_l447_44741

/-- The number of ways to distribute n guests into k rooms such that no room is empty. -/
def distributeGuests (n k : ℕ) : ℕ :=
  (k^n : ℕ) - k * ((k-1)^n : ℕ) + (k.choose 2) * ((k-2)^n : ℕ) - (k.choose 3) * ((k-3)^n : ℕ)

/-- Theorem stating that the number of ways to distribute 15 guests into 4 rooms
    such that no room is empty is equal to 4^15 - 4 * 3^15 + 6 * 2^15 - 4. -/
theorem fifteen_guests_four_rooms :
  distributeGuests 15 4 = 4^15 - 4 * 3^15 + 6 * 2^15 - 4 := by
  sorry

#eval distributeGuests 15 4

end fifteen_guests_four_rooms_l447_44741


namespace garage_sale_items_l447_44716

theorem garage_sale_items (prices : Finset ℕ) (radio_price : ℕ) : 
  prices.card = 43 ∧ radio_price ∈ prices ∧ 
  (prices.filter (λ x => x > radio_price)).card = 8 ∧
  (prices.filter (λ x => x < radio_price)).card = 34 →
  prices.card = 43 :=
by sorry

end garage_sale_items_l447_44716


namespace simplify_and_evaluate_l447_44769

theorem simplify_and_evaluate (a : ℝ) (h : a = 2) : 
  (1 - 1 / (a + 1)) / (a / (a^2 - 1)) = 1 := by
  sorry

end simplify_and_evaluate_l447_44769


namespace two_statements_incorrect_l447_44783

-- Define a type for geometric statements
inductive GeometricStatement
  | ParallelogramOppositeAngles
  | PolygonExteriorAngles
  | TriangleRotation
  | AngleMagnification
  | CircleCircumferenceRadiusRatio
  | CircleCircumferenceAreaRatio

-- Define a function to check if a statement is correct
def isCorrect (s : GeometricStatement) : Bool :=
  match s with
  | .ParallelogramOppositeAngles => false
  | .PolygonExteriorAngles => true
  | .TriangleRotation => true
  | .AngleMagnification => true
  | .CircleCircumferenceRadiusRatio => true
  | .CircleCircumferenceAreaRatio => false

-- Define the list of all statements
def allStatements : List GeometricStatement :=
  [.ParallelogramOppositeAngles, .PolygonExteriorAngles, .TriangleRotation,
   .AngleMagnification, .CircleCircumferenceRadiusRatio, .CircleCircumferenceAreaRatio]

-- Theorem: Exactly 2 out of 6 statements are incorrect
theorem two_statements_incorrect :
  (allStatements.filter (fun s => ¬(isCorrect s))).length = 2 := by
  sorry


end two_statements_incorrect_l447_44783


namespace right_triangle_hypotenuse_l447_44748

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 45 → b = 60 → c^2 = a^2 + b^2 → c = 75 := by
  sorry

end right_triangle_hypotenuse_l447_44748


namespace position_of_2013_l447_44772

/-- Represents the position of a number in the arrangement -/
structure Position where
  row : Nat
  column : Nat
  deriving Repr

/-- Calculates the position of a given odd number in the arrangement -/
def position_of_odd_number (n : Nat) : Position :=
  sorry

theorem position_of_2013 : position_of_odd_number 2013 = ⟨45, 17⟩ := by
  sorry

end position_of_2013_l447_44772


namespace closest_cube_root_to_50_l447_44714

theorem closest_cube_root_to_50 :
  ∀ n : ℤ, |((2:ℝ)^n)^(1/3) - 50| ≥ |((2:ℝ)^17)^(1/3) - 50| :=
by sorry

end closest_cube_root_to_50_l447_44714


namespace binomial_60_3_l447_44710

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l447_44710


namespace salary_comparison_l447_44788

def hansel_initial : ℕ := 30000
def hansel_raise : ℚ := 10 / 100

def gretel_initial : ℕ := 30000
def gretel_raise : ℚ := 15 / 100

def rapunzel_initial : ℕ := 40000
def rapunzel_raise : ℚ := 8 / 100

def rumpelstiltskin_initial : ℕ := 35000
def rumpelstiltskin_raise : ℚ := 12 / 100

def new_salary (initial : ℕ) (raise : ℚ) : ℚ :=
  initial * (1 + raise)

theorem salary_comparison :
  (new_salary gretel_initial gretel_raise - new_salary hansel_initial hansel_raise = 1500) ∧
  (new_salary gretel_initial gretel_raise < new_salary rapunzel_initial rapunzel_raise) ∧
  (new_salary gretel_initial gretel_raise < new_salary rumpelstiltskin_initial rumpelstiltskin_raise) :=
by sorry

end salary_comparison_l447_44788


namespace four_digit_sum_1989_l447_44712

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- Apply the digit sum transformation n times -/
def iterateDigitSum (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterateDigitSum (digitSum n) k

/-- The main theorem stating that applying digit sum 4 times to 1989 results in 9 -/
theorem four_digit_sum_1989 : iterateDigitSum 1989 4 = 9 := by
  sorry

end four_digit_sum_1989_l447_44712


namespace sheep_wandered_off_percentage_l447_44760

theorem sheep_wandered_off_percentage 
  (total_sheep : ℕ) 
  (rounded_up_percentage : ℚ) 
  (sheep_in_pen : ℕ) 
  (sheep_in_wilderness : ℕ) 
  (h1 : rounded_up_percentage = 90 / 100) 
  (h2 : sheep_in_pen = 81) 
  (h3 : sheep_in_wilderness = 9) 
  (h4 : ↑sheep_in_pen = rounded_up_percentage * ↑total_sheep) 
  (h5 : total_sheep = sheep_in_pen + sheep_in_wilderness) : 
  (↑sheep_in_wilderness / ↑total_sheep) * 100 = 10 := by
sorry

end sheep_wandered_off_percentage_l447_44760


namespace solve_cupcake_problem_l447_44738

def cupcake_problem (initial_cupcakes : ℕ) (sold_cupcakes : ℕ) (final_cupcakes : ℕ) : Prop :=
  initial_cupcakes - sold_cupcakes + (final_cupcakes - (initial_cupcakes - sold_cupcakes)) = 20

theorem solve_cupcake_problem :
  cupcake_problem 26 20 26 := by
  sorry

end solve_cupcake_problem_l447_44738


namespace maggies_total_earnings_l447_44743

/-- Maggie's earnings from selling magazine subscriptions --/
def maggies_earnings (price_per_subscription : ℕ) 
  (parents_subscriptions : ℕ) 
  (grandfather_subscriptions : ℕ) 
  (next_door_neighbor_subscriptions : ℕ) : ℕ :=
  let total_subscriptions := 
    parents_subscriptions + 
    grandfather_subscriptions + 
    next_door_neighbor_subscriptions + 
    (2 * next_door_neighbor_subscriptions)
  total_subscriptions * price_per_subscription

/-- Theorem stating Maggie's earnings --/
theorem maggies_total_earnings : 
  maggies_earnings 5 4 1 2 = 55 := by
  sorry

end maggies_total_earnings_l447_44743


namespace beans_remaining_fraction_l447_44721

/-- The fraction of beans remaining in a jar after some have been removed -/
theorem beans_remaining_fraction (jar_weight : ℝ) (full_beans_weight : ℝ) 
  (h1 : jar_weight = 0.1 * (jar_weight + full_beans_weight))
  (h2 : ∃ remaining_beans : ℝ, jar_weight + remaining_beans = 0.6 * (jar_weight + full_beans_weight)) :
  ∃ remaining_beans : ℝ, remaining_beans / full_beans_weight = 5 / 9 := by
  sorry

end beans_remaining_fraction_l447_44721


namespace legacy_gold_bars_l447_44708

/-- The number of gold bars Legacy has -/
def legacy_bars : ℕ := sorry

/-- The number of gold bars Aleena has -/
def aleena_bars : ℕ := legacy_bars - 2

/-- The value of one gold bar in dollars -/
def bar_value : ℕ := 2200

/-- The total value of gold bars Legacy and Aleena have together -/
def total_value : ℕ := 17600

theorem legacy_gold_bars :
  legacy_bars = 5 ∧
  aleena_bars = legacy_bars - 2 ∧
  bar_value = 2200 ∧
  total_value = 17600 ∧
  total_value = bar_value * (legacy_bars + aleena_bars) :=
sorry

end legacy_gold_bars_l447_44708


namespace kola_solution_water_percentage_l447_44793

theorem kola_solution_water_percentage
  (initial_volume : ℝ)
  (initial_kola_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (final_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_kola_percentage = 5)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 10)
  (h5 : added_kola = 6.8)
  (h6 : final_sugar_percentage = 7.5)
  : ∃ (initial_water_percentage : ℝ),
    initial_water_percentage = 88 ∧
    initial_water_percentage + 
      (100 - initial_water_percentage - initial_kola_percentage) + 
      initial_kola_percentage = 100 ∧
    (100 - initial_water_percentage - initial_kola_percentage) / 100 * initial_volume + added_sugar = 
      final_sugar_percentage / 100 * (initial_volume + added_sugar + added_water + added_kola) :=
by sorry

end kola_solution_water_percentage_l447_44793


namespace emilys_number_proof_l447_44789

theorem emilys_number_proof :
  ∃! n : ℕ, 
    (216 ∣ n) ∧ 
    (45 ∣ n) ∧ 
    (1000 < n) ∧ 
    (n < 3000) ∧ 
    (n = 2160) := by
  sorry

end emilys_number_proof_l447_44789


namespace three_intersecting_lines_l447_44764

/-- The parabola defined by y² = 3x -/
def parabola (x y : ℝ) : Prop := y^2 = 3*x

/-- A point lies on a line through (0, 2) -/
def line_through_A (m : ℝ) (x y : ℝ) : Prop := y = m*x + 2

/-- A line intersects the parabola at exactly one point -/
def single_intersection (m : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line_through_A m p.1 p.2

/-- There are exactly 3 lines through (0, 2) that intersect the parabola at one point -/
theorem three_intersecting_lines : ∃! l : Finset ℝ, 
  l.card = 3 ∧ (∀ m ∈ l, single_intersection m) ∧
  (∀ m : ℝ, single_intersection m → m ∈ l) :=
sorry

end three_intersecting_lines_l447_44764


namespace inequality_properties_l447_44756

theorem inequality_properties (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) (h3 : d > 0) : 
  (a - d > b - c) ∧ 
  (a * c^2 > b * c^2) ∧
  (∃ a b c d : ℝ, a > b ∧ c > d ∧ d > 0 ∧ a * c ≤ b * d) ∧
  (∃ a b c d : ℝ, a > b ∧ c > d ∧ d > 0 ∧ a / d ≤ b / c) :=
by sorry

end inequality_properties_l447_44756


namespace bicycle_price_increase_l447_44742

theorem bicycle_price_increase (P : ℝ) : 
  (P * 1.15 = 253) → P = 220 := by
  sorry

end bicycle_price_increase_l447_44742


namespace factor_expression_l447_44766

theorem factor_expression (y : ℝ) : 75 * y + 45 = 15 * (5 * y + 3) := by
  sorry

end factor_expression_l447_44766


namespace parallelogram_base_length_l447_44775

/-- Given a parallelogram with area 98 sq m and altitude twice the base, prove the base is 7 m -/
theorem parallelogram_base_length : 
  ∀ (base altitude : ℝ), 
  (base * altitude = 98) →  -- Area of parallelogram
  (altitude = 2 * base) →   -- Altitude is twice the base
  base = 7 := by
sorry

end parallelogram_base_length_l447_44775


namespace sin_equality_condition_l447_44744

theorem sin_equality_condition :
  (∀ A B : ℝ, A = B → Real.sin A = Real.sin B) ∧
  (∃ A B : ℝ, Real.sin A = Real.sin B ∧ A ≠ B) := by
  sorry

end sin_equality_condition_l447_44744


namespace second_grade_volunteers_l447_44778

/-- Given a total population and a subgroup, calculate the proportion of volunteers
    to be selected from the subgroup in a stratified random sampling. -/
def stratified_sampling_proportion (total_population : ℕ) (subgroup : ℕ) (total_volunteers : ℕ) : ℕ :=
  (subgroup * total_volunteers) / total_population

/-- Prove that in a stratified random sampling of 30 volunteers from a population of 3000 students,
    where 1000 students are in the second grade, the number of volunteers to be selected from
    the second grade is 10. -/
theorem second_grade_volunteers :
  stratified_sampling_proportion 3000 1000 30 = 10 := by
  sorry

end second_grade_volunteers_l447_44778


namespace alexey_min_banks_l447_44768

/-- The minimum number of banks needed to fully insure a given amount of money -/
def min_banks (total_amount : ℕ) (max_payout : ℕ) : ℕ :=
  (total_amount + max_payout - 1) / max_payout

/-- Theorem stating the minimum number of banks needed for Alexey's case -/
theorem alexey_min_banks :
  min_banks 10000000 1400000 = 8 := by
  sorry

end alexey_min_banks_l447_44768


namespace margin_in_terms_of_selling_price_l447_44763

theorem margin_in_terms_of_selling_price 
  (n : ℝ) (C S M : ℝ) 
  (h1 : M = (2/n) * C) 
  (h2 : S = C + M) : 
  M = (2/(n+2)) * S := by
sorry

end margin_in_terms_of_selling_price_l447_44763


namespace correct_systematic_sampling_l447_44750

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Generates a sample based on the systematic sampling scheme. -/
def generate_sample (s : SystematicSampling) : List ℕ :=
  List.range s.sample_size |>.map (λ i => s.start + i * s.interval)

/-- The theorem to be proved. -/
theorem correct_systematic_sampling :
  let s : SystematicSampling := {
    population_size := 60,
    sample_size := 6,
    start := 3,
    interval := 10
  }
  generate_sample s = [3, 13, 23, 33, 43, 53] :=
by
  sorry


end correct_systematic_sampling_l447_44750


namespace remainder_97_37_mod_100_l447_44759

theorem remainder_97_37_mod_100 : 97^37 % 100 = 77 := by
  sorry

end remainder_97_37_mod_100_l447_44759


namespace ellipse_constant_product_l447_44734

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line passing through (1, 0) with slope k -/
def line (x y k : ℝ) : Prop := y = k * (x - 1)

/-- The dot product of vectors PE and QE -/
def dot_product (xP yP xQ yQ xE : ℝ) : ℝ :=
  (xE - xP) * (xE - xQ) + (-yP) * (-yQ)

theorem ellipse_constant_product :
  ∀ (xP yP xQ yQ k : ℝ),
    ellipse xP yP →
    ellipse xQ yQ →
    line xP yP k →
    line xQ yQ k →
    xP ≠ xQ →
    dot_product xP yP xQ yQ (17/8) = 33/64 := by sorry

end ellipse_constant_product_l447_44734


namespace four_digit_divisible_count_l447_44757

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_all (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0

theorem four_digit_divisible_count :
  ∃! (s : Finset ℕ), s.card = 4 ∧
  (∀ n : ℕ, n ∈ s ↔ (is_four_digit n ∧ divisible_by_all n)) :=
sorry

end four_digit_divisible_count_l447_44757


namespace min_RS_value_l447_44773

/-- Represents a rhombus ABCD with given diagonals -/
structure Rhombus where
  AC : ℝ
  BD : ℝ

/-- Represents a point M on side AB of the rhombus -/
structure PointM where
  BM : ℝ

/-- The minimum value of RS given the rhombus and point M -/
noncomputable def min_RS (r : Rhombus) (m : PointM) : ℝ :=
  Real.sqrt (8 * m.BM^2 - 40 * m.BM + 400)

/-- Theorem stating the minimum value of RS -/
theorem min_RS_value (r : Rhombus) : 
  r.AC = 24 → r.BD = 40 → ∃ (m : PointM), min_RS r m = 5 * Real.sqrt 14 := by
  sorry

#check min_RS_value

end min_RS_value_l447_44773


namespace square_perimeter_sum_l447_44732

theorem square_perimeter_sum (a b : ℝ) (h1 : a + b = 85) (h2 : a - b = 41) :
  4 * (Real.sqrt a.toNNReal + Real.sqrt b.toNNReal) = 4 * (Real.sqrt 63 + Real.sqrt 22) :=
by sorry

end square_perimeter_sum_l447_44732


namespace distance_to_focus_l447_44723

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus (x : ℝ) : 
  let P : ℝ × ℝ := (x, (1/4) * x^2)
  let parabola := {(x, y) : ℝ × ℝ | y = (1/4) * x^2}
  P ∈ parabola → P.2 = 4 → ∃ F : ℝ × ℝ, F.2 = 1/4 ∧ dist P F = 5 := by
  sorry

end distance_to_focus_l447_44723


namespace barbaras_to_mikes_age_ratio_l447_44739

/-- Given that Mike is currently 16 years old and Barbara will be 16 years old
    when Mike is 24 years old, prove that the ratio of Barbara's current age
    to Mike's current age is 1:2. -/
theorem barbaras_to_mikes_age_ratio :
  let mike_current_age : ℕ := 16
  let mike_future_age : ℕ := 24
  let barbara_future_age : ℕ := 16
  let age_difference : ℕ := mike_future_age - mike_current_age
  let barbara_current_age : ℕ := barbara_future_age - age_difference
  (barbara_current_age : ℚ) / mike_current_age = 1 / 2 := by
  sorry

end barbaras_to_mikes_age_ratio_l447_44739


namespace circle_tangents_theorem_no_single_common_tangent_l447_44762

/-- Represents the number of common tangents between two circles -/
inductive CommonTangents
  | zero
  | two
  | three
  | four

/-- Represents the configuration of two circles -/
structure CircleConfiguration where
  r1 : ℝ  -- radius of the first circle
  r2 : ℝ  -- radius of the second circle
  d : ℝ   -- distance between the centers of the circles

/-- Function to determine the number of common tangents based on circle configuration -/
def numberOfCommonTangents (config : CircleConfiguration) : CommonTangents :=
  sorry

/-- Theorem stating that two circles with radii 10 and 4 can have 0, 2, 3, or 4 common tangents -/
theorem circle_tangents_theorem :
  ∀ (d : ℝ),
  let config := CircleConfiguration.mk 10 4 d
  (numberOfCommonTangents config = CommonTangents.zero) ∨
  (numberOfCommonTangents config = CommonTangents.two) ∨
  (numberOfCommonTangents config = CommonTangents.three) ∨
  (numberOfCommonTangents config = CommonTangents.four) :=
by sorry

/-- Theorem stating that two circles with radii 10 and 4 cannot have exactly 1 common tangent -/
theorem no_single_common_tangent :
  ∀ (d : ℝ),
  let config := CircleConfiguration.mk 10 4 d
  numberOfCommonTangents config ≠ CommonTangents.zero :=
by sorry

end circle_tangents_theorem_no_single_common_tangent_l447_44762


namespace peggy_dolls_l447_44798

/-- The number of dolls Peggy has at the end -/
def final_dolls (initial : ℕ) (grandmother : ℕ) : ℕ :=
  initial + grandmother + grandmother / 2

/-- Theorem stating that Peggy ends up with 51 dolls -/
theorem peggy_dolls : final_dolls 6 30 = 51 := by
  sorry

end peggy_dolls_l447_44798


namespace pony_discount_rate_l447_44761

/-- Represents the discount rate for Fox jeans -/
def F : ℝ := sorry

/-- Represents the discount rate for Pony jeans -/
def P : ℝ := sorry

/-- Regular price of Fox jeans -/
def fox_price : ℝ := 15

/-- Regular price of Pony jeans -/
def pony_price : ℝ := 18

/-- Total savings from purchasing 3 pairs of Fox jeans and 2 pairs of Pony jeans -/
def total_savings : ℝ := 8.64

/-- The sum of discount rates for Fox and Pony jeans -/
def total_discount : ℝ := 22

theorem pony_discount_rate : 
  F + P = total_discount ∧ 
  3 * (fox_price * F / 100) + 2 * (pony_price * P / 100) = total_savings →
  P = 14 := by sorry

end pony_discount_rate_l447_44761


namespace ants_after_five_hours_l447_44785

/-- The number of ants in the jar after a given number of hours -/
def antsInJar (initialAnts : ℕ) (hours : ℕ) : ℕ :=
  initialAnts * (2 ^ hours)

/-- Theorem stating that 50 ants doubling every hour for 5 hours results in 1600 ants -/
theorem ants_after_five_hours :
  antsInJar 50 5 = 1600 := by
  sorry

end ants_after_five_hours_l447_44785


namespace unique_salaries_l447_44705

/-- Represents the weekly salaries of three employees -/
structure Salaries where
  n : ℝ  -- Salary of employee N
  m : ℝ  -- Salary of employee M
  p : ℝ  -- Salary of employee P

/-- Checks if the given salaries satisfy the problem conditions -/
def satisfiesConditions (s : Salaries) : Prop :=
  s.m = 1.2 * s.n ∧
  s.p = 1.5 * s.m ∧
  s.n + s.m + s.p = 1500

/-- Theorem stating that the given salaries are the unique solution -/
theorem unique_salaries : 
  ∃! s : Salaries, satisfiesConditions s ∧ 
    s.n = 375 ∧ s.m = 450 ∧ s.p = 675 := by
  sorry

end unique_salaries_l447_44705


namespace area_of_ω_l447_44754

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (7, 15)
def B : ℝ × ℝ := (15, 9)

-- State that A and B lie on ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- State that the tangent lines intersect at a point on the x-axis
axiom tangents_intersect_x_axis : ∃ x : ℝ, (x, 0) ∈ tangent_A ∩ tangent_B

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_ω : circle_area ω = 6525 * Real.pi / 244 := sorry

end area_of_ω_l447_44754


namespace max_d_value_l447_44797

/-- The sequence term for a given n -/
def a (n : ℕ) : ℕ := 100 + n^2

/-- The greatest common divisor of consecutive terms -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The theorem stating the maximum value of d_n -/
theorem max_d_value : ∃ (N : ℕ), ∀ (n : ℕ), n > 0 → d n ≤ 401 ∧ d N = 401 := by
  sorry

end max_d_value_l447_44797


namespace forty_percent_bought_something_l447_44733

/-- Given advertising costs, number of customers, item price, and profit,
    calculates the percentage of customers who made a purchase. -/
def percentage_of_customers_who_bought (advertising_cost : ℕ) (num_customers : ℕ) 
  (item_price : ℕ) (profit : ℕ) : ℚ :=
  (profit / item_price : ℚ) / num_customers * 100

/-- Theorem stating that under the given conditions, 
    40% of customers made a purchase. -/
theorem forty_percent_bought_something :
  percentage_of_customers_who_bought 1000 100 25 1000 = 40 := by
  sorry

#eval percentage_of_customers_who_bought 1000 100 25 1000

end forty_percent_bought_something_l447_44733


namespace gcd_of_three_numbers_l447_44770

theorem gcd_of_three_numbers : Nat.gcd 10711 (Nat.gcd 15809 28041) = 1 := by
  sorry

end gcd_of_three_numbers_l447_44770


namespace non_overlapping_area_l447_44753

/-- Rectangle ABCD with side lengths 4 and 6 -/
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)
  (h_AB : AB = 4)
  (h_BC : BC = 6)

/-- The fold that makes B and D coincide -/
structure Fold (rect : Rectangle) :=
  (E : ℝ × ℝ)  -- Point E on the crease
  (F : ℝ × ℝ)  -- Point F on the crease
  (h_coincide : E.1 + F.1 = rect.AB ∧ E.2 + F.2 = rect.BC)  -- B and D coincide after folding

/-- The theorem stating the area of the non-overlapping part -/
theorem non_overlapping_area (rect : Rectangle) (fold : Fold rect) :
  ∃ (area : ℝ), area = 20 / 3 ∧ area = 2 * (1 / 2 * rect.AB * (rect.BC - fold.E.2)) :=
sorry

end non_overlapping_area_l447_44753


namespace rocket_soaring_time_l447_44715

/-- Proves that the soaring time of a rocket is 12 seconds given specific conditions -/
theorem rocket_soaring_time :
  let soaring_speed : ℝ := 150
  let plummet_distance : ℝ := 600
  let plummet_time : ℝ := 3
  let average_speed : ℝ := 160
  let soaring_time : ℝ := 12

  (soaring_speed * soaring_time + plummet_distance) / (soaring_time + plummet_time) = average_speed :=
by
  sorry


end rocket_soaring_time_l447_44715


namespace min_value_of_f_l447_44745

def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem min_value_of_f (a : ℝ) : (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3) ↔ a = 1 ∨ a = 7 := by
  sorry

end min_value_of_f_l447_44745


namespace cube_root_equals_square_root_l447_44700

theorem cube_root_equals_square_root :
  ∀ x : ℝ, (x ^ (1/3) = x ^ (1/2)) → x = 0 :=
by
  sorry

end cube_root_equals_square_root_l447_44700


namespace angle_ABC_bisector_l447_44728

theorem angle_ABC_bisector (ABC : Real) : 
  (ABC / 2 = (180 - ABC) / 6) → ABC = 45 := by
  sorry

end angle_ABC_bisector_l447_44728


namespace probability_at_least_one_boy_and_girl_l447_44707

theorem probability_at_least_one_boy_and_girl (p : ℝ) : 
  p = 1/2 → (1 - 2 * p^4) = 7/8 := by sorry

end probability_at_least_one_boy_and_girl_l447_44707


namespace cantors_theorem_l447_44776

theorem cantors_theorem (X : Type u) : ¬∃(f : X → Set X), Function.Bijective f :=
  sorry

end cantors_theorem_l447_44776


namespace table_length_is_77_l447_44725

/-- The length of a rectangular table covered by overlapping paper sheets. -/
def table_length : ℕ :=
  let table_width : ℕ := 80
  let sheet_width : ℕ := 8
  let sheet_height : ℕ := 5
  let offset : ℕ := 1
  let sheets_needed : ℕ := table_width - sheet_width
  sheet_height + sheets_needed

theorem table_length_is_77 : table_length = 77 := by
  sorry

end table_length_is_77_l447_44725


namespace solve_equation_l447_44740

theorem solve_equation (x : ℝ) : 
  ((2*x + 8) + (7*x + 3) + (3*x + 9)) / 3 = 5*x^2 - 8*x + 2 → 
  x = (36 + Real.sqrt 2136) / 30 ∨ x = (36 - Real.sqrt 2136) / 30 :=
by sorry

end solve_equation_l447_44740


namespace difference_of_squares_factorization_l447_44755

theorem difference_of_squares_factorization (y : ℝ) : 
  100 - 16 * y^2 = 4 * (5 - 2*y) * (5 + 2*y) := by
  sorry

end difference_of_squares_factorization_l447_44755


namespace trig_identity_proof_l447_44751

theorem trig_identity_proof (α : ℝ) : 
  Real.cos (α - 35 * π / 180) * Real.cos (25 * π / 180 + α) + 
  Real.sin (α - 35 * π / 180) * Real.sin (25 * π / 180 + α) = 1 / 2 := by
sorry

end trig_identity_proof_l447_44751


namespace min_value_sum_of_roots_l447_44724

theorem min_value_sum_of_roots (x : ℝ) :
  ∃ (y : ℝ), ∀ (x : ℝ),
    Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 - x)^2) ≥ y ∧
    (∃ (z : ℝ), Real.sqrt (z^2 + (1 - z)^2) + Real.sqrt ((1 - z)^2 + (1 - z)^2) = y) :=
by
  -- Proof goes here
  sorry

end min_value_sum_of_roots_l447_44724


namespace equation_satisfied_l447_44702

theorem equation_satisfied (x y z : ℤ) : 
  x = z ∧ y = x + 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
sorry

end equation_satisfied_l447_44702


namespace sequence_ratio_l447_44727

def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  is_arithmetic_sequence 1 a₁ a₂ 9 →
  is_geometric_sequence 1 b₁ b₂ b₃ 9 →
  b₂ / (a₁ + a₂) = 3 / 10 :=
by sorry

end sequence_ratio_l447_44727


namespace arithmetic_operations_l447_44774

theorem arithmetic_operations :
  ((-12) + (-6) - (-28) = 10) ∧
  ((-8/5) * (15/4) / (-9) = 2/3) ∧
  ((-3/16 - 7/24 + 5/6) * (-48) = -17) ∧
  (-(3^2) + (7/8 - 1) * ((-2)^2) = -19/2) := by
  sorry

end arithmetic_operations_l447_44774


namespace apples_distribution_l447_44735

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 5

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 2 * benny_apples

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The number of apples each friend received -/
def apples_per_friend : ℕ := total_apples / num_friends

theorem apples_distribution :
  apples_per_friend = 5 :=
sorry

end apples_distribution_l447_44735


namespace geometric_sequence_property_l447_44736

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (3 * (a 3)^2 - 11 * (a 3) + 9 = 0) →
  (3 * (a 9)^2 - 11 * (a 9) + 9 = 0) →
  (a 6 = Real.sqrt 3 ∨ a 6 = -Real.sqrt 3) :=
by sorry

end geometric_sequence_property_l447_44736


namespace rational_sqrt_n_minus_3_over_n_plus_1_l447_44779

theorem rational_sqrt_n_minus_3_over_n_plus_1 
  (r q n : ℚ) 
  (h : 1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q)) :
  ∃ (a b : ℚ), b ≠ 0 ∧ (n - 3) / (n + 1) = (a / b) ^ 2 :=
sorry

end rational_sqrt_n_minus_3_over_n_plus_1_l447_44779


namespace function_property_l447_44701

theorem function_property (f : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ x, f (2 + x) = f (-x))
  (h2 : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f y < f x)
  (h3 : f (1 - m) < f m) :
  m > 1/2 := by sorry

end function_property_l447_44701


namespace cinema_meeting_day_l447_44781

theorem cinema_meeting_day : Nat.lcm (Nat.lcm 4 5) 6 = 60 := by
  sorry

end cinema_meeting_day_l447_44781


namespace fraction_zero_implies_x_negative_three_l447_44780

theorem fraction_zero_implies_x_negative_three (x : ℝ) : 
  (x + 3) / (x - 4) = 0 → x = -3 := by
  sorry

end fraction_zero_implies_x_negative_three_l447_44780


namespace total_decorations_count_l447_44767

/-- The number of decorations in each box -/
def decorations_per_box : ℕ := 4 + 1 + 5

/-- The number of families receiving a box -/
def number_of_families : ℕ := 11

/-- The number of boxes given to the community center -/
def community_center_boxes : ℕ := 1

/-- The total number of decorations handed out -/
def total_decorations : ℕ := decorations_per_box * (number_of_families + community_center_boxes)

theorem total_decorations_count : total_decorations = 120 := by
  sorry


end total_decorations_count_l447_44767


namespace sin_plus_cos_eq_neg_one_solution_set_l447_44790

theorem sin_plus_cos_eq_neg_one_solution_set :
  {x : ℝ | Real.sin x + Real.cos x = -1} =
  {x : ℝ | ∃ n : ℤ, x = (2*n - 1)*π ∨ x = 2*n*π - π/2} := by
  sorry

end sin_plus_cos_eq_neg_one_solution_set_l447_44790


namespace cube_volume_from_surface_area_l447_44737

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 150 → volume = 125 := by
  sorry

end cube_volume_from_surface_area_l447_44737


namespace stating_seedling_cost_equations_l447_44746

/-- Represents the cost of seedlings and their price difference -/
structure SeedlingCost where
  x : ℝ  -- Cost of one pine seedling in yuan
  y : ℝ  -- Cost of one tamarisk seedling in yuan
  total_cost : 4 * x + 3 * y = 180  -- Total cost equation
  price_difference : x - y = 10  -- Price difference equation

/-- 
Theorem stating that the given system of equations correctly represents 
the cost of pine and tamarisk seedlings under the given conditions
-/
theorem seedling_cost_equations (cost : SeedlingCost) : 
  (4 * cost.x + 3 * cost.y = 180) ∧ (cost.x - cost.y = 10) := by
  sorry

end stating_seedling_cost_equations_l447_44746


namespace set_of_positive_rationals_l447_44730

theorem set_of_positive_rationals (S : Set ℚ) :
  (∀ a b : ℚ, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a * b) ∈ S) →
  (∀ r : ℚ, (r ∈ S ∧ -r ∉ S ∧ r ≠ 0) ∨ (-r ∈ S ∧ r ∉ S ∧ r ≠ 0) ∨ (r = 0 ∧ r ∉ S ∧ -r ∉ S)) →
  S = {r : ℚ | r > 0} :=
by sorry

end set_of_positive_rationals_l447_44730


namespace coin_toss_problem_l447_44792

theorem coin_toss_problem (n : ℕ) 
  (total_outcomes : ℕ) 
  (equally_likely : total_outcomes = 8)
  (die_roll_prob : ℚ) 
  (die_roll_prob_value : die_roll_prob = 1/3) :
  (2^n = total_outcomes) → n = 3 := by
  sorry

end coin_toss_problem_l447_44792


namespace total_marbles_l447_44765

def marble_collection (jar1 jar2 jar3 : ℕ) : Prop :=
  (jar1 = 80) ∧
  (jar2 = 2 * jar1) ∧
  (jar3 = jar1 / 4) ∧
  (jar1 + jar2 + jar3 = 260)

theorem total_marbles :
  ∃ (jar1 jar2 jar3 : ℕ), marble_collection jar1 jar2 jar3 :=
sorry

end total_marbles_l447_44765


namespace segment_AE_length_l447_44704

-- Define the quadrilateral ABCD and point E
structure Quadrilateral :=
  (A B C D E : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let d_AB := Real.sqrt ((q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2)
  let d_CD := Real.sqrt ((q.D.1 - q.C.1)^2 + (q.D.2 - q.C.2)^2)
  let d_AC := Real.sqrt ((q.C.1 - q.A.1)^2 + (q.C.2 - q.A.2)^2)
  let d_AE := Real.sqrt ((q.E.1 - q.A.1)^2 + (q.E.2 - q.A.2)^2)
  let d_EC := Real.sqrt ((q.C.1 - q.E.1)^2 + (q.C.2 - q.E.2)^2)
  d_AB = 10 ∧ d_CD = 15 ∧ d_AC = 18 ∧
  (q.E.1 - q.A.1) * (q.C.1 - q.A.1) + (q.E.2 - q.A.2) * (q.C.2 - q.A.2) = d_AE * d_AC ∧
  (q.E.1 - q.B.1) * (q.D.1 - q.B.1) + (q.E.2 - q.B.2) * (q.D.2 - q.B.2) = 
    Real.sqrt ((q.E.1 - q.B.1)^2 + (q.E.2 - q.B.2)^2) * Real.sqrt ((q.D.1 - q.B.1)^2 + (q.D.2 - q.B.2)^2) ∧
  d_AE / d_EC = 10 / 15

theorem segment_AE_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  Real.sqrt ((q.E.1 - q.A.1)^2 + (q.E.2 - q.A.2)^2) = 36 / 5 := by
  sorry

end segment_AE_length_l447_44704


namespace correct_distribution_probability_l447_44706

/-- The number of rolls of each type -/
def rolls_per_type : ℕ := 3

/-- The number of types of rolls -/
def num_types : ℕ := 4

/-- The total number of rolls -/
def total_rolls : ℕ := rolls_per_type * num_types

/-- The number of guests -/
def num_guests : ℕ := 3

/-- The number of rolls each guest receives -/
def rolls_per_guest : ℕ := num_types

/-- The probability of each guest getting one roll of each type -/
def probability_correct_distribution : ℚ := 27 / 1925

theorem correct_distribution_probability :
  (rolls_per_type : ℚ) * (rolls_per_type - 1) * (rolls_per_type - 2) /
  (total_rolls * (total_rolls - 1) * (total_rolls - 2) * (total_rolls - 3)) *
  ((rolls_per_type - 1) * (rolls_per_type - 1) * (rolls_per_type - 1) /
  ((total_rolls - 4) * (total_rolls - 5) * (total_rolls - 6) * (total_rolls - 7))) *
  1 = probability_correct_distribution := by
  sorry

end correct_distribution_probability_l447_44706


namespace mall_parking_lot_cars_l447_44752

/-- The number of cars parked in a mall's parking lot -/
def number_of_cars : ℕ := 10

/-- The number of customers in each car -/
def customers_per_car : ℕ := 5

/-- The number of sales made by the sports store -/
def sports_store_sales : ℕ := 20

/-- The number of sales made by the music store -/
def music_store_sales : ℕ := 30

/-- Theorem stating that the number of cars is correct given the conditions -/
theorem mall_parking_lot_cars :
  number_of_cars * customers_per_car = sports_store_sales + music_store_sales :=
by sorry

end mall_parking_lot_cars_l447_44752


namespace cobbler_friday_hours_l447_44718

/-- Represents the cobbler's work week -/
structure CobblerWeek where
  shoes_per_hour : ℕ
  hours_per_day : ℕ
  days_before_friday : ℕ
  total_shoes_per_week : ℕ

/-- Calculates the number of hours worked on Friday -/
def friday_hours (week : CobblerWeek) : ℕ :=
  (week.total_shoes_per_week - week.shoes_per_hour * week.hours_per_day * week.days_before_friday) / week.shoes_per_hour

/-- Theorem stating that the cobbler works 3 hours on Friday -/
theorem cobbler_friday_hours :
  let week : CobblerWeek := {
    shoes_per_hour := 3,
    hours_per_day := 8,
    days_before_friday := 4,
    total_shoes_per_week := 105
  }
  friday_hours week = 3 := by
  sorry

end cobbler_friday_hours_l447_44718


namespace real_part_of_i_squared_times_one_plus_i_l447_44722

theorem real_part_of_i_squared_times_one_plus_i : 
  Complex.re (Complex.I^2 * (1 + Complex.I)) = -1 := by sorry

end real_part_of_i_squared_times_one_plus_i_l447_44722


namespace highest_power_of_three_in_M_l447_44731

def M : ℕ := sorry

def is_highest_power_of_three (n : ℕ) (j : ℕ) : Prop :=
  3^j ∣ n ∧ ∀ k > j, ¬(3^k ∣ n)

theorem highest_power_of_three_in_M :
  is_highest_power_of_three M 0 := by sorry

end highest_power_of_three_in_M_l447_44731


namespace geometric_sequence_second_term_l447_44726

/-- A geometric sequence with first term 5 and third term 20 has second term 10 -/
theorem geometric_sequence_second_term :
  ∀ (a : ℝ) (r : ℝ),
    a = 5 →
    a * r^2 = 20 →
    a * r = 10 :=
by
  sorry

end geometric_sequence_second_term_l447_44726


namespace not_right_triangle_when_A_eq_B_eq_3C_l447_44749

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Theorem statement
theorem not_right_triangle_when_A_eq_B_eq_3C (t : Triangle) 
  (h : t.A = t.B ∧ t.A = 3 * t.C) : 
  ¬ is_right_triangle t := by
  sorry

end not_right_triangle_when_A_eq_B_eq_3C_l447_44749


namespace problem_1_l447_44717

theorem problem_1 : (1 : ℝ) * (1 + Real.rpow 8 (1/3 : ℝ))^0 + abs (-2) - Real.sqrt 9 = 0 := by
  sorry

end problem_1_l447_44717


namespace largest_square_four_digits_base7_l447_44719

/-- Converts a decimal number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ := sorry

/-- Checks if a number has exactly 4 digits when written in base 7 -/
def hasFourDigitsBase7 (n : ℕ) : Prop :=
  (toBase7 n).length = 4

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := sorry

theorem largest_square_four_digits_base7 :
  M = (toBase7 66).foldl (fun acc d => acc * 7 + d) 0 ∧
  hasFourDigitsBase7 (M ^ 2) ∧
  ∀ n : ℕ, n > M → ¬hasFourDigitsBase7 (n ^ 2) :=
sorry

end largest_square_four_digits_base7_l447_44719


namespace cube_coloring_probability_l447_44758

/-- The probability of a single color being chosen for a face -/
def color_probability : ℚ := 1/3

/-- The number of pairs of opposite faces in a cube -/
def opposite_face_pairs : ℕ := 3

/-- The probability that a pair of opposite faces has different colors -/
def diff_color_prob : ℚ := 2/3

/-- The probability that all pairs of opposite faces have different colors -/
def all_diff_prob : ℚ := diff_color_prob ^ opposite_face_pairs

theorem cube_coloring_probability :
  1 - all_diff_prob = 19/27 := by sorry

end cube_coloring_probability_l447_44758
