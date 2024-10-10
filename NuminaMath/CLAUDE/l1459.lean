import Mathlib

namespace max_number_in_sample_l1459_145974

/-- Represents a systematic sample from a range of products -/
structure SystematicSample where
  total_products : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Creates a systematic sample given total products and sample size -/
def create_systematic_sample (total_products sample_size : ℕ) : SystematicSample :=
  { total_products := total_products
  , sample_size := sample_size
  , start := 0  -- Assuming start is 0 for simplicity
  , interval := total_products / sample_size
  }

/-- Checks if a number is in the systematic sample -/
def is_in_sample (sample : SystematicSample) (n : ℕ) : Prop :=
  ∃ k, 0 ≤ k ∧ k < sample.sample_size ∧ n = sample.start + k * sample.interval

/-- Gets the maximum number in the systematic sample -/
def max_in_sample (sample : SystematicSample) : ℕ :=
  sample.start + (sample.sample_size - 1) * sample.interval

/-- Theorem: If 58 is in a systematic sample of size 10 from 80 products, 
    then the maximum number in the sample is 74 -/
theorem max_number_in_sample :
  let sample := create_systematic_sample 80 10
  is_in_sample sample 58 → max_in_sample sample = 74 := by
  sorry


end max_number_in_sample_l1459_145974


namespace park_outer_diameter_l1459_145965

/-- Given a park layout with a central statue, lawn, and jogging path, 
    this theorem proves the diameter of the outer boundary. -/
theorem park_outer_diameter 
  (statue_diameter : ℝ) 
  (lawn_width : ℝ) 
  (jogging_path_width : ℝ) 
  (h1 : statue_diameter = 8) 
  (h2 : lawn_width = 10) 
  (h3 : jogging_path_width = 5) : 
  statue_diameter / 2 + lawn_width + jogging_path_width = 19 ∧ 
  2 * (statue_diameter / 2 + lawn_width + jogging_path_width) = 38 :=
sorry

end park_outer_diameter_l1459_145965


namespace thursday_dogs_l1459_145946

/-- The number of dogs Harry walks on Monday, Wednesday, and Friday -/
def dogs_mon_wed_fri : ℕ := 7

/-- The number of dogs Harry walks on Tuesday -/
def dogs_tuesday : ℕ := 12

/-- The amount Harry is paid per dog -/
def pay_per_dog : ℕ := 5

/-- Harry's total earnings for the week -/
def total_earnings : ℕ := 210

/-- The number of days Harry walks 7 dogs -/
def days_with_seven_dogs : ℕ := 3

theorem thursday_dogs :
  ∃ (dogs_thursday : ℕ),
    dogs_thursday * pay_per_dog =
      total_earnings -
      (days_with_seven_dogs * dogs_mon_wed_fri + dogs_tuesday) * pay_per_dog ∧
    dogs_thursday = 9 :=
sorry

end thursday_dogs_l1459_145946


namespace miriam_flowers_per_day_l1459_145950

/-- The number of flowers Miriam can take care of in 6 days -/
def total_flowers : ℕ := 360

/-- The number of days Miriam works -/
def work_days : ℕ := 6

/-- The number of flowers Miriam can take care of in one day -/
def flowers_per_day : ℕ := total_flowers / work_days

theorem miriam_flowers_per_day : flowers_per_day = 60 := by
  sorry

end miriam_flowers_per_day_l1459_145950


namespace carla_earnings_l1459_145949

/-- Carla's earnings over two weeks in June --/
theorem carla_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 18 →
  hours_week2 = 28 →
  extra_earnings = 63 →
  ∃ (hourly_wage : ℚ),
    hourly_wage * (hours_week2 - hours_week1 : ℚ) = extra_earnings ∧
    hourly_wage * (hours_week1 + hours_week2 : ℚ) = 289.80 := by
  sorry

#check carla_earnings

end carla_earnings_l1459_145949


namespace f_form_l1459_145943

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_continuous : Continuous f
axiom f_functional_equation : ∀ x y : ℝ, f (Real.sqrt (x^2 + y^2)) = f x * f y

-- State the theorem to be proved
theorem f_form : ∀ x : ℝ, f x = (f 1) ^ (x^2) := by sorry

end f_form_l1459_145943


namespace arithmetic_sequence_problem_l1459_145971

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₄ = 5 and a₉ = 17, then a₁₄ = 29. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arithmetic : IsArithmeticSequence a)
    (h_a4 : a 4 = 5)
    (h_a9 : a 9 = 17) : 
  a 14 = 29 := by
  sorry


end arithmetic_sequence_problem_l1459_145971


namespace molecular_weight_NH4_correct_l1459_145991

/-- The molecular weight of NH4 in grams per mole -/
def molecular_weight_NH4 : ℝ := 18

/-- The number of moles in the given sample -/
def sample_moles : ℝ := 7

/-- The total weight of the sample in grams -/
def sample_weight : ℝ := 126

/-- Theorem stating that the molecular weight of NH4 is correct given the sample information -/
theorem molecular_weight_NH4_correct :
  molecular_weight_NH4 * sample_moles = sample_weight :=
sorry

end molecular_weight_NH4_correct_l1459_145991


namespace total_cost_for_cakes_l1459_145954

/-- The number of cakes Claire wants to make -/
def num_cakes : ℕ := 2

/-- The number of packages of flour required for one cake -/
def packages_per_cake : ℕ := 2

/-- The cost of one package of flour in dollars -/
def cost_per_package : ℕ := 3

/-- Theorem: The total cost of flour for making 2 cakes is $12 -/
theorem total_cost_for_cakes : num_cakes * packages_per_cake * cost_per_package = 12 := by
  sorry

end total_cost_for_cakes_l1459_145954


namespace hyperbola_foci_intersection_l1459_145915

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the circle with diameter equal to the distance between its foci
    intersects one of its asymptotes at the point (3,4),
    then a = 3 and b = 4. -/
theorem hyperbola_foci_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2) →
  (∃ (x y : ℝ), x^2 + y^2 = c^2 ∧ y/x = b/a ∧ x = 3 ∧ y = 4) →
  a = 3 ∧ b = 4 := by
sorry

end hyperbola_foci_intersection_l1459_145915


namespace integers_between_neg_sqrt2_and_sqrt2_l1459_145914

theorem integers_between_neg_sqrt2_and_sqrt2 :
  {x : ℤ | -Real.sqrt 2 < x ∧ x < Real.sqrt 2} = {-1, 0, 1} := by sorry

end integers_between_neg_sqrt2_and_sqrt2_l1459_145914


namespace number_of_siblings_l1459_145911

def total_spent : ℕ := 150
def cost_per_sibling : ℕ := 30
def cost_per_parent : ℕ := 30
def num_parents : ℕ := 2

theorem number_of_siblings :
  (total_spent - num_parents * cost_per_parent) / cost_per_sibling = 3 := by
  sorry

end number_of_siblings_l1459_145911


namespace line_equation_proof_l1459_145956

/-- Given two points (1, 0.5) and (1.5, 2), and the fact that the line passing through these points
    splits 8 circles such that the total circle area to the left of the line is 4π,
    prove that the equation of this line is 6x - 2y = 5. -/
theorem line_equation_proof (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) (num_circles : ℕ) (left_area : ℝ) :
  p1 = (1, 0.5) →
  p2 = (1.5, 2) →
  num_circles = 8 →
  left_area = 4 * Real.pi →
  ∃ (f : ℝ → ℝ), (∀ x y, f x = y ↔ 6 * x - 2 * y = 5) ∧
                 (∀ x, f x = (x - p1.1) * ((p2.2 - p1.2) / (p2.1 - p1.1)) + p1.2) :=
by sorry

end line_equation_proof_l1459_145956


namespace sams_calculation_l1459_145940

theorem sams_calculation (x y : ℝ) : 
  x + 2 * 2 + y = x * 2 + 2 + y → x + y = 4 := by
  sorry

end sams_calculation_l1459_145940


namespace f_symmetry_l1459_145924

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the theorem
theorem f_symmetry (a b : ℝ) :
  f a b 2017 = 7 → f a b (-2017) = -11 := by
  sorry

end f_symmetry_l1459_145924


namespace work_completion_days_l1459_145938

/-- The number of days required for the second group to complete the work -/
def days_for_second_group : ℕ := 4

/-- The daily work output of a boy -/
def boy_work : ℝ := 1

/-- The daily work output of a man -/
def man_work : ℝ := 2 * boy_work

/-- The total amount of work to be done -/
def total_work : ℝ := (12 * man_work + 16 * boy_work) * 5

theorem work_completion_days :
  (13 * man_work + 24 * boy_work) * days_for_second_group = total_work := by sorry

end work_completion_days_l1459_145938


namespace function_minimum_value_l1459_145988

/-- The function f(x) = x + a / (x - 2) where x > 2 and f(3) = 7 has a minimum value of 6 -/
theorem function_minimum_value (a : ℝ) : 
  (∀ x > 2, ∃ y, y = x + a / (x - 2)) → 
  (3 + a / (3 - 2) = 7) → 
  (∃ m : ℝ, ∀ x > 2, x + a / (x - 2) ≥ m ∧ ∃ x₀ > 2, x₀ + a / (x₀ - 2) = m) →
  (∀ x > 2, x + a / (x - 2) ≥ 6) ∧ ∃ x₀ > 2, x₀ + a / (x₀ - 2) = 6 :=
by sorry

end function_minimum_value_l1459_145988


namespace smallest_b_value_l1459_145917

theorem smallest_b_value (k a b : ℝ) (h1 : k > 1) (h2 : k < a) (h3 : a < b)
  (h4 : k + a ≤ b) (h5 : 1/a + 1/b ≤ 1/k) : b ≥ 2*k := by
  sorry

end smallest_b_value_l1459_145917


namespace club_count_l1459_145995

theorem club_count (total : ℕ) (black : ℕ) (red : ℕ) (spades : ℕ) (diamonds : ℕ) (hearts : ℕ) (clubs : ℕ) :
  total = 13 →
  black = 7 →
  red = 6 →
  diamonds = 2 * spades →
  hearts = 2 * diamonds →
  total = spades + diamonds + hearts + clubs →
  black = spades + clubs →
  red = diamonds + hearts →
  clubs = 6 := by
sorry

end club_count_l1459_145995


namespace johns_final_elevation_l1459_145919

/-- Calculates the final elevation after descending for a given time. -/
def finalElevation (startElevation : ℝ) (descentRate : ℝ) (time : ℝ) : ℝ :=
  startElevation - descentRate * time

/-- Proves that John's final elevation is 350 feet. -/
theorem johns_final_elevation :
  let startElevation : ℝ := 400
  let descentRate : ℝ := 10
  let time : ℝ := 5
  finalElevation startElevation descentRate time = 350 := by
  sorry

end johns_final_elevation_l1459_145919


namespace rectangle_dimensions_l1459_145925

theorem rectangle_dimensions : ∃ (a b : ℝ), 
  b = a + 3 ∧ 
  2*a + 2*b + a = a*b ∧ 
  a = 3 ∧ 
  b = 6 := by
sorry

end rectangle_dimensions_l1459_145925


namespace valid_assignment_example_l1459_145981

def is_variable (s : String) : Prop := s.length > 0 ∧ s.all Char.isAlpha

def is_expression (s : String) : Prop := s.length > 0

def is_valid_assignment (s : String) : Prop :=
  ∃ (lhs rhs : String),
    s = lhs ++ " = " ++ rhs ∧
    is_variable lhs ∧
    is_expression rhs

theorem valid_assignment_example :
  is_valid_assignment "A = A*A + A - 2" := by sorry

end valid_assignment_example_l1459_145981


namespace nicky_profit_l1459_145906

def card_value_traded : ℕ := 8
def num_cards_traded : ℕ := 2
def card_value_received : ℕ := 21

def profit : ℕ := card_value_received - (card_value_traded * num_cards_traded)

theorem nicky_profit :
  profit = 5 := by sorry

end nicky_profit_l1459_145906


namespace crazy_silly_school_movies_l1459_145997

/-- The number of remaining movies to watch in the 'crazy silly school' series -/
def remaining_movies (total : ℕ) (watched : ℕ) : ℕ :=
  total - watched

theorem crazy_silly_school_movies : 
  remaining_movies 17 7 = 10 := by
  sorry

end crazy_silly_school_movies_l1459_145997


namespace rectangle_longer_side_length_l1459_145923

/-- Given a circle of radius 6 cm tangent to three sides of a rectangle,
    if the rectangle's area is three times the circle's area,
    then the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side_length (circle_radius : ℝ) (rectangle_shorter_side rectangle_longer_side : ℝ) :
  circle_radius = 6 →
  rectangle_shorter_side = 2 * circle_radius →
  rectangle_shorter_side * rectangle_longer_side = 3 * Real.pi * circle_radius^2 →
  rectangle_longer_side = 9 * Real.pi := by
  sorry

end rectangle_longer_side_length_l1459_145923


namespace hyperbola_asymptotes_l1459_145998

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end hyperbola_asymptotes_l1459_145998


namespace constant_chord_length_l1459_145990

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the satellite circle -/
def satellite_circle (x y a b : ℝ) : Prop := x^2 + y^2 = a^2 + b^2

/-- Theorem statement -/
theorem constant_chord_length (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_ecc : (a^2 - b^2) / a^2 = 1/2)
  (h_point : ellipse 2 (Real.sqrt 2) a b)
  (h_sat : satellite_circle 2 (Real.sqrt 2) a b) :
  ∃ (M N : ℝ × ℝ),
    ∀ (P : ℝ × ℝ), satellite_circle P.1 P.2 a b →
      ∃ (l₁ l₂ : ℝ → ℝ),
        (∀ x, (l₁ x - P.2) * (l₂ x - P.2) = -(x - P.1)^2) ∧
        (∃! x₁, ellipse x₁ (l₁ x₁) a b) ∧
        (∃! x₂, ellipse x₂ (l₂ x₂) a b) ∧
        satellite_circle M.1 M.2 a b ∧
        satellite_circle N.1 N.2 a b ∧
        (M.1 - N.1)^2 + (M.2 - N.2)^2 = 48 :=
by
  sorry

end constant_chord_length_l1459_145990


namespace speed_gain_per_week_l1459_145935

def initial_speed : ℝ := 80
def training_weeks : ℕ := 16
def speed_increase_percentage : ℝ := 0.20

theorem speed_gain_per_week :
  let final_speed := initial_speed * (1 + speed_increase_percentage)
  let total_speed_gain := final_speed - initial_speed
  let speed_gain_per_week := total_speed_gain / training_weeks
  speed_gain_per_week = 1 := by
  sorry

end speed_gain_per_week_l1459_145935


namespace sequence_difference_theorem_l1459_145932

theorem sequence_difference_theorem (a : Fin 29 → ℤ) 
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_bound : ∀ k, k ≤ 22 → a (k + 7) - a k ≤ 13) :
  ∃ i j, a i - a j = 4 := by
sorry

end sequence_difference_theorem_l1459_145932


namespace probability_three_fives_out_of_five_dice_probability_exactly_three_fives_l1459_145922

/-- The probability of exactly 3 out of 5 fair 10-sided dice showing the number 5 -/
theorem probability_three_fives_out_of_five_dice : ℚ :=
  81 / 10000

/-- A fair 10-sided die -/
def fair_10_sided_die : Finset ℕ := Finset.range 10

/-- The probability of rolling a 5 on a fair 10-sided die -/
def prob_roll_5 : ℚ := 1 / 10

/-- The probability of not rolling a 5 on a fair 10-sided die -/
def prob_not_roll_5 : ℚ := 9 / 10

/-- The number of ways to choose 3 dice out of 5 -/
def ways_to_choose_3_out_of_5 : ℕ := 10

theorem probability_exactly_three_fives (n : ℕ) (k : ℕ) 
  (h1 : n = 5) (h2 : k = 3) : 
  probability_three_fives_out_of_five_dice = 
    (ways_to_choose_3_out_of_5 : ℚ) * (prob_roll_5 ^ k) * (prob_not_roll_5 ^ (n - k)) :=
sorry

end probability_three_fives_out_of_five_dice_probability_exactly_three_fives_l1459_145922


namespace exists_quadrilateral_equal_angle_tangents_l1459_145934

/-- A planar quadrilateral is represented by its four interior angles -/
structure PlanarQuadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 360

/-- Theorem: There exists a planar quadrilateral where the tangents of all its interior angles are equal -/
theorem exists_quadrilateral_equal_angle_tangents : 
  ∃ q : PlanarQuadrilateral, Real.tan q.α = Real.tan q.β ∧ Real.tan q.β = Real.tan q.γ ∧ Real.tan q.γ = Real.tan q.δ :=
sorry

end exists_quadrilateral_equal_angle_tangents_l1459_145934


namespace overlap_ratio_l1459_145929

theorem overlap_ratio (circle_area square_area overlap_area : ℝ) 
  (h1 : overlap_area = 0.5 * circle_area)
  (h2 : overlap_area = 0.25 * square_area) :
  (square_area - overlap_area) / (circle_area + square_area - overlap_area) = 3/5 := by
sorry

end overlap_ratio_l1459_145929


namespace complex_solution_l1459_145993

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- State the theorem
theorem complex_solution :
  ∃ z : ℂ, det 2 (-1) z (z * Complex.I) = 1 + Complex.I ∧ z = 3/5 - 1/5 * Complex.I :=
sorry

end complex_solution_l1459_145993


namespace inequality_proof_l1459_145960

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  a - d > b - c := by
  sorry

end inequality_proof_l1459_145960


namespace sam_speed_calculation_l1459_145966

def alex_speed : ℚ := 6
def jamie_relative_speed : ℚ := 4/5
def sam_relative_speed : ℚ := 3/4

theorem sam_speed_calculation :
  alex_speed * jamie_relative_speed * sam_relative_speed = 18/5 := by
  sorry

end sam_speed_calculation_l1459_145966


namespace x_in_terms_of_y_l1459_145941

theorem x_in_terms_of_y (x y : ℝ) (h : x / (x - 3) = (y^2 + 3*y + 1) / (y^2 + 3*y - 4)) :
  x = (3*y^2 + 9*y + 3) / 5 := by
sorry

end x_in_terms_of_y_l1459_145941


namespace baseball_cards_equality_l1459_145904

theorem baseball_cards_equality (J M C : ℕ) : 
  C = 20 → 
  M = C - 6 → 
  J + M + C = 48 → 
  J = M := by sorry

end baseball_cards_equality_l1459_145904


namespace trigonometric_identity_l1459_145948

theorem trigonometric_identity (α : Real) 
  (h : (1 + Real.sin α) * (1 - Real.cos α) = 1) : 
  (1 - Real.sin α) * (1 + Real.cos α) = 1 - Real.sin (2 * α) := by
  sorry

end trigonometric_identity_l1459_145948


namespace range_of_x_when_m_is_one_range_of_m_for_sufficient_condition_l1459_145967

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 10*x + 16 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 ≤ 0

-- Theorem for part (1)
theorem range_of_x_when_m_is_one (x : ℝ) :
  (∃ m : ℝ, m = 1 ∧ m > 0 ∧ (p x ∨ q x m)) → x ∈ Set.Icc 1 8 :=
sorry

-- Theorem for part (2)
theorem range_of_m_for_sufficient_condition (m : ℝ) :
  (m > 0 ∧ (∀ x : ℝ, q x m → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x m)) →
  m ∈ Set.Icc 2 (8/3) :=
sorry

end range_of_x_when_m_is_one_range_of_m_for_sufficient_condition_l1459_145967


namespace eighth_power_fraction_l1459_145987

theorem eighth_power_fraction (x : ℝ) (h : x > 0) :
  (x^(1/2)) / (x^(1/4)) = x^(1/4) :=
by sorry

end eighth_power_fraction_l1459_145987


namespace kylie_made_five_bracelets_l1459_145977

/-- The number of beaded bracelets Kylie made on Wednesday -/
def bracelets_made_wednesday (
  monday_necklaces : ℕ)
  (tuesday_necklaces : ℕ)
  (wednesday_earrings : ℕ)
  (beads_per_necklace : ℕ)
  (beads_per_bracelet : ℕ)
  (beads_per_earring : ℕ)
  (total_beads_used : ℕ) : ℕ :=
  (total_beads_used - 
   (monday_necklaces + tuesday_necklaces) * beads_per_necklace - 
   wednesday_earrings * beads_per_earring) / 
  beads_per_bracelet

/-- Theorem stating that Kylie made 5 beaded bracelets on Wednesday -/
theorem kylie_made_five_bracelets : 
  bracelets_made_wednesday 10 2 7 20 10 5 325 = 5 := by
  sorry

end kylie_made_five_bracelets_l1459_145977


namespace supermarket_profit_analysis_l1459_145969

/-- Represents the daily sales volume as a function of selling price -/
def sales_volume (x : ℤ) : ℝ := -5 * x + 150

/-- Represents the daily profit as a function of selling price -/
def profit (x : ℤ) : ℝ := (sales_volume x) * (x - 10)

theorem supermarket_profit_analysis 
  (x : ℤ) 
  (h_range : 10 ≤ x ∧ x ≤ 15) 
  (h_sales_12 : sales_volume 12 = 90) 
  (h_sales_14 : sales_volume 14 = 80) :
  (∃ (k b : ℝ), ∀ (x : ℤ), sales_volume x = k * x + b) ∧ 
  (profit 14 = 320) ∧
  (∀ (y : ℤ), 10 ≤ y ∧ y ≤ 15 → profit y ≤ profit 15) ∧
  (profit 15 = 375) :=
sorry

end supermarket_profit_analysis_l1459_145969


namespace quadratic_roots_sum_product_l1459_145994

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 14) → 
  p + q = 69 := by
  sorry

end quadratic_roots_sum_product_l1459_145994


namespace train_bridge_crossing_time_l1459_145951

/-- Proves the time taken for a train to cross a bridge given its length, speed, and time to pass a fixed point on the bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (signal_post_time : ℝ) 
  (bridge_fixed_point_time : ℝ) 
  (h1 : train_length = 600) 
  (h2 : signal_post_time = 40) 
  (h3 : bridge_fixed_point_time = 1200) :
  let train_speed := train_length / signal_post_time
  let bridge_length := train_speed * bridge_fixed_point_time - train_length
  let total_distance := bridge_length + train_length
  total_distance / train_speed = 1240 := by
  sorry

end train_bridge_crossing_time_l1459_145951


namespace ratio_fifth_to_first_l1459_145936

/-- An arithmetic sequence with a non-zero common difference where a₁, a₂, and a₅ form a geometric sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : (a 2) ^ 2 = a 1 * a 5

/-- The ratio of the fifth term to the first term in the special arithmetic sequence is 9. -/
theorem ratio_fifth_to_first (seq : ArithmeticSequence) : seq.a 5 / seq.a 1 = 9 := by
  sorry

end ratio_fifth_to_first_l1459_145936


namespace yellow_balls_count_l1459_145927

theorem yellow_balls_count (red_balls : ℕ) (probability_red : ℚ) (yellow_balls : ℕ) : 
  red_balls = 10 →
  probability_red = 2/5 →
  (red_balls : ℚ) / ((red_balls : ℚ) + (yellow_balls : ℚ)) = probability_red →
  yellow_balls = 15 := by
sorry

end yellow_balls_count_l1459_145927


namespace construct_octagon_from_square_l1459_145901

/-- A square sheet of paper --/
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

/-- A regular octagon --/
structure RegularOctagon :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Represents the ability to fold paper --/
def can_fold : Prop := True

/-- Represents the ability to cut along creases --/
def can_cut_along_creases : Prop := True

/-- Represents the prohibition of using a compass --/
def no_compass : Prop := True

/-- Represents the prohibition of using a ruler --/
def no_ruler : Prop := True

/-- Theorem stating that a regular octagon can be constructed from a square sheet of paper --/
theorem construct_octagon_from_square 
  (s : Square) 
  (fold : can_fold) 
  (cut : can_cut_along_creases) 
  (no_compass : no_compass) 
  (no_ruler : no_ruler) : 
  ∃ (o : RegularOctagon), True :=
sorry

end construct_octagon_from_square_l1459_145901


namespace fraction_simplification_l1459_145955

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by
  sorry

end fraction_simplification_l1459_145955


namespace ellipse_m_value_l1459_145920

/-- An ellipse with equation x²/(10-m) + y²/(m-2) = 1, major axis along y-axis, and focal length 4 -/
structure Ellipse (m : ℝ) :=
  (eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1)
  (major_axis : m - 2 > 10 - m)
  (focal_length : ℝ)
  (focal_length_eq : focal_length = 4)

/-- The value of m for the given ellipse is 8 -/
theorem ellipse_m_value (e : Ellipse m) : m = 8 := by
  sorry

end ellipse_m_value_l1459_145920


namespace jacket_selling_price_l1459_145926

/-- Calculates the total selling price of a jacket given the original price,
    discount rate, tax rate, and processing fee. -/
def total_selling_price (original_price discount_rate tax_rate processing_fee : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_tax := discounted_price * (1 + tax_rate)
  price_with_tax + processing_fee

/-- Theorem stating that the total selling price of the jacket is $95.72 -/
theorem jacket_selling_price :
  total_selling_price 120 0.30 0.08 5 = 95.72 := by
  sorry

#eval total_selling_price 120 0.30 0.08 5

end jacket_selling_price_l1459_145926


namespace complex_sum_magnitude_l1459_145903

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 0) :
  Complex.abs (a + b + c) = 3 :=
by sorry

end complex_sum_magnitude_l1459_145903


namespace specific_box_volume_l1459_145900

/-- The volume of an open box constructed from a rectangular sheet --/
def box_volume (sheet_length sheet_width y : ℝ) : ℝ :=
  (sheet_length - 2 * y) * (sheet_width - 2 * y) * y

/-- Theorem stating the volume of the specific box described in the problem --/
theorem specific_box_volume (y : ℝ) :
  box_volume 15 12 y = 180 * y - 54 * y^2 + 4 * y^3 :=
by sorry

end specific_box_volume_l1459_145900


namespace elongation_rate_improved_l1459_145931

def elongation_rate_comparison (x y : Fin 10 → ℝ) : Prop :=
  let z : Fin 10 → ℝ := fun i => x i - y i
  let z_mean : ℝ := (Finset.sum Finset.univ (fun i => z i)) / 10
  let z_variance : ℝ := (Finset.sum Finset.univ (fun i => (z i - z_mean)^2)) / 10
  z_mean = 11 ∧ 
  z_variance = 61 ∧ 
  z_mean ≥ 2 * Real.sqrt (z_variance / 10)

theorem elongation_rate_improved (x y : Fin 10 → ℝ) 
  (h : elongation_rate_comparison x y) : 
  ∃ (z_mean z_variance : ℝ), 
    z_mean = 11 ∧ 
    z_variance = 61 ∧ 
    z_mean ≥ 2 * Real.sqrt (z_variance / 10) :=
by
  sorry

end elongation_rate_improved_l1459_145931


namespace mr_mcpherson_contribution_l1459_145928

/-- Calculates the amount Mr. McPherson needs to raise for rent -/
theorem mr_mcpherson_contribution (total_rent : ℝ) (mrs_mcpherson_percentage : ℝ) :
  total_rent = 1200 →
  mrs_mcpherson_percentage = 30 →
  total_rent - (mrs_mcpherson_percentage / 100 * total_rent) = 840 := by
sorry

end mr_mcpherson_contribution_l1459_145928


namespace det_A_eq_48_l1459_145992

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, -2; 8, 5, -4; 3, 3, 6]

theorem det_A_eq_48 : Matrix.det A = 48 := by sorry

end det_A_eq_48_l1459_145992


namespace dave_deleted_eleven_apps_l1459_145980

/-- The number of apps Dave initially had on his phone -/
def initial_apps : ℕ := 16

/-- The number of apps Dave had left after deletion -/
def remaining_apps : ℕ := 5

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := initial_apps - remaining_apps

theorem dave_deleted_eleven_apps : deleted_apps = 11 := by
  sorry

end dave_deleted_eleven_apps_l1459_145980


namespace intersection_point_y_coordinate_l1459_145921

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the slope of the tangent at a point
def tangent_slope (x : ℝ) : ℝ := 4 * x

-- Define the condition for perpendicular tangents
def perpendicular_tangents (a b : ℝ) : Prop :=
  tangent_slope a * tangent_slope b = -1

-- Define the y-coordinate of the intersection point
def intersection_y (a b : ℝ) : ℝ := 2 * a * b

-- Theorem statement
theorem intersection_point_y_coordinate 
  (a b : ℝ) 
  (ha : parabola a = 2 * a^2) 
  (hb : parabola b = 2 * b^2) 
  (hperp : perpendicular_tangents a b) :
  intersection_y a b = -1/2 := by
  sorry

end intersection_point_y_coordinate_l1459_145921


namespace electronics_store_profit_l1459_145944

theorem electronics_store_profit (n : ℕ) (CA : ℝ) : 
  let CB := 2 * CA
  let SA := (2 / 3) * CA
  let SB := 1.2 * CB
  let total_cost := n * CA + n * CB
  let total_sales := n * SA + n * SB
  (total_sales - total_cost) / total_cost = 0.1
  := by sorry

end electronics_store_profit_l1459_145944


namespace one_child_truthful_l1459_145976

structure Child where
  name : String
  truthful : Bool

def grisha_claim (masha sasha natasha : Child) : Prop :=
  masha.truthful ∧ sasha.truthful ∧ natasha.truthful

def contradictions_exist (masha sasha natasha : Child) : Prop :=
  ¬(masha.truthful ∧ sasha.truthful ∧ natasha.truthful)

theorem one_child_truthful (masha sasha natasha : Child) :
  grisha_claim masha sasha natasha →
  contradictions_exist masha sasha natasha →
  ∃! c : Child, c ∈ [masha, sasha, natasha] ∧ c.truthful :=
by
  sorry

#check one_child_truthful

end one_child_truthful_l1459_145976


namespace smoothie_price_l1459_145970

theorem smoothie_price (cake_price : ℚ) (smoothies_sold : ℕ) (cakes_sold : ℕ) (total_revenue : ℚ) :
  cake_price = 2 →
  smoothies_sold = 40 →
  cakes_sold = 18 →
  total_revenue = 156 →
  ∃ (smoothie_price : ℚ), smoothie_price * smoothies_sold + cake_price * cakes_sold = total_revenue ∧ smoothie_price = 3 :=
by sorry

end smoothie_price_l1459_145970


namespace tileIV_in_rectangle_C_l1459_145975

-- Define the tile sides
inductive Side
| Top
| Right
| Bottom
| Left

-- Define the tiles
structure Tile :=
  (id : Nat)
  (top : Nat)
  (right : Nat)
  (bottom : Nat)
  (left : Nat)

-- Define the rectangles
inductive Rectangle
| A
| B
| C
| D

-- Define the placement of tiles
def Placement := Tile → Rectangle

-- Define the adjacency relation between rectangles
def Adjacent : Rectangle → Rectangle → Prop := sorry

-- Define the matching condition for adjacent tiles
def MatchingSides (t1 t2 : Tile) (s1 s2 : Side) : Prop := sorry

-- Define the validity of a placement
def ValidPlacement (p : Placement) : Prop := sorry

-- Define the tiles from the problem
def tileI : Tile := ⟨1, 6, 8, 3, 7⟩
def tileII : Tile := ⟨2, 7, 6, 2, 9⟩
def tileIII : Tile := ⟨3, 5, 1, 9, 0⟩
def tileIV : Tile := ⟨4, 0, 9, 4, 5⟩

-- Theorem statement
theorem tileIV_in_rectangle_C :
  ∀ (p : Placement), ValidPlacement p → p tileIV = Rectangle.C := by
  sorry

end tileIV_in_rectangle_C_l1459_145975


namespace derivative_of_one_plus_cos_l1459_145909

theorem derivative_of_one_plus_cos (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 1 + Real.cos x
  HasDerivAt f (-Real.sin x) x := by sorry

end derivative_of_one_plus_cos_l1459_145909


namespace equation_solutions_l1459_145985

theorem equation_solutions : 
  let f (x : ℝ) := 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5))
  ∀ x : ℝ, f x = 1/8 ↔ x = 7 ∨ x = -2 :=
by sorry

end equation_solutions_l1459_145985


namespace eulerian_circuit_iff_even_degree_l1459_145912

/-- A graph is a pair of a type of vertices and an edge relation -/
structure Graph (V : Type) :=
  (adj : V → V → Prop)

/-- The degree of a vertex in a graph is the number of edges incident to it -/
def degree {V : Type} (G : Graph V) (v : V) : ℕ := sorry

/-- An Eulerian circuit in a graph is a path that traverses every edge exactly once and returns to the starting vertex -/
def has_eulerian_circuit {V : Type} (G : Graph V) : Prop := sorry

/-- Theorem: A graph has an Eulerian circuit if and only if every vertex has even degree -/
theorem eulerian_circuit_iff_even_degree {V : Type} (G : Graph V) :
  has_eulerian_circuit G ↔ ∀ v : V, Even (degree G v) := by sorry

end eulerian_circuit_iff_even_degree_l1459_145912


namespace janet_crayons_l1459_145905

/-- The number of crayons Michelle has initially -/
def michelle_initial : ℕ := 2

/-- The number of crayons Michelle has after Janet gives her all of her crayons -/
def michelle_final : ℕ := 4

/-- The number of crayons Janet has initially -/
def janet_initial : ℕ := michelle_final - michelle_initial

theorem janet_crayons : janet_initial = 2 := by sorry

end janet_crayons_l1459_145905


namespace cos_fourth_power_identity_l1459_145986

theorem cos_fourth_power_identity (θ : ℝ) : 
  (Real.cos θ)^4 = (1/8) * Real.cos (4*θ) + (1/2) * Real.cos (2*θ) + 0 * Real.cos θ := by
  sorry

end cos_fourth_power_identity_l1459_145986


namespace union_A_B_complement_A_intersect_B_A_subset_C_implies_a_greater_than_seven_l1459_145918

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem for question 1
theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

-- Theorem for question 2
theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by sorry

-- Theorem for question 3
theorem A_subset_C_implies_a_greater_than_seven (a : ℝ) : 
  A ⊆ C a → a > 7 := by sorry

end union_A_B_complement_A_intersect_B_A_subset_C_implies_a_greater_than_seven_l1459_145918


namespace no_valid_labeling_l1459_145942

-- Define a type for the vertices of a tetrahedron
inductive Vertex : Type
| A : Vertex
| B : Vertex
| C : Vertex
| D : Vertex

-- Define a type for the faces of a tetrahedron
inductive Face : Type
| ABC : Face
| ABD : Face
| ACD : Face
| BCD : Face

-- Define a labeling function
def Labeling := Vertex → Fin 4

-- Define a function to get the sum of a face given a labeling
def faceSum (l : Labeling) (f : Face) : Nat :=
  match f with
  | Face.ABC => (l Vertex.A).val + (l Vertex.B).val + (l Vertex.C).val
  | Face.ABD => (l Vertex.A).val + (l Vertex.B).val + (l Vertex.D).val
  | Face.ACD => (l Vertex.A).val + (l Vertex.C).val + (l Vertex.D).val
  | Face.BCD => (l Vertex.B).val + (l Vertex.C).val + (l Vertex.D).val

-- Define a predicate for a valid labeling
def isValidLabeling (l : Labeling) : Prop :=
  (∀ (v1 v2 : Vertex), v1 ≠ v2 → l v1 ≠ l v2) ∧
  (∀ (f1 f2 : Face), faceSum l f1 = faceSum l f2)

-- Theorem: There are no valid labelings
theorem no_valid_labeling : ¬∃ (l : Labeling), isValidLabeling l := by
  sorry


end no_valid_labeling_l1459_145942


namespace fib_150_mod_9_l1459_145953

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- The period of the Fibonacci sequence modulo 9 -/
def fib_mod_9_period : ℕ := 24

theorem fib_150_mod_9 :
  fib 149 % 9 = 8 := by
  sorry

end fib_150_mod_9_l1459_145953


namespace floor_expression_l1459_145972

theorem floor_expression (n : ℕ) (h : n = 2009) : 
  ⌊((n + 1)^3 / ((n - 1) * n : ℝ) - (n - 1)^3 / (n * (n + 1) : ℝ))⌋ = 8 := by
  sorry

end floor_expression_l1459_145972


namespace largest_prime_diff_126_l1459_145957

/-- Two natural numbers are different if they are not equal -/
def different (a b : ℕ) : Prop := a ≠ b

/-- A natural number is even if it's divisible by 2 -/
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- The largest prime difference for 126 -/
theorem largest_prime_diff_126 : 
  ∃ (p q : ℕ), 
    Prime p ∧ 
    Prime q ∧ 
    different p q ∧
    p + q = 126 ∧
    even 126 ∧ 
    126 > 7 ∧
    ∀ (r s : ℕ), Prime r → Prime s → different r s → r + s = 126 → s - r ≤ 100 :=
sorry

end largest_prime_diff_126_l1459_145957


namespace special_function_value_at_neg_three_l1459_145958

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y) ∧ (f 1 = 2)

theorem special_function_value_at_neg_three 
  (f : ℝ → ℝ) (h : special_function f) : f (-3) = -12 := by
  sorry

end special_function_value_at_neg_three_l1459_145958


namespace paintings_distribution_l1459_145961

theorem paintings_distribution (total_paintings : ℕ) (num_rooms : ℕ) (paintings_per_room : ℕ) :
  total_paintings = 32 →
  num_rooms = 4 →
  total_paintings = num_rooms * paintings_per_room →
  paintings_per_room = 8 := by
  sorry

end paintings_distribution_l1459_145961


namespace equation_solution_l1459_145959

theorem equation_solution (k : ℝ) : (∃ x : ℝ, 2 * x + k - 3 = 6 ∧ x = 3) → k = 3 := by
  sorry

end equation_solution_l1459_145959


namespace no_positive_integer_solutions_l1459_145930

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + y^2 + 1 = x^3 := by
  sorry

end no_positive_integer_solutions_l1459_145930


namespace three_prime_pairs_sum_52_l1459_145916

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given number -/
def count_prime_pairs (sum : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (sum - p)) (Finset.range (sum / 2 + 1))).card / 2

/-- Theorem stating that there are exactly 3 unordered pairs of prime numbers that sum to 52 -/
theorem three_prime_pairs_sum_52 : count_prime_pairs 52 = 3 := by
  sorry

end three_prime_pairs_sum_52_l1459_145916


namespace cos_300_degrees_l1459_145999

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_degrees_l1459_145999


namespace complex_real_roots_relationship_l1459_145982

theorem complex_real_roots_relationship (a : ℝ) : 
  ¬(∀ x : ℂ, x^2 + a*x - a = 0 → (∀ y : ℝ, y^2 - a*y + a ≠ 0)) ∧
  ¬(∀ y : ℝ, y^2 - a*y + a = 0 → (∀ x : ℂ, x^2 + a*x - a ≠ 0)) :=
by sorry

end complex_real_roots_relationship_l1459_145982


namespace dana_marcus_pencil_difference_l1459_145907

/-- Given that Dana has 15 more pencils than Jayden, Jayden has twice as many pencils as Marcus,
    and Jayden has 20 pencils, prove that Dana has 25 more pencils than Marcus. -/
theorem dana_marcus_pencil_difference :
  ∀ (dana jayden marcus : ℕ),
  dana = jayden + 15 →
  jayden = 2 * marcus →
  jayden = 20 →
  dana - marcus = 25 := by
sorry

end dana_marcus_pencil_difference_l1459_145907


namespace f_is_even_l1459_145933

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Theorem: f is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end f_is_even_l1459_145933


namespace sales_increase_l1459_145908

theorem sales_increase (P : ℝ) (N : ℝ) (h1 : P > 0) (h2 : N > 0) :
  let discount_rate : ℝ := 0.1
  let income_increase_rate : ℝ := 0.08
  let new_price : ℝ := P * (1 - discount_rate)
  let N' : ℝ := N * (1 + income_increase_rate) / (1 - discount_rate)
  (N' - N) / N = 0.2 :=
by sorry

end sales_increase_l1459_145908


namespace walter_age_2005_l1459_145945

theorem walter_age_2005 (walter_age_2000 : ℕ) (grandmother_age_2000 : ℕ) : 
  walter_age_2000 = grandmother_age_2000 / 3 →
  (2000 - walter_age_2000) + (2000 - grandmother_age_2000) = 3896 →
  walter_age_2000 + 5 = 31 :=
by
  sorry

end walter_age_2005_l1459_145945


namespace log_stack_theorem_l1459_145983

/-- Represents a stack of logs -/
structure LogStack where
  bottom_row : ℕ
  top_row : ℕ
  row_difference : ℕ

/-- Calculates the number of rows in the log stack -/
def num_rows (stack : LogStack) : ℕ :=
  (stack.bottom_row - stack.top_row) / stack.row_difference + 1

/-- Calculates the total number of logs in the stack -/
def total_logs (stack : LogStack) : ℕ :=
  (num_rows stack * (stack.bottom_row + stack.top_row)) / 2

/-- The main theorem about the log stack -/
theorem log_stack_theorem (stack : LogStack) 
  (h1 : stack.bottom_row = 15)
  (h2 : stack.top_row = 5)
  (h3 : stack.row_difference = 2) :
  total_logs stack = 60 ∧ stack.top_row = 5 := by
  sorry

end log_stack_theorem_l1459_145983


namespace final_milk_composition_l1459_145963

/-- The percentage of milk remaining after each replacement operation -/
def replacement_factor : ℝ := 0.7

/-- The number of replacement operations performed -/
def num_operations : ℕ := 5

/-- The final percentage of milk in the container after all operations -/
def final_milk_percentage : ℝ := replacement_factor ^ num_operations * 100

/-- Theorem stating the final percentage of milk after the operations -/
theorem final_milk_composition :
  ∃ ε > 0, |final_milk_percentage - 16.807| < ε :=
sorry

end final_milk_composition_l1459_145963


namespace system_solution_l1459_145939

theorem system_solution (x y z t : ℝ) : 
  (x * y - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18) → 
  ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) := by
  sorry

end system_solution_l1459_145939


namespace min_abs_diff_sum_l1459_145962

theorem min_abs_diff_sum (x a b : ℚ) : 
  x ≠ a ∧ x ≠ b ∧ a ≠ b → 
  a > b → 
  (∀ y : ℚ, |y - a| + |y - b| ≥ 2) ∧ (∃ z : ℚ, |z - a| + |z - b| = 2) →
  2022 + a - b = 2024 := by
sorry

end min_abs_diff_sum_l1459_145962


namespace hearing_aid_cost_proof_l1459_145910

/-- The cost of a single hearing aid -/
def hearing_aid_cost : ℝ := 2500

/-- The insurance coverage percentage -/
def insurance_coverage : ℝ := 0.80

/-- The amount John pays for both hearing aids -/
def john_payment : ℝ := 1000

/-- Theorem stating that the cost of each hearing aid is $2500 -/
theorem hearing_aid_cost_proof : 
  (1 - insurance_coverage) * (2 * hearing_aid_cost) = john_payment := by
  sorry

end hearing_aid_cost_proof_l1459_145910


namespace smallest_n_congruence_two_satisfies_congruence_smallest_n_is_two_l1459_145947

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 721 * n ≡ 1137 * n [ZMOD 30] → n ≥ 2 :=
sorry

theorem two_satisfies_congruence : 721 * 2 ≡ 1137 * 2 [ZMOD 30] :=
sorry

theorem smallest_n_is_two : 
  ∃ (n : ℕ), n > 0 ∧ 721 * n ≡ 1137 * n [ZMOD 30] ∧ 
  ∀ (m : ℕ), m > 0 ∧ 721 * m ≡ 1137 * m [ZMOD 30] → n ≤ m :=
sorry

end smallest_n_congruence_two_satisfies_congruence_smallest_n_is_two_l1459_145947


namespace complex_magnitude_l1459_145964

-- Define complex numbers w and z
variable (w z : ℂ)

-- Define the given conditions
theorem complex_magnitude (h1 : w * z = 20 - 15 * I) (h2 : Complex.abs w = 5) :
  Complex.abs z = 5 := by
  sorry

end complex_magnitude_l1459_145964


namespace product_prs_is_96_l1459_145968

theorem product_prs_is_96 (p r s : ℕ) 
  (hp : 4^p - 4^3 = 192)
  (hr : 3^r + 81 = 162)
  (hs : 7^s - 7^2 = 3994) :
  p * r * s = 96 := by
  sorry

end product_prs_is_96_l1459_145968


namespace delphine_chocolates_day1_l1459_145978

/-- Represents the number of chocolates Delphine ate on each day -/
structure ChocolatesEaten where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Theorem stating the number of chocolates Delphine ate on the first day -/
theorem delphine_chocolates_day1 (c : ChocolatesEaten) : c.day1 = 4 :=
  by
  have h1 : c.day2 = 2 * c.day1 - 3 := sorry
  have h2 : c.day3 = c.day1 - 2 := sorry
  have h3 : c.day4 = c.day3 - 1 := sorry
  have h4 : c.day1 + c.day2 + c.day3 + c.day4 + 12 = 24 := sorry
  sorry

#check delphine_chocolates_day1

end delphine_chocolates_day1_l1459_145978


namespace simplify_and_evaluate_l1459_145902

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 + 3 / (a - 2)) / ((a^2 + 2*a + 1) / (a - 2)) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l1459_145902


namespace wrapping_paper_area_l1459_145913

/-- Calculates the area of a square sheet of wrapping paper needed to wrap a rectangular box -/
theorem wrapping_paper_area (box_length box_width box_height extra_fold : ℝ) :
  box_length = 10 ∧ box_width = 10 ∧ box_height = 5 ∧ extra_fold = 2 →
  (box_width / 2 + box_height + extra_fold) ^ 2 = 144 := by
  sorry

end wrapping_paper_area_l1459_145913


namespace power_of_negative_square_l1459_145979

theorem power_of_negative_square (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end power_of_negative_square_l1459_145979


namespace fraction_undefined_l1459_145952

theorem fraction_undefined (x : ℚ) : (2 * x + 1 = 0) ↔ (x = -1/2) := by
  sorry

end fraction_undefined_l1459_145952


namespace snow_leopard_arrangement_l1459_145984

theorem snow_leopard_arrangement (n : ℕ) (h : n = 9) : 
  (2 * Nat.factorial (n - 3)) = 1440 := by
  sorry

end snow_leopard_arrangement_l1459_145984


namespace polynomial_sum_theorem_l1459_145989

theorem polynomial_sum_theorem (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^10 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  (a₀ + a₁) + (a₀ + a₂) + (a₀ + a₃) + (a₀ + a₄) + (a₀ + a₅) + (a₀ + a₆) + (a₀ + a₇) + (a₀ + a₈) + (a₀ + a₉) + (a₀ + a₁₀) = 10 :=
by
  sorry

end polynomial_sum_theorem_l1459_145989


namespace range_of_t_t_value_for_diameter_6_l1459_145973

-- Define the equation of the circle
def circle_equation (x y t : ℝ) : Prop :=
  x^2 + y^2 + (Real.sqrt 3 * t + 1) * x + t * y + t^2 - 2 = 0

-- Theorem for the range of t
theorem range_of_t :
  ∀ t : ℝ, (∃ x y : ℝ, circle_equation x y t) → t > -(3 * Real.sqrt 3) / 2 :=
sorry

-- Theorem for the value of t when diameter is 6
theorem t_value_for_diameter_6 :
  ∃! t : ℝ, (∃ x y : ℝ, circle_equation x y t) ∧ 
  (∃ x₁ y₁ x₂ y₂ : ℝ, circle_equation x₁ y₁ t ∧ circle_equation x₂ y₂ t ∧ 
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6) ∧
  t = (9 * Real.sqrt 3) / 2 :=
sorry

end range_of_t_t_value_for_diameter_6_l1459_145973


namespace three_integer_solutions_quadratic_inequality_l1459_145937

theorem three_integer_solutions_quadratic_inequality (b : ℤ) : 
  (∃! n : ℕ, n = 2 ∧ 
    (∃ s : Finset ℤ, s.card = n ∧ 
      (∀ b' ∈ s, (∃! t : Finset ℤ, t.card = 3 ∧ 
        (∀ x ∈ t, x^2 + b' * x + 6 ≤ 0) ∧ 
        (∀ x : ℤ, x^2 + b' * x + 6 ≤ 0 → x ∈ t))))) :=
sorry

end three_integer_solutions_quadratic_inequality_l1459_145937


namespace same_terminal_side_330_neg_30_l1459_145996

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

/-- The angle -30° -/
def angle_neg_30 : ℝ := -30

/-- The angle 330° -/
def angle_330 : ℝ := 330

/-- Theorem: 330° has the same terminal side as -30° -/
theorem same_terminal_side_330_neg_30 :
  same_terminal_side angle_330 angle_neg_30 := by
  sorry

end same_terminal_side_330_neg_30_l1459_145996
