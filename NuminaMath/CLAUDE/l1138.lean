import Mathlib

namespace NUMINAMATH_CALUDE_largest_prime_2015_digits_square_minus_one_div_15_l1138_113827

/-- The largest prime with 2015 digits -/
def p : ℕ := sorry

/-- p is prime -/
axiom p_prime : Nat.Prime p

/-- p has 2015 digits -/
axiom p_digits : 10^2014 ≤ p ∧ p < 10^2015

/-- p is the largest such prime -/
axiom p_largest : ∀ q : ℕ, Nat.Prime q → 10^2014 ≤ q ∧ q < 10^2015 → q ≤ p

theorem largest_prime_2015_digits_square_minus_one_div_15 : 15 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_2015_digits_square_minus_one_div_15_l1138_113827


namespace NUMINAMATH_CALUDE_advance_tickets_sold_l1138_113870

theorem advance_tickets_sold (advance_cost same_day_cost total_tickets total_receipts : ℕ) 
  (h1 : advance_cost = 20)
  (h2 : same_day_cost = 30)
  (h3 : total_tickets = 60)
  (h4 : total_receipts = 1600) :
  ∃ (advance_sold : ℕ), 
    advance_sold * advance_cost + (total_tickets - advance_sold) * same_day_cost = total_receipts ∧ 
    advance_sold = 20 :=
by sorry

end NUMINAMATH_CALUDE_advance_tickets_sold_l1138_113870


namespace NUMINAMATH_CALUDE_tourist_guide_distribution_l1138_113848

theorem tourist_guide_distribution :
  let n_tourists : ℕ := 8
  let n_guides : ℕ := 3
  let total_distributions := n_guides ^ n_tourists
  let at_least_one_empty := n_guides * (n_guides - 1) ^ n_tourists
  let at_least_two_empty := n_guides * 1 ^ n_tourists
  total_distributions - at_least_one_empty + at_least_two_empty = 5796 :=
by sorry

end NUMINAMATH_CALUDE_tourist_guide_distribution_l1138_113848


namespace NUMINAMATH_CALUDE_gcd_equality_from_division_l1138_113828

theorem gcd_equality_from_division (a b q r : ℤ) :
  b > 0 →
  0 ≤ r →
  r < b →
  a = b * q + r →
  Int.gcd a b = Int.gcd b r := by
  sorry

end NUMINAMATH_CALUDE_gcd_equality_from_division_l1138_113828


namespace NUMINAMATH_CALUDE_margo_travel_distance_l1138_113897

/-- The total distance Margo traveled given her jogging and walking times and average speed -/
theorem margo_travel_distance (jog_time walk_time avg_speed : ℝ) : 
  jog_time = 12 / 60 →
  walk_time = 25 / 60 →
  avg_speed = 5 →
  avg_speed * (jog_time + walk_time) = 3.085 :=
by sorry

end NUMINAMATH_CALUDE_margo_travel_distance_l1138_113897


namespace NUMINAMATH_CALUDE_bathroom_volume_l1138_113831

theorem bathroom_volume (length width height area volume : ℝ) : 
  length = 4 →
  area = 8 →
  height = 7 →
  area = length * width →
  volume = length * width * height →
  volume = 56 := by
sorry

end NUMINAMATH_CALUDE_bathroom_volume_l1138_113831


namespace NUMINAMATH_CALUDE_employee_hire_year_l1138_113845

/-- Rule of 70 retirement provision -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := sorry

/-- The employee's age when hired -/
def hire_age : ℕ := 32

/-- The year the employee became eligible for retirement -/
def retirement_year : ℕ := 2007

theorem employee_hire_year :
  rule_of_70 (hire_age + (retirement_year - hire_year)) (retirement_year - hire_year) ∧
  hire_year = 1969 := by
  sorry

end NUMINAMATH_CALUDE_employee_hire_year_l1138_113845


namespace NUMINAMATH_CALUDE_perpendicular_vectors_tan_2x_l1138_113812

theorem perpendicular_vectors_tan_2x (x : ℝ) : 
  let a : ℝ × ℝ := (Real.cos x, Real.sin x)
  let b : ℝ × ℝ := (Real.sqrt 3, -1)
  (a.1 * b.1 + a.2 * b.2 = 0) → Real.tan (2 * x) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_tan_2x_l1138_113812


namespace NUMINAMATH_CALUDE_maintenance_cost_third_year_l1138_113884

/-- Represents the maintenance cost function for factory equipment -/
def maintenance_cost (x : ℝ) : ℝ := 0.8 * x + 1.5

/-- Proves that the maintenance cost for equipment in its third year is 3.9 ten thousand yuan -/
theorem maintenance_cost_third_year :
  maintenance_cost 3 = 3.9 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_cost_third_year_l1138_113884


namespace NUMINAMATH_CALUDE_sin_alpha_plus_7pi_over_6_l1138_113883

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) 
  (h : Real.cos (α - π / 6) + Real.sin α = (4 / 5) * Real.sqrt 3) : 
  Real.sin (α + 7 * π / 6) = -(4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_7pi_over_6_l1138_113883


namespace NUMINAMATH_CALUDE_rain_forest_animal_count_l1138_113825

/-- The number of animals in the Rain Forest exhibit -/
def rain_forest_animals : ℕ := 7

/-- The number of animals in the Reptile House -/
def reptile_house_animals : ℕ := 16

/-- Theorem stating the relationship between the number of animals in the Rain Forest exhibit and the Reptile House -/
theorem rain_forest_animal_count : 
  reptile_house_animals = 3 * rain_forest_animals - 5 ∧ 
  rain_forest_animals = 7 := by
  sorry

end NUMINAMATH_CALUDE_rain_forest_animal_count_l1138_113825


namespace NUMINAMATH_CALUDE_prob_2012_higher_than_2011_l1138_113864

/-- Probability of guessing the correct answer to each question -/
def p : ℝ := 0.25

/-- Probability of guessing the incorrect answer to each question -/
def q : ℝ := 1 - p

/-- Number of questions in the 2011 exam -/
def n_2011 : ℕ := 20

/-- Number of correct answers required to pass in 2011 -/
def k_2011 : ℕ := 3

/-- Number of questions in the 2012 exam -/
def n_2012 : ℕ := 40

/-- Number of correct answers required to pass in 2012 -/
def k_2012 : ℕ := 6

/-- Probability of passing the exam in 2011 -/
def prob_2011 : ℝ := 1 - (Finset.sum (Finset.range k_2011) (λ i => Nat.choose n_2011 i * p^i * q^(n_2011 - i)))

/-- Probability of passing the exam in 2012 -/
def prob_2012 : ℝ := 1 - (Finset.sum (Finset.range k_2012) (λ i => Nat.choose n_2012 i * p^i * q^(n_2012 - i)))

/-- Theorem stating that the probability of passing in 2012 is higher than in 2011 -/
theorem prob_2012_higher_than_2011 : prob_2012 > prob_2011 := by
  sorry

end NUMINAMATH_CALUDE_prob_2012_higher_than_2011_l1138_113864


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1138_113842

/-- The perimeter of a rectangle with area 500 cm² and one side 25 cm is 90 cm. -/
theorem rectangle_perimeter (a b : ℝ) (h_area : a * b = 500) (h_side : a = 25) : 
  2 * (a + b) = 90 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1138_113842


namespace NUMINAMATH_CALUDE_sin_cos_equation_integer_solution_l1138_113869

theorem sin_cos_equation_integer_solution (x : ℤ) :
  (∃ t : ℤ, x = 4 * t + 1 ∨ x = 4 * t - 1) ↔ 
  Real.sin (π * (2 * ↑x - 1)) = Real.cos (π * ↑x / 2) :=
sorry

end NUMINAMATH_CALUDE_sin_cos_equation_integer_solution_l1138_113869


namespace NUMINAMATH_CALUDE_certain_number_proof_l1138_113800

theorem certain_number_proof : ∃ n : ℕ, (73 * n) % 8 = 7 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1138_113800


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1138_113821

/-- The quadratic function f(x) = 3x^2 - 6x + 5 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := 2

/-- Theorem: The vertex of the quadratic function f(x) = 3x^2 - 6x + 5 is at the point (1, 2) -/
theorem quadratic_vertex :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1138_113821


namespace NUMINAMATH_CALUDE_danny_drive_to_work_l1138_113865

/-- Represents the distance Danny drives between different locations -/
structure DannyDrive where
  x : ℝ  -- Distance from Danny's house to the first friend's house
  first_to_second : ℝ := 0.5 * x
  second_to_third : ℝ := 2 * x
  third_to_fourth : ℝ  -- Will be calculated
  fourth_to_work : ℝ   -- To be proven

/-- Calculates the total distance driven up to the third friend's house -/
def total_to_third (d : DannyDrive) : ℝ :=
  d.x + d.first_to_second + d.second_to_third

/-- Theorem stating the distance Danny drives between the fourth friend's house and work -/
theorem danny_drive_to_work (d : DannyDrive) 
    (h1 : d.third_to_fourth = (1/3) * total_to_third d) 
    (h2 : d.fourth_to_work = 3 * (total_to_third d + d.third_to_fourth)) : 
  d.fourth_to_work = 14 * d.x := by
  sorry


end NUMINAMATH_CALUDE_danny_drive_to_work_l1138_113865


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_nine_l1138_113807

theorem product_of_repeating_decimal_and_nine : ∃ (s : ℚ),
  (∀ (n : ℕ), s * 10^(3*n) - s * 10^(3*n-3) = 123 * 10^(3*n-3)) ∧
  s * 9 = 41 / 37 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_nine_l1138_113807


namespace NUMINAMATH_CALUDE_luke_spent_3_dollars_per_week_l1138_113840

def luke_problem (lawn_income weed_income total_weeks : ℕ) : Prop :=
  let total_income := lawn_income + weed_income
  total_income / total_weeks = 3

theorem luke_spent_3_dollars_per_week :
  luke_problem 9 18 9 := by
  sorry

end NUMINAMATH_CALUDE_luke_spent_3_dollars_per_week_l1138_113840


namespace NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l1138_113834

/-- Represents a bookstore inventory -/
structure Bookstore where
  total : ℕ
  historical_fiction : ℕ
  historical_fiction_new : ℕ
  other_new : ℕ

/-- Conditions for Joel's bookstore -/
def joels_bookstore (b : Bookstore) : Prop :=
  b.historical_fiction = (2 * b.total) / 5 ∧
  b.historical_fiction_new = (2 * b.historical_fiction) / 5 ∧
  b.other_new = (2 * (b.total - b.historical_fiction)) / 5

/-- Theorem: In Joel's bookstore, 2/5 of all new releases are historical fiction -/
theorem historical_fiction_new_releases_fraction (b : Bookstore) 
  (h : joels_bookstore b) : 
  (b.historical_fiction_new : ℚ) / (b.historical_fiction_new + b.other_new) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l1138_113834


namespace NUMINAMATH_CALUDE_min_distance_AD_l1138_113838

/-- Given points A, B, C, D, E in a metric space, prove that the minimum distance between A and D is 2, given the distances between other points. -/
theorem min_distance_AD (X : Type*) [MetricSpace X] (A B C D E : X) : 
  dist A B = 12 →
  dist B C = 7 →
  dist C E = 2 →
  dist E D = 5 →
  ∃ (d : ℝ), d ≥ 2 ∧ dist A D ≥ d := by
  sorry

end NUMINAMATH_CALUDE_min_distance_AD_l1138_113838


namespace NUMINAMATH_CALUDE_f_30_value_l1138_113892

/-- A function from positive integers to positive integers satisfying certain properties -/
def special_function (f : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, f (n + 1) > f n) ∧ 
  (∀ m n : ℕ+, f (m * n) = f m * f n) ∧
  (∀ m n : ℕ+, m ≠ n → m ^ m.val = n ^ n.val → (f m = n ∨ f n = m))

theorem f_30_value (f : ℕ+ → ℕ+) (h : special_function f) : f 30 = 900 := by
  sorry

end NUMINAMATH_CALUDE_f_30_value_l1138_113892


namespace NUMINAMATH_CALUDE_linear_system_integer_solution_l1138_113822

theorem linear_system_integer_solution (a b : ℤ) :
  ∃ (x y z t : ℤ), x + y + 2*z + 2*t = a ∧ 2*x - 2*y + z - t = b := by
sorry

end NUMINAMATH_CALUDE_linear_system_integer_solution_l1138_113822


namespace NUMINAMATH_CALUDE_ellipse_equation_l1138_113885

/-- An ellipse centered at the origin -/
structure Ellipse where
  equation : ℝ → ℝ → Prop

/-- A hyperbola -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- The eccentricity of a conic section -/
def eccentricity (c : ℝ) : ℝ := c

/-- Theorem: Given an ellipse centered at the origin sharing a common focus with 
    the hyperbola 2x^2 - 2y^2 = 1, and their eccentricities being reciprocal to 
    each other, the equation of the ellipse is x^2/2 + y^2 = 1 -/
theorem ellipse_equation 
  (e : Ellipse) 
  (h : Hyperbola) 
  (h_eq : h.equation = fun x y => 2 * x^2 - 2 * y^2 = 1) 
  (common_focus : ∃ (f : ℝ × ℝ), f ∈ {p | ∃ (x y : ℝ), p = (x, y) ∧ h.equation x y} ∧ 
                                 f ∈ {p | ∃ (x y : ℝ), p = (x, y) ∧ e.equation x y})
  (reciprocal_eccentricity : ∃ (e_ecc h_ecc : ℝ), 
    eccentricity e_ecc * eccentricity h_ecc = 1) :
  e.equation = fun x y => x^2 / 2 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1138_113885


namespace NUMINAMATH_CALUDE_alice_ball_probability_l1138_113899

/-- Probability of Alice tossing the ball to Bob -/
def alice_toss_prob : ℚ := 1/3

/-- Probability of Alice keeping the ball -/
def alice_keep_prob : ℚ := 2/3

/-- Probability of Bob tossing the ball to Alice -/
def bob_toss_prob : ℚ := 1/4

/-- Probability of Bob keeping the ball -/
def bob_keep_prob : ℚ := 3/4

/-- Alice starts with the ball -/
def alice_starts : Prop := True

/-- The probability that Alice has the ball after two turns -/
def prob_alice_after_two_turns : ℚ := 37/108

theorem alice_ball_probability :
  alice_starts →
  prob_alice_after_two_turns = alice_keep_prob * alice_keep_prob + alice_toss_prob * bob_toss_prob :=
by sorry

end NUMINAMATH_CALUDE_alice_ball_probability_l1138_113899


namespace NUMINAMATH_CALUDE_william_wins_l1138_113817

theorem william_wins (total_rounds : ℕ) (william_advantage : ℕ) (william_wins : ℕ) : 
  total_rounds = 15 → 
  william_advantage = 5 → 
  william_wins = total_rounds / 2 + william_advantage → 
  william_wins = 10 := by
sorry

end NUMINAMATH_CALUDE_william_wins_l1138_113817


namespace NUMINAMATH_CALUDE_sum_of_squares_geq_sum_of_products_sqrt_inequality_l1138_113881

-- Statement 1
theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ a*b + a*c + b*c :=
sorry

-- Statement 2
theorem sqrt_inequality :
  Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_geq_sum_of_products_sqrt_inequality_l1138_113881


namespace NUMINAMATH_CALUDE_joel_donation_l1138_113833

/-- The number of toys Joel donated -/
def joels_toys : ℕ := 22

/-- The number of toys Joel's sister donated -/
def sisters_toys : ℕ := 11

/-- The number of toys Joel's friends donated -/
def friends_toys : ℕ := 75

/-- The total number of donated toys -/
def total_toys : ℕ := 108

theorem joel_donation :
  (friends_toys + sisters_toys + joels_toys = total_toys) ∧
  (joels_toys = 2 * sisters_toys) ∧
  (friends_toys = 18 + 42 + 2 + 13) :=
by sorry

end NUMINAMATH_CALUDE_joel_donation_l1138_113833


namespace NUMINAMATH_CALUDE_three_lines_intersection_l1138_113887

/-- Three lines intersect at a single point if and only if k = -7 -/
theorem three_lines_intersection (x y k : ℝ) : 
  (∃! p : ℝ × ℝ, (y = 7*x + 5 ∧ y = -3*x - 35 ∧ y = 4*x + k) → p.1 = x ∧ p.2 = y) ↔ k = -7 :=
by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l1138_113887


namespace NUMINAMATH_CALUDE_expression_simplification_l1138_113867

theorem expression_simplification :
  let a := Real.sqrt 2
  let b := Real.sqrt 3
  b > a ∧ (8 : ℝ) ^ (1/3) = 2 →
  |a - b| + (8 : ℝ) ^ (1/3) - a * (a - 1) = b := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1138_113867


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1138_113829

/-- Arithmetic sequence a_i -/
def a (d i : ℕ) : ℕ := 1 + 2 * (i - 1) * d

/-- Arithmetic sequence b_i -/
def b (d i : ℕ) : ℕ := 1 + (i - 1) * d

/-- Sum of first k terms of a_i -/
def s (d k : ℕ) : ℕ := k + k * (k - 1) * d

/-- Sum of first k terms of b_i -/
def t (d k : ℕ) : ℕ := k + k * (k - 1) * (d / 2)

/-- A_n sequence -/
def A (d n : ℕ) : ℕ := s d (t d n)

/-- B_n sequence -/
def B (d n : ℕ) : ℕ := t d (s d n)

/-- Main theorem -/
theorem arithmetic_sequence_difference (d n : ℕ) :
  A d (n + 1) - A d n = (1 + n * d)^3 ∧
  B d (n + 1) - B d n = (n * d)^3 + (1 + n * d)^3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1138_113829


namespace NUMINAMATH_CALUDE_log_less_than_one_range_l1138_113862

theorem log_less_than_one_range (a : ℝ) :
  (∃ (x : ℝ), Real.log x / Real.log a < 1) → a ∈ Set.union (Set.Ioo 0 1) (Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_log_less_than_one_range_l1138_113862


namespace NUMINAMATH_CALUDE_equivalence_of_equations_l1138_113850

theorem equivalence_of_equations (p : ℕ) (hp : Nat.Prime p) :
  (∃ (x s : ℤ), x^2 - x + 3 - p * s = 0) ↔
  (∃ (y t : ℤ), y^2 - y + 25 - p * t = 0) := by
sorry

end NUMINAMATH_CALUDE_equivalence_of_equations_l1138_113850


namespace NUMINAMATH_CALUDE_reciprocal_problem_l1138_113882

theorem reciprocal_problem (x : ℚ) : 8 * x = 6 → 60 * (1 / x) = 80 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l1138_113882


namespace NUMINAMATH_CALUDE_walts_investment_rate_l1138_113809

/-- Given Walt's investment scenario, prove the unknown interest rate --/
theorem walts_investment_rate : 
  let total_amount : ℝ := 9000
  let known_rate : ℝ := 0.08
  let known_investment : ℝ := 4000
  let total_interest : ℝ := 770
  let unknown_investment : ℝ := total_amount - known_investment
  ∃ (unknown_rate : ℝ),
    known_investment * known_rate + unknown_investment * unknown_rate = total_interest ∧
    unknown_rate = 0.09
  := by sorry

end NUMINAMATH_CALUDE_walts_investment_rate_l1138_113809


namespace NUMINAMATH_CALUDE_sum_and_divide_theorem_l1138_113853

theorem sum_and_divide_theorem (n a : ℕ) (ha : a > 1) :
  let sum := (n * (n + 1)) / 2 - (n / a) * ((n / a) * a + a) / 2
  sum / (a * (a - 1) / 2) = (n / a)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_divide_theorem_l1138_113853


namespace NUMINAMATH_CALUDE_science_club_team_selection_l1138_113898

theorem science_club_team_selection (total_boys : Nat) (total_girls : Nat) 
  (selected_boys : Nat) (selected_girls : Nat) :
  total_boys = 10 → total_girls = 12 → selected_boys = 5 → selected_girls = 3 →
  (Nat.choose total_boys selected_boys) * (Nat.choose total_girls selected_girls) = 55440 := by
  sorry

end NUMINAMATH_CALUDE_science_club_team_selection_l1138_113898


namespace NUMINAMATH_CALUDE_sum_even_implies_one_even_l1138_113875

theorem sum_even_implies_one_even (a b c : ℕ) :
  Even (a + b + c) → ¬(Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end NUMINAMATH_CALUDE_sum_even_implies_one_even_l1138_113875


namespace NUMINAMATH_CALUDE_probability_of_specific_selection_l1138_113895

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 8

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 7

/-- The number of jackets in the drawer -/
def num_jackets : ℕ := 3

/-- The total number of clothing items in the drawer -/
def total_items : ℕ := num_shirts + num_shorts + num_socks + num_jackets

/-- The number of items to be selected -/
def items_to_select : ℕ := 4

theorem probability_of_specific_selection :
  (num_shirts : ℚ) * num_shorts * num_socks * num_jackets /
  (total_items.choose items_to_select) = 144 / 1815 :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_selection_l1138_113895


namespace NUMINAMATH_CALUDE_largest_integer_solution_inequality_l1138_113856

theorem largest_integer_solution_inequality (x : ℤ) :
  (∀ y : ℤ, -y ≥ 2*y + 3 → y ≤ -1) ∧ (-(-1) ≥ 2*(-1) + 3) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_solution_inequality_l1138_113856


namespace NUMINAMATH_CALUDE_number_added_after_doubling_l1138_113863

theorem number_added_after_doubling (x : ℕ) (y : ℕ) (h : x = 13) :
  3 * (2 * x + y) = 99 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_added_after_doubling_l1138_113863


namespace NUMINAMATH_CALUDE_finite_sequence_k_value_l1138_113852

/-- A finite sequence with k terms satisfying the given conditions -/
def FiniteSequence (k : ℕ) (a : ℕ → ℝ) : Prop :=
  (∀ n ∈ Finset.range (k - 2), a (n + 2) = a n - (n + 1) / a (n + 1)) ∧
  a 1 = 24 ∧
  a 2 = 51 ∧
  a k = 0

/-- The theorem stating that k must be 50 for the given conditions -/
theorem finite_sequence_k_value :
  ∀ k : ℕ, ∀ a : ℕ → ℝ, FiniteSequence k a → k = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_finite_sequence_k_value_l1138_113852


namespace NUMINAMATH_CALUDE_mixture_problem_l1138_113872

/-- Proves that the percentage of the first solution is 30% given the conditions of the mixture problem. -/
theorem mixture_problem (total_volume : ℝ) (result_percentage : ℝ) (second_solution_percentage : ℝ)
  (first_solution_volume : ℝ) (second_solution_volume : ℝ)
  (h1 : total_volume = 40)
  (h2 : result_percentage = 45)
  (h3 : second_solution_percentage = 80)
  (h4 : first_solution_volume = 28)
  (h5 : second_solution_volume = 12)
  (h6 : total_volume = first_solution_volume + second_solution_volume)
  (h7 : result_percentage / 100 * total_volume =
        (first_solution_percentage / 100 * first_solution_volume) +
        (second_solution_percentage / 100 * second_solution_volume)) :
  first_solution_percentage = 30 :=
sorry

end NUMINAMATH_CALUDE_mixture_problem_l1138_113872


namespace NUMINAMATH_CALUDE_mingyoungs_math_score_l1138_113804

theorem mingyoungs_math_score 
  (korean : ℝ) 
  (english : ℝ) 
  (math : ℝ) 
  (h1 : (korean + english) / 2 = 89) 
  (h2 : (korean + english + math) / 3 = 91) : 
  math = 95 :=
sorry

end NUMINAMATH_CALUDE_mingyoungs_math_score_l1138_113804


namespace NUMINAMATH_CALUDE_confidence_95_error_5_l1138_113841

/-- Represents the confidence level as a real number between 0 and 1 -/
def ConfidenceLevel : Type := {r : ℝ // 0 < r ∧ r < 1}

/-- Represents the probability of making an incorrect inference -/
def ErrorProbability : Type := {r : ℝ // 0 ≤ r ∧ r ≤ 1}

/-- Given a confidence level, calculates the probability of making an incorrect inference -/
def calculateErrorProbability (cl : ConfidenceLevel) : ErrorProbability :=
  sorry

/-- The theorem states that for a 95% confidence level, the error probability is 5% -/
theorem confidence_95_error_5 :
  let cl95 : ConfidenceLevel := ⟨0.95, by sorry⟩
  calculateErrorProbability cl95 = ⟨0.05, by sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_confidence_95_error_5_l1138_113841


namespace NUMINAMATH_CALUDE_draw_four_from_fifteen_l1138_113879

/-- The number of ways to draw k balls from n balls without replacement -/
def drawBalls (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k > n then 0
  else n * drawBalls (n - 1) (k - 1)

theorem draw_four_from_fifteen :
  drawBalls 15 4 = 32760 := by
  sorry

end NUMINAMATH_CALUDE_draw_four_from_fifteen_l1138_113879


namespace NUMINAMATH_CALUDE_five_line_intersections_l1138_113839

/-- Represents a configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ
  no_three_point_intersection : Bool

/-- The maximum number of intersections for n lines -/
def max_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the impossibility of 11 intersections and possibility of 9 intersections -/
theorem five_line_intersections (config : LineConfiguration) :
  config.num_lines = 5 ∧ config.no_three_point_intersection = true →
  (config.num_intersections ≠ 11 ∧ 
   ∃ (config' : LineConfiguration), 
     config'.num_lines = 5 ∧ 
     config'.no_three_point_intersection = true ∧ 
     config'.num_intersections = 9) := by
  sorry

end NUMINAMATH_CALUDE_five_line_intersections_l1138_113839


namespace NUMINAMATH_CALUDE_lost_card_number_l1138_113874

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ≤ n) : 
  (n * (n + 1)) / 2 - 101 = 4 := by
  sorry

#check lost_card_number

end NUMINAMATH_CALUDE_lost_card_number_l1138_113874


namespace NUMINAMATH_CALUDE_class_average_problem_l1138_113876

theorem class_average_problem (total_students : Nat) (high_score_students : Nat) 
  (zero_score_students : Nat) (high_score : Nat) (class_average : Rat) :
  total_students = 25 →
  high_score_students = 3 →
  zero_score_students = 3 →
  high_score = 95 →
  class_average = 45.6 →
  let remaining_students := total_students - high_score_students - zero_score_students
  let total_score := (total_students : Rat) * class_average
  let high_score_total := (high_score_students : Rat) * high_score
  let remaining_average := (total_score - high_score_total) / remaining_students
  remaining_average = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l1138_113876


namespace NUMINAMATH_CALUDE_only_D_is_certain_l1138_113878

structure Event where
  name : String
  is_certain : Bool

def A : Event := { name := "Moonlight in front of the bed", is_certain := false }
def B : Event := { name := "Lonely smoke in the desert", is_certain := false }
def C : Event := { name := "Reach for the stars with your hand", is_certain := false }
def D : Event := { name := "Yellow River flows into the sea", is_certain := true }

def events : List Event := [A, B, C, D]

theorem only_D_is_certain : ∃! e : Event, e ∈ events ∧ e.is_certain := by
  sorry

end NUMINAMATH_CALUDE_only_D_is_certain_l1138_113878


namespace NUMINAMATH_CALUDE_polynomial_value_at_root_l1138_113893

theorem polynomial_value_at_root (p : ℝ) : 
  p^3 - 5*p + 1 = 0 → p^4 - 3*p^3 - 5*p^2 + 16*p + 2015 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_root_l1138_113893


namespace NUMINAMATH_CALUDE_parallelogram_bisecting_line_slope_l1138_113877

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Determines if a line through the origin cuts a parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (m : ℝ) : Prop := sorry

/-- The specific parallelogram from the problem -/
def problem_parallelogram : Parallelogram :=
  { v1 := { x := 2, y := 5 }
  , v2 := { x := 2, y := 23 }
  , v3 := { x := 7, y := 38 }
  , v4 := { x := 7, y := 20 }
  }

theorem parallelogram_bisecting_line_slope :
  cuts_into_congruent_polygons problem_parallelogram (43/9) := by sorry

end NUMINAMATH_CALUDE_parallelogram_bisecting_line_slope_l1138_113877


namespace NUMINAMATH_CALUDE_uniform_color_subgrid_l1138_113888

/-- A color type with two possible values -/
inductive Color
| Red
| Blue

/-- A point in the grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A function that assigns a color to each point in the grid -/
def ColoringFunction := GridPoint → Color

/-- A theorem stating that in any two-color infinite grid, there exist two horizontal
    and two vertical lines forming a subgrid with uniformly colored intersection points -/
theorem uniform_color_subgrid
  (coloring : ColoringFunction) :
  ∃ (x₁ x₂ y₁ y₂ : ℤ) (c : Color),
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    coloring ⟨x₁, y₁⟩ = c ∧
    coloring ⟨x₁, y₂⟩ = c ∧
    coloring ⟨x₂, y₁⟩ = c ∧
    coloring ⟨x₂, y₂⟩ = c :=
by sorry


end NUMINAMATH_CALUDE_uniform_color_subgrid_l1138_113888


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l1138_113891

theorem complex_pure_imaginary (m : ℝ) : 
  (m^2 - 4 + (m + 2)*Complex.I = 0) → m = 2 :=
sorry


end NUMINAMATH_CALUDE_complex_pure_imaginary_l1138_113891


namespace NUMINAMATH_CALUDE_tim_necklace_profit_l1138_113820

/-- Represents the properties of a necklace type -/
structure NecklaceType where
  charms : ℕ
  charmCost : ℕ
  sellingPrice : ℕ

/-- Calculates the profit for a single necklace -/
def profit (n : NecklaceType) : ℕ :=
  n.sellingPrice - n.charms * n.charmCost

/-- Represents the sales information -/
structure Sales where
  typeA : NecklaceType
  typeB : NecklaceType
  soldA : ℕ
  soldB : ℕ

/-- Calculates the total profit from all sales -/
def totalProfit (s : Sales) : ℕ :=
  s.soldA * profit s.typeA + s.soldB * profit s.typeB

/-- Tim's necklace business theorem -/
theorem tim_necklace_profit :
  let s : Sales := {
    typeA := { charms := 8, charmCost := 10, sellingPrice := 125 },
    typeB := { charms := 12, charmCost := 18, sellingPrice := 280 },
    soldA := 45,
    soldB := 35
  }
  totalProfit s = 4265 := by sorry

end NUMINAMATH_CALUDE_tim_necklace_profit_l1138_113820


namespace NUMINAMATH_CALUDE_intersection_constraint_l1138_113871

theorem intersection_constraint (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  (A ∩ B = {3}) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_constraint_l1138_113871


namespace NUMINAMATH_CALUDE_log_product_equals_two_thirds_l1138_113886

theorem log_product_equals_two_thirds : 
  Real.log 2 / Real.log 3 * Real.log 9 / Real.log 8 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_two_thirds_l1138_113886


namespace NUMINAMATH_CALUDE_marble_problem_solution_l1138_113816

/-- Represents the number of marbles of each color in a box -/
structure MarbleBox where
  red : ℕ
  green : ℕ
  yellow : ℕ
  other : ℕ

/-- Calculates the total number of marbles in the box -/
def MarbleBox.total (box : MarbleBox) : ℕ :=
  box.red + box.green + box.yellow + box.other

/-- Represents the conditions of the marble problem -/
def marble_problem (box : MarbleBox) : Prop :=
  box.red = 20 ∧
  box.green = 3 * box.red ∧
  box.yellow = box.green / 5 ∧
  box.total = 4 * box.green

theorem marble_problem_solution (box : MarbleBox) :
  marble_problem box → box.other = 148 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_solution_l1138_113816


namespace NUMINAMATH_CALUDE_find_b_value_l1138_113843

theorem find_b_value (b : ℚ) : 
  ((-8 : ℚ)^2 + b * (-8 : ℚ) - 15 = 0) → b = 49/8 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l1138_113843


namespace NUMINAMATH_CALUDE_sector_perimeter_l1138_113811

/-- Given a circular sector with area 2 cm² and central angle 4 radians, its perimeter is 6 cm. -/
theorem sector_perimeter (r : ℝ) (θ : ℝ) : 
  (1/2 * r^2 * θ = 2) → θ = 4 → (r * θ + 2 * r = 6) := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l1138_113811


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1138_113806

def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 12

theorem longest_segment_in_cylinder :
  let diameter := 2 * cylinder_radius
  let longest_segment := Real.sqrt (cylinder_height ^ 2 + diameter ^ 2)
  longest_segment = Real.sqrt 244 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1138_113806


namespace NUMINAMATH_CALUDE_number_times_three_equals_33_l1138_113889

theorem number_times_three_equals_33 : ∃ x : ℝ, 3 * x = 33 ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_number_times_three_equals_33_l1138_113889


namespace NUMINAMATH_CALUDE_aunt_angela_nieces_l1138_113894

theorem aunt_angela_nieces (total_jellybeans : ℕ) (num_nephews : ℕ) (jellybeans_per_child : ℕ) 
  (h1 : total_jellybeans = 70)
  (h2 : num_nephews = 3)
  (h3 : jellybeans_per_child = 14) :
  total_jellybeans / jellybeans_per_child - num_nephews = 2 :=
by sorry

end NUMINAMATH_CALUDE_aunt_angela_nieces_l1138_113894


namespace NUMINAMATH_CALUDE_x_squared_y_not_less_than_x_cubed_plus_y_fifth_l1138_113860

theorem x_squared_y_not_less_than_x_cubed_plus_y_fifth 
  (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) : 
  x^2 * y ≥ x^3 + y^5 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_y_not_less_than_x_cubed_plus_y_fifth_l1138_113860


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1138_113854

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    a(n+1) = r * a(n) for all n -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  IsGeometric a →
  (3 * (a 3)^2 - 11 * (a 3) + 9 = 0) →
  (3 * (a 7)^2 - 11 * (a 7) + 9 = 0) →
  a 5 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1138_113854


namespace NUMINAMATH_CALUDE_terminal_side_angles_l1138_113847

def angle_set (k : ℤ) : ℝ := k * 360 - 1560

theorem terminal_side_angles :
  (∃ k : ℤ, angle_set k = 240) ∧
  (∃ k : ℤ, angle_set k = -120) ∧
  (∀ α : ℝ, (∃ k : ℤ, angle_set k = α) → α ≥ 240 ∨ α ≤ -120) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_angles_l1138_113847


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1138_113805

theorem unique_solution_for_equation (m p q : ℕ) : 
  m > 0 ∧ 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  2^m * p^2 + 1 = q^5 → 
  m = 1 ∧ p = 11 ∧ q = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1138_113805


namespace NUMINAMATH_CALUDE_greatest_whole_number_inequality_l1138_113890

theorem greatest_whole_number_inequality :
  ∀ x : ℤ, (7 * x - 8 < 4 - 2 * x) → x ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_whole_number_inequality_l1138_113890


namespace NUMINAMATH_CALUDE_tangent_length_to_circle_l1138_113849

/-- The length of the tangent segment from the origin to a circle passing through three given points -/
theorem tangent_length_to_circle (A B C : ℝ × ℝ) : 
  A = (2, 3) → B = (4, 6) → C = (3, 9) → 
  ∃ (circle : Set (ℝ × ℝ)), 
    (A ∈ circle ∧ B ∈ circle ∧ C ∈ circle) ∧
    (∃ (T : ℝ × ℝ), T ∈ circle ∧ 
      (∀ (P : ℝ × ℝ), P ∈ circle → dist (0, 0) P ≥ dist (0, 0) T) ∧
      dist (0, 0) T = Real.sqrt (10 + 3 * Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_to_circle_l1138_113849


namespace NUMINAMATH_CALUDE_m_range_characterization_l1138_113866

def r (m : ℝ) (x : ℝ) : Prop := Real.sin x + Real.cos x > m

def s (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 > 0

theorem m_range_characterization (m : ℝ) :
  (∀ x : ℝ, (r m x ∧ ¬(s m x)) ∨ (¬(r m x) ∧ s m x)) ↔ 
  (m ≤ -2 ∨ (-Real.sqrt 2 ≤ m ∧ m < 2)) :=
sorry

end NUMINAMATH_CALUDE_m_range_characterization_l1138_113866


namespace NUMINAMATH_CALUDE_S_is_line_l1138_113819

-- Define the complex number (2+5i)
def a : ℂ := 2 + 5 * Complex.I

-- Define the set S
def S : Set ℂ := {z : ℂ | ∃ (r : ℝ), a * z = r}

-- Theorem stating that S is a line
theorem S_is_line : ∃ (m b : ℝ), S = {z : ℂ | z.im = m * z.re + b} :=
sorry

end NUMINAMATH_CALUDE_S_is_line_l1138_113819


namespace NUMINAMATH_CALUDE_divide_simplify_expand_and_evaluate_l1138_113830

-- Part 1
theorem divide_simplify (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  28 * x^4 * y^2 / (7 * x^3 * y) = 4 * x * y :=
sorry

-- Part 2
theorem expand_and_evaluate :
  ∀ x y : ℝ, (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 12*x*y + 10*y^2 ∧
  (let x : ℝ := 1/3; let y : ℝ := 1/2; (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 4.5) :=
sorry

end NUMINAMATH_CALUDE_divide_simplify_expand_and_evaluate_l1138_113830


namespace NUMINAMATH_CALUDE_tin_in_new_alloy_tin_amount_is_correct_l1138_113855

/-- The amount of tin in a new alloy formed by mixing two alloys -/
theorem tin_in_new_alloy (alloy_a_mass : ℝ) (alloy_b_mass : ℝ) 
  (lead_tin_ratio_a : ℝ × ℝ) (tin_copper_ratio_b : ℝ × ℝ) : ℝ :=
  let tin_in_a := (lead_tin_ratio_a.2 / (lead_tin_ratio_a.1 + lead_tin_ratio_a.2)) * alloy_a_mass
  let tin_in_b := (tin_copper_ratio_b.1 / (tin_copper_ratio_b.1 + tin_copper_ratio_b.2)) * alloy_b_mass
  tin_in_a + tin_in_b

/-- The amount of tin in the new alloy is 139.5 kg -/
theorem tin_amount_is_correct : 
  tin_in_new_alloy 120 180 (2, 3) (3, 5) = 139.5 := by
  sorry

end NUMINAMATH_CALUDE_tin_in_new_alloy_tin_amount_is_correct_l1138_113855


namespace NUMINAMATH_CALUDE_percentage_difference_l1138_113861

theorem percentage_difference (n : ℝ) (h : n = 140) : (4/5 * n) - (65/100 * n) = 21 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1138_113861


namespace NUMINAMATH_CALUDE_nonreal_roots_product_l1138_113802

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 6*x^3 + 15*x^2 - 20*x = 4040) → 
  (∃ a b : ℂ, a ≠ b ∧ a.im ≠ 0 ∧ b.im ≠ 0 ∧ 
   (x = a ∨ x = b) ∧ 
   (x^4 - 6*x^3 + 15*x^2 - 20*x = 4040) ∧ 
   (a * b = 4 + Real.sqrt 4056)) := by
sorry

end NUMINAMATH_CALUDE_nonreal_roots_product_l1138_113802


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1138_113837

theorem absolute_value_simplification (a b : ℚ) (ha : a < 0) (hb : b > 0) :
  |a - b| + b = -a + 2*b := by sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1138_113837


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_expression_l1138_113858

theorem largest_prime_divisor_of_expression : 
  ∃ p : ℕ, 
    Prime p ∧ 
    p ∣ (Nat.factorial 12 + Nat.factorial 13 + 17) ∧
    ∀ q : ℕ, Prime q → q ∣ (Nat.factorial 12 + Nat.factorial 13 + 17) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_expression_l1138_113858


namespace NUMINAMATH_CALUDE_mary_snake_count_l1138_113824

/-- The number of breeding balls -/
def num_breeding_balls : ℕ := 3

/-- The number of snakes in each breeding ball -/
def snakes_per_ball : ℕ := 8

/-- The number of additional pairs of snakes -/
def num_snake_pairs : ℕ := 6

/-- The total number of snakes Mary saw -/
def total_snakes : ℕ := num_breeding_balls * snakes_per_ball + 2 * num_snake_pairs

theorem mary_snake_count : total_snakes = 36 := by
  sorry

end NUMINAMATH_CALUDE_mary_snake_count_l1138_113824


namespace NUMINAMATH_CALUDE_f_at_one_plus_sqrt_two_l1138_113826

-- Define the function f
def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

-- State the theorem
theorem f_at_one_plus_sqrt_two : f (1 + Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_at_one_plus_sqrt_two_l1138_113826


namespace NUMINAMATH_CALUDE_median_in_60_64_interval_l1138_113896

/-- Represents the score intervals in the histogram --/
inductive ScoreInterval
| I50_54
| I55_59
| I60_64
| I65_69
| I70_74

/-- The frequency of scores in each interval --/
def frequency : ScoreInterval → Nat
| ScoreInterval.I50_54 => 3
| ScoreInterval.I55_59 => 5
| ScoreInterval.I60_64 => 10
| ScoreInterval.I65_69 => 15
| ScoreInterval.I70_74 => 20

/-- The total number of students --/
def totalStudents : Nat := 100

/-- The position of the median in the ordered list of scores --/
def medianPosition : Nat := (totalStudents + 1) / 2

/-- Theorem stating that the median score is in the interval 60-64 --/
theorem median_in_60_64_interval :
  ∃ k : Nat, k ≤ medianPosition ∧
  (frequency ScoreInterval.I70_74 + frequency ScoreInterval.I65_69 + frequency ScoreInterval.I60_64) ≥ k ∧
  (frequency ScoreInterval.I70_74 + frequency ScoreInterval.I65_69) < k :=
by sorry

end NUMINAMATH_CALUDE_median_in_60_64_interval_l1138_113896


namespace NUMINAMATH_CALUDE_other_rectangle_perimeter_l1138_113851

/-- Represents the perimeter of a rectangle --/
def rectangle_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Represents the side length of the original square --/
def square_side : ℝ := 5

/-- Represents the perimeter of the first rectangle --/
def first_rectangle_perimeter : ℝ := 16

theorem other_rectangle_perimeter :
  ∀ (l w : ℝ),
  l + w = square_side →
  rectangle_perimeter l w = first_rectangle_perimeter →
  rectangle_perimeter square_side (square_side - w) = 14 :=
by sorry

end NUMINAMATH_CALUDE_other_rectangle_perimeter_l1138_113851


namespace NUMINAMATH_CALUDE_younger_brother_silver_fraction_l1138_113813

/-- The fraction of total silver received by the younger brother in a treasure division problem -/
theorem younger_brother_silver_fraction (x y : ℝ) 
  (h1 : x / 5 + y / 7 = 100)  -- Elder brother's share
  (h2 : x / 7 + (700 - x) / 7 = 100)  -- Younger brother's share
  : (700 - x) / (7 * y) = (y - (y - x / 5) / 2) / y := by
  sorry

end NUMINAMATH_CALUDE_younger_brother_silver_fraction_l1138_113813


namespace NUMINAMATH_CALUDE_subtraction_decimal_l1138_113836

theorem subtraction_decimal : 3.56 - 1.29 = 2.27 := by sorry

end NUMINAMATH_CALUDE_subtraction_decimal_l1138_113836


namespace NUMINAMATH_CALUDE_photographer_theorem_l1138_113844

/-- Represents the number of birds of each species -/
structure BirdCount where
  starlings : Nat
  wagtails : Nat
  woodpeckers : Nat

/-- The initial bird count -/
def initial_birds : BirdCount :=
  { starlings := 8, wagtails := 7, woodpeckers := 5 }

/-- The total number of birds -/
def total_birds : Nat := 20

/-- The number of photos to be taken -/
def photos_taken : Nat := 7

/-- Predicate to check if the remaining birds meet the condition -/
def meets_condition (b : BirdCount) : Prop :=
  (b.starlings ≥ 4 ∧ (b.wagtails ≥ 3 ∨ b.woodpeckers ≥ 3)) ∨
  (b.wagtails ≥ 4 ∧ (b.starlings ≥ 3 ∨ b.woodpeckers ≥ 3)) ∨
  (b.woodpeckers ≥ 4 ∧ (b.starlings ≥ 3 ∨ b.wagtails ≥ 3))

theorem photographer_theorem :
  ∀ (remaining : BirdCount),
    remaining.starlings + remaining.wagtails + remaining.woodpeckers = total_birds - photos_taken →
    remaining.starlings ≤ initial_birds.starlings →
    remaining.wagtails ≤ initial_birds.wagtails →
    remaining.woodpeckers ≤ initial_birds.woodpeckers →
    meets_condition remaining :=
by
  sorry

end NUMINAMATH_CALUDE_photographer_theorem_l1138_113844


namespace NUMINAMATH_CALUDE_divisibility_condition_l1138_113818

theorem divisibility_condition (a b : ℕ+) : 
  (∃ k : ℕ, (b.val ^ 2 + 3 * a.val) = a.val ^ 2 * b.val * k) ↔ 
  ((a, b) = (1, 1) ∨ (a, b) = (1, 3)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1138_113818


namespace NUMINAMATH_CALUDE_inequality_theorem_l1138_113815

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x

noncomputable def g (x : ℝ) : ℝ := x / (Real.exp x)

theorem inequality_theorem (k : ℝ) :
  (∀ x x₂ : ℝ, x > 0 → x₂ > 0 → g x / k ≤ f x₂ / (k + 1)) →
  k ≥ 1 / (2 * Real.exp 1 - 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1138_113815


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l1138_113810

theorem min_sum_absolute_values :
  (∀ x : ℝ, |x + 3| + |x + 4| + |x + 6| ≥ 3) ∧
  (∃ x : ℝ, |x + 3| + |x + 4| + |x + 6| = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l1138_113810


namespace NUMINAMATH_CALUDE_tangent_line_at_2_l1138_113808

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

-- Theorem statement
theorem tangent_line_at_2 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_l1138_113808


namespace NUMINAMATH_CALUDE_ma_xiaohu_speed_ma_xiaohu_speed_proof_l1138_113823

/-- Proves that Ma Xiaohu's speed is 80 meters per minute given the problem conditions -/
theorem ma_xiaohu_speed : ℝ → Prop :=
  fun (x : ℝ) ↦
    let total_distance : ℝ := 1800
    let catch_up_distance : ℝ := 200
    let father_delay : ℝ := 10
    let father_speed : ℝ := 2 * x
    let ma_distance : ℝ := total_distance - catch_up_distance
    let ma_time : ℝ := ma_distance / x
    let father_time : ℝ := ma_distance / father_speed
    ma_time - father_time = father_delay → x = 80

/-- Proof of the theorem -/
theorem ma_xiaohu_speed_proof : ma_xiaohu_speed 80 := by
  sorry

end NUMINAMATH_CALUDE_ma_xiaohu_speed_ma_xiaohu_speed_proof_l1138_113823


namespace NUMINAMATH_CALUDE_parallel_lines_exist_points_not_on_line_l1138_113803

-- Define the line equation
def line_equation (α x y : ℝ) : Prop :=
  Real.cos α * (x - 2) + Real.sin α * (y + 1) = 1

-- Statement ②: There exist different real numbers α₁, α₂, such that the corresponding lines l₁, l₂ are parallel
theorem parallel_lines_exist : ∃ α₁ α₂ : ℝ, α₁ ≠ α₂ ∧
  ∀ x y : ℝ, line_equation α₁ x y ↔ line_equation α₂ x y :=
sorry

-- Statement ③: There are at least two points in the coordinate plane that are not on the line l
theorem points_not_on_line : ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
  (∀ α : ℝ, ¬line_equation α x₁ y₁ ∧ ¬line_equation α x₂ y₂) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_exist_points_not_on_line_l1138_113803


namespace NUMINAMATH_CALUDE_factor_polynomial_l1138_113835

theorem factor_polynomial (x : ℝ) :
  x^4 - 36*x^2 + 25 = (x^2 - 6*x + 5) * (x^2 + 6*x + 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1138_113835


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1138_113859

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x) / (3 * x - 1) > 1 ↔ 1 / 3 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1138_113859


namespace NUMINAMATH_CALUDE_sequence_properties_arithmetic_sequence_l1138_113873

def a_n (n a : ℕ+) : ℚ := n / (n + a)

theorem sequence_properties (a : ℕ+) :
  (∃ r : ℚ, a_n 1 a * r = a_n 3 a ∧ a_n 3 a * r = a_n 15 a) →
  a = 9 :=
sorry

theorem arithmetic_sequence (a k : ℕ+) :
  k ≥ 3 →
  (a_n 1 a + a_n k a = 2 * a_n 2 a) →
  ((a = 1 ∧ k = 5) ∨ (a = 2 ∧ k = 4)) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_arithmetic_sequence_l1138_113873


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l1138_113801

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  totalVolume : ℝ
  orangePercent : ℝ
  watermelonPercent : ℝ
  grapeVolume : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_volume (drink : FruitDrink) 
  (h1 : drink.orangePercent = 0.25)
  (h2 : drink.watermelonPercent = 0.4)
  (h3 : drink.grapeVolume = 105)
  (h4 : drink.orangePercent + drink.watermelonPercent + drink.grapeVolume / drink.totalVolume = 1) :
  drink.totalVolume = 300 := by
  sorry

#check fruit_drink_volume

end NUMINAMATH_CALUDE_fruit_drink_volume_l1138_113801


namespace NUMINAMATH_CALUDE_min_positive_period_of_f_l1138_113868

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x - Real.sqrt 3 * cos x) * (cos x - Real.sqrt 3 * sin x)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π :=
sorry

end NUMINAMATH_CALUDE_min_positive_period_of_f_l1138_113868


namespace NUMINAMATH_CALUDE_common_tangent_line_l1138_113857

/-- Two circles O₁ and O₂ in the Cartesian coordinate system -/
structure TwoCircles where
  m : ℝ
  r₁ : ℝ
  r₂ : ℝ
  h₁ : m > 0
  h₂ : r₁ > 0
  h₃ : r₂ > 0
  h₄ : r₁ * r₂ = 2
  h₅ : (3 : ℝ) = r₁ / m
  h₆ : (1 : ℝ) = r₁
  h₇ : (2 : ℝ) ^ 2 + (2 : ℝ) ^ 2 = (2 - r₁ / m) ^ 2 + (2 - r₁) ^ 2 + r₁ ^ 2
  h₈ : (2 : ℝ) ^ 2 + (2 : ℝ) ^ 2 = (2 - r₂ / m) ^ 2 + (2 - r₂) ^ 2 + r₂ ^ 2

/-- The equation of another common tangent line is y = (4/3)x -/
theorem common_tangent_line (c : TwoCircles) :
  ∃ (k : ℝ), k = 4 / 3 ∧ ∀ (x y : ℝ), y = k * x → 
  (∃ (t : ℝ), (x - 3) ^ 2 + (y - 1) ^ 2 = t ^ 2 ∧ (x - c.r₂ / c.m) ^ 2 + (y - c.r₂) ^ 2 = t ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_common_tangent_line_l1138_113857


namespace NUMINAMATH_CALUDE_triangle_count_l1138_113846

/-- The total number of triangles in a specially divided rectangle -/
def total_triangles (small_right : ℕ) (isosceles_quarter_width : ℕ) (isosceles_third_length : ℕ) (larger_right : ℕ) (large_isosceles : ℕ) : ℕ :=
  small_right + isosceles_quarter_width + isosceles_third_length + larger_right + large_isosceles

/-- Theorem stating the total number of triangles in the specially divided rectangle -/
theorem triangle_count :
  total_triangles 24 8 12 16 4 = 64 := by sorry

end NUMINAMATH_CALUDE_triangle_count_l1138_113846


namespace NUMINAMATH_CALUDE_expression_evaluation_l1138_113832

theorem expression_evaluation :
  let expr1 := (27 / 8) ^ (-2/3) - (49 / 9) ^ (1/2) + (0.2)^(-2) * (3 / 25)
  let expr2 := -5 * (Real.log 4 / Real.log 9) + (Real.log (32 / 9) / Real.log 3) - 5^(Real.log 3 / Real.log 5)
  (expr1 = 10/9) ∧ (expr2 = -5 * (Real.log 2 / Real.log 3) - 5) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1138_113832


namespace NUMINAMATH_CALUDE_circle_radius_d_value_l1138_113880

theorem circle_radius_d_value (x y : ℝ) (d : ℝ) : 
  (∀ x y, x^2 - 8*x + y^2 + 10*y + d = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25) → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_d_value_l1138_113880


namespace NUMINAMATH_CALUDE_quadratic_function_value_l1138_113814

/-- A quadratic function f(x) = ax^2 + bx + 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- Theorem: If f(1) = 3 and f(2) = 6, then f(3) = 10 -/
theorem quadratic_function_value (a b : ℝ) :
  f a b 1 = 3 → f a b 2 = 6 → f a b 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l1138_113814
