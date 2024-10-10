import Mathlib

namespace arctan_sum_equals_pi_over_four_l887_88737

theorem arctan_sum_equals_pi_over_four :
  ∃ (n : ℕ), n > 0 ∧ Real.arctan (1/3) + Real.arctan (1/5) + Real.arctan (1/7) + Real.arctan (1/n : ℝ) = π/4 ∧ n = 8 := by
  sorry

end arctan_sum_equals_pi_over_four_l887_88737


namespace tan_ratio_inequality_l887_88758

theorem tan_ratio_inequality (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) :
  (Real.tan α) / α < (Real.tan β) / β := by
  sorry

end tan_ratio_inequality_l887_88758


namespace annual_yield_improvement_l887_88702

/-- The percentage improvement in annual yield given last year's and this year's ranges -/
theorem annual_yield_improvement (last_year_range this_year_range : ℝ) 
  (h1 : last_year_range = 10000)
  (h2 : this_year_range = 11500) :
  (this_year_range - last_year_range) / last_year_range * 100 = 15 := by
  sorry

end annual_yield_improvement_l887_88702


namespace olivia_initial_wallet_l887_88718

/-- The amount of money Olivia spent at the supermarket -/
def amount_spent : ℕ := 15

/-- The amount of money Olivia has left after spending -/
def amount_left : ℕ := 63

/-- The initial amount of money in Olivia's wallet -/
def initial_amount : ℕ := amount_spent + amount_left

theorem olivia_initial_wallet : initial_amount = 78 := by
  sorry

end olivia_initial_wallet_l887_88718


namespace problem_solving_probability_l887_88727

theorem problem_solving_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/3) (h3 : p3 = 1/4) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/5 := by
  sorry

end problem_solving_probability_l887_88727


namespace measure_all_masses_l887_88779

-- Define the set of weights
def weights : List ℕ := [1, 3, 9, 27, 81]

-- Define a function to check if a mass can be measured
def can_measure (mass : ℕ) : Prop :=
  ∃ (a b c d e : ℤ), 
    a * 1 + b * 3 + c * 9 + d * 27 + e * 81 = mass ∧ 
    (a ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (b ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (c ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (d ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (e ∈ ({-1, 0, 1} : Set ℤ))

-- Theorem statement
theorem measure_all_masses : 
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 121 → can_measure m :=
by sorry

end measure_all_masses_l887_88779


namespace quadratic_inequality_range_l887_88709

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
sorry

end quadratic_inequality_range_l887_88709


namespace final_population_theorem_l887_88753

/-- Calculates the population after two years of change -/
def population_after_two_years (initial_population : ℕ) : ℕ :=
  let after_increase := initial_population * 130 / 100
  let after_decrease := after_increase * 70 / 100
  after_decrease

/-- Theorem stating the final population after two years of change -/
theorem final_population_theorem :
  population_after_two_years 15000 = 13650 := by
  sorry

end final_population_theorem_l887_88753


namespace x_varies_as_three_fifths_of_z_l887_88722

-- Define the relationships between x, y, and z
def varies_as_cube (x y : ℝ) : Prop := ∃ k : ℝ, x = k * y^3

def varies_as_fifth_root (y z : ℝ) : Prop := ∃ j : ℝ, y = j * z^(1/5)

def varies_as_power (x z : ℝ) (n : ℝ) : Prop := ∃ m : ℝ, x = m * z^n

-- State the theorem
theorem x_varies_as_three_fifths_of_z (x y z : ℝ) :
  varies_as_cube x y → varies_as_fifth_root y z → varies_as_power x z (3/5) :=
by sorry

end x_varies_as_three_fifths_of_z_l887_88722


namespace number_increased_by_twenty_percent_l887_88760

theorem number_increased_by_twenty_percent (x : ℝ) : x * 1.2 = 1080 ↔ x = 900 := by sorry

end number_increased_by_twenty_percent_l887_88760


namespace volleyball_lineup_count_l887_88780

def choose (n k : ℕ) : ℕ := Nat.choose n k

def volleyball_lineups (total_players triplets : ℕ) (max_triplets : ℕ) : ℕ :=
  let non_triplets := total_players - triplets - 1  -- Subtract 1 for the captain
  let case0 := choose non_triplets 5
  let case1 := triplets * choose non_triplets 4
  let case2 := choose triplets 2 * choose non_triplets 3
  case0 + case1 + case2

theorem volleyball_lineup_count :
  volleyball_lineups 15 4 2 = 1812 := by
  sorry

end volleyball_lineup_count_l887_88780


namespace fraction_product_cubes_evaluate_fraction_product_l887_88755

theorem fraction_product_cubes (a b c d : ℚ) :
  (a / b) ^ 3 * (c / d) ^ 3 = (a * c / (b * d)) ^ 3 :=
by sorry

theorem evaluate_fraction_product :
  (8 / 9 : ℚ) ^ 3 * (3 / 4 : ℚ) ^ 3 = 8 / 27 :=
by sorry

end fraction_product_cubes_evaluate_fraction_product_l887_88755


namespace handshake_theorem_l887_88735

theorem handshake_theorem (n : ℕ) (h : n = 8) :
  let total_people := n
  let num_teams := n / 2
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2 = 24 :=
by sorry

end handshake_theorem_l887_88735


namespace rectangle_triangle_area_ratio_l887_88752

theorem rectangle_triangle_area_ratio : 
  ∀ (L W : ℝ), L > 0 → W > 0 →
  (L * W) / ((1/2) * L * W) = 2 := by
sorry

end rectangle_triangle_area_ratio_l887_88752


namespace tangent_roots_sum_identity_l887_88765

theorem tangent_roots_sum_identity (p q : ℝ) (α β : ℝ) :
  (Real.tan α + Real.tan β = -p) →
  (Real.tan α * Real.tan β = q) →
  Real.sin (α + β)^2 + p * Real.sin (α + β) * Real.cos (α + β) + q * Real.cos (α + β)^2 = q := by
  sorry

end tangent_roots_sum_identity_l887_88765


namespace wage_increase_with_productivity_l887_88785

/-- Represents the linear regression equation for workers' wages as a function of labor productivity -/
def wage_equation (x : ℝ) : ℝ := 50 + 80 * x

/-- Theorem stating that an increase of 1 in labor productivity leads to an increase of 80 in wages -/
theorem wage_increase_with_productivity (x : ℝ) :
  wage_equation (x + 1) - wage_equation x = 80 := by
  sorry

end wage_increase_with_productivity_l887_88785


namespace sin_cos_identity_indeterminate_l887_88773

theorem sin_cos_identity_indeterminate (α : Real) : 
  α ∈ Set.Ioo 0 Real.pi → 
  (Real.sin α)^2 + Real.cos (2 * α) = 1 → 
  ∀ β ∈ Set.Ioo 0 Real.pi, (Real.sin β)^2 + Real.cos (2 * β) = 1 ∧ 
  ¬∃!t, t = Real.tan α := by
  sorry

end sin_cos_identity_indeterminate_l887_88773


namespace increase_by_percentage_l887_88746

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 1500 →
  percentage = 20 →
  final = initial * (1 + percentage / 100) →
  final = 1800 := by
  sorry

end increase_by_percentage_l887_88746


namespace roots_of_quadratic_equation_l887_88768

theorem roots_of_quadratic_equation : 
  ∀ x : ℝ, x^2 = 2*x ↔ x = 0 ∨ x = 2 := by sorry

end roots_of_quadratic_equation_l887_88768


namespace puzzle_sum_l887_88739

theorem puzzle_sum (A B C D : Nat) : 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * 1000 + B - (5000 + C * 10 + 9) = 1000 + D * 100 + 90 + 3 →
  A + B + C + D = 18 := by
  sorry

end puzzle_sum_l887_88739


namespace function_proof_l887_88778

theorem function_proof (f : ℕ → ℕ) 
  (h1 : f 0 = 1)
  (h2 : f 2016 = 2017)
  (h3 : ∀ n, f (f n) + f n = 2 * n + 3) :
  ∀ n, f n = n + 1 := by
sorry

end function_proof_l887_88778


namespace sin_2alpha_value_l887_88742

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end sin_2alpha_value_l887_88742


namespace geometric_sequence_sum_constant_l887_88792

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    is S_n = 3^(n-2) + m, prove that m = -1/9 -/
theorem geometric_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℝ) :
  (∀ n, S n = 3^(n-2) + m) →
  (∀ n, a (n+1) / a n = a (n+2) / a (n+1)) →
  (a 1 = S 1) →
  (∀ n, a (n+1) = S (n+1) - S n) →
  m = -1/9 := by
sorry

end geometric_sequence_sum_constant_l887_88792


namespace cello_viola_pairs_count_l887_88743

/-- The number of cello-viola pairs in a music store, where each pair consists of
    a cello and a viola made from the same tree. -/
def cello_viola_pairs : ℕ := 70

theorem cello_viola_pairs_count (total_cellos : ℕ) (total_violas : ℕ) 
  (prob_same_tree : ℚ) (h1 : total_cellos = 800) (h2 : total_violas = 600) 
  (h3 : prob_same_tree = 14583333333333335 / 100000000000000000) : 
  cello_viola_pairs = (prob_same_tree * total_cellos * total_violas : ℚ).num := by
  sorry

end cello_viola_pairs_count_l887_88743


namespace least_positive_integer_divisibility_l887_88736

theorem least_positive_integer_divisibility : ∃ d : ℕ+, 
  (∀ k : ℕ+, k < d → ¬(13 ∣ (k^3 + 1000))) ∧ (13 ∣ (d^3 + 1000)) ∧ d = 1 := by
  sorry

end least_positive_integer_divisibility_l887_88736


namespace median_of_class_distribution_l887_88723

/-- Represents the distribution of weekly reading times for students -/
structure ReadingTimeDistribution where
  six_hours : Nat
  seven_hours : Nat
  eight_hours : Nat
  nine_hours : Nat

/-- Calculates the median of a given reading time distribution -/
def median (d : ReadingTimeDistribution) : Real :=
  sorry

/-- The specific distribution of reading times for the 30 students -/
def class_distribution : ReadingTimeDistribution :=
  { six_hours := 7
  , seven_hours := 8
  , eight_hours := 5
  , nine_hours := 10 }

/-- The theorem stating that the median of the given distribution is 7.5 -/
theorem median_of_class_distribution :
  median class_distribution = 7.5 := by sorry

end median_of_class_distribution_l887_88723


namespace min_value_problem_l887_88766

open Real

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : log 2 * x + log 8 * y = log 2) : 
  (∀ a b : ℝ, a > 0 → b > 0 → log 2 * a + log 8 * b = log 2 → 1/x + 1/y ≤ 1/a + 1/b) ∧ 
  (∃ c d : ℝ, c > 0 ∧ d > 0 ∧ log 2 * c + log 8 * d = log 2 ∧ 1/c + 1/d = 4 + 2 * sqrt 3) :=
sorry

end min_value_problem_l887_88766


namespace cone_base_circumference_l887_88713

/-- The circumference of the base of a right circular cone formed from a circular piece of paper 
    with radius 6 inches, after removing a 180-degree sector, is equal to 6π inches. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) : 
  let full_circumference := 2 * π * r
  let removed_angle := π  -- 180 degrees in radians
  let remaining_angle := 2 * π - removed_angle
  let base_circumference := (remaining_angle / (2 * π)) * full_circumference
  base_circumference = 6 * π := by
sorry


end cone_base_circumference_l887_88713


namespace book_sale_loss_percentage_l887_88762

/-- Proves that the percentage of loss is 10% given the conditions of the problem -/
theorem book_sale_loss_percentage (CP : ℝ) : 
  CP > 720 ∧ 880 = 1.10 * CP → (CP - 720) / CP * 100 = 10 := by
  sorry

end book_sale_loss_percentage_l887_88762


namespace sequence_ratio_l887_88763

/-- Given an arithmetic sequence a and a geometric sequence b with specific conditions,
    prove that the ratio of their second terms is 1. -/
theorem sequence_ratio (a b : ℕ → ℚ) : 
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- a is arithmetic
  (∀ n : ℕ, b (n + 1) / b n = b 1 / b 0) →  -- b is geometric
  a 0 = -1 →                                -- a₁ = -1
  b 0 = -1 →                                -- b₁ = -1
  a 3 = 8 →                                 -- a₄ = 8
  b 3 = 8 →                                 -- b₄ = 8
  a 1 / b 1 = 1 :=                          -- a₂/b₂ = 1
by sorry

end sequence_ratio_l887_88763


namespace inequality_solution_set_l887_88757

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 < -2*x + 15

-- Define the solution set
def solution_set : Set ℝ := {x | -5 < x ∧ x < 3}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end inequality_solution_set_l887_88757


namespace parcel_cost_correct_l887_88725

/-- The cost function for sending a parcel post package -/
def parcel_cost (P : ℕ) : ℕ :=
  12 + 5 * P

/-- Theorem stating the correctness of the parcel cost function -/
theorem parcel_cost_correct (P : ℕ) (h : P ≥ 1) :
  parcel_cost P = 15 + 5 * (P - 1) + 2 :=
by sorry

end parcel_cost_correct_l887_88725


namespace hyperbola_eccentricity_l887_88786

/-- The eccentricity of a hyperbola with asymptotes tangent to a specific circle -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 ∧
   (b * x + a * y = 0 ∨ b * x - a * y = 0) ∧
   (x - Real.sqrt 2)^2 + y^2 = 1) →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l887_88786


namespace power_of_two_multiple_one_two_l887_88754

/-- A function that checks if a natural number only contains digits 1 and 2 in its decimal representation -/
def onlyOneAndTwo (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 2

/-- For every power of 2, there exists a multiple of it that only contains digits 1 and 2 -/
theorem power_of_two_multiple_one_two :
  ∀ k : ℕ, ∃ n : ℕ, 2^k ∣ n ∧ onlyOneAndTwo n := by
  sorry

end power_of_two_multiple_one_two_l887_88754


namespace trig_identity_l887_88761

theorem trig_identity (θ : Real) (h : Real.tan θ = Real.sqrt 3) :
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := by
  sorry

end trig_identity_l887_88761


namespace slinkums_shipment_correct_l887_88741

/-- The total number of Mr. Slinkums in the initial shipment -/
def total_slinkums : ℕ := 200

/-- The percentage of Mr. Slinkums on display -/
def display_percentage : ℚ := 30 / 100

/-- The number of Mr. Slinkums in storage -/
def storage_slinkums : ℕ := 140

/-- Theorem stating that the total number of Mr. Slinkums is correct given the conditions -/
theorem slinkums_shipment_correct : 
  (1 - display_percentage) * (total_slinkums : ℚ) = storage_slinkums := by
  sorry

end slinkums_shipment_correct_l887_88741


namespace new_number_correct_l887_88790

/-- Given a two-digit number with tens' digit t and units' digit u,
    the function calculates the new three-digit number formed by
    reversing the digits and placing 2 after the reversed number. -/
def new_number (t u : ℕ) : ℕ :=
  100 * u + 10 * t + 2

/-- Theorem stating that the new_number function correctly calculates
    the desired three-digit number for any two-digit number. -/
theorem new_number_correct (t u : ℕ) (h1 : t ≥ 1) (h2 : t ≤ 9) (h3 : u ≤ 9) :
  new_number t u = 100 * u + 10 * t + 2 :=
by sorry

end new_number_correct_l887_88790


namespace april_greatest_drop_l887_88705

/-- Represents the months from January to June --/
inductive Month
| January
| February
| March
| April
| May
| June

/-- Returns the price of the smartphone at the end of the given month --/
def price (m : Month) : Int :=
  match m with
  | Month.January => 350
  | Month.February => 330
  | Month.March => 370
  | Month.April => 340
  | Month.May => 320
  | Month.June => 300

/-- Calculates the price drop from one month to the next --/
def priceDrop (m : Month) : Int :=
  match m with
  | Month.January => price Month.January - price Month.February
  | Month.February => price Month.February - price Month.March
  | Month.March => price Month.March - price Month.April
  | Month.April => price Month.April - price Month.May
  | Month.May => price Month.May - price Month.June
  | Month.June => 0  -- No next month defined

/-- Theorem stating that April had the greatest monthly drop in price --/
theorem april_greatest_drop :
  ∀ m : Month, m ≠ Month.April → priceDrop Month.April ≥ priceDrop m :=
by sorry

end april_greatest_drop_l887_88705


namespace garden_tomato_percentage_l887_88794

theorem garden_tomato_percentage :
  let total_plants : ℕ := 20 + 15
  let second_garden_tomatoes : ℕ := 15 / 3
  let total_tomatoes : ℕ := (total_plants * 20) / 100
  let first_garden_tomatoes : ℕ := total_tomatoes - second_garden_tomatoes
  (first_garden_tomatoes : ℚ) / 20 * 100 = 10 := by
  sorry

end garden_tomato_percentage_l887_88794


namespace remainder_theorem_l887_88711

theorem remainder_theorem (x : ℝ) : 
  ∃ (P : ℝ → ℝ) (S : ℝ → ℝ), 
    (∀ x, x^105 = (x^2 - 4*x + 3) * P x + S x) ∧ 
    (∀ x, S x = (3^105 * (x - 1) - (x - 2)) / 2) :=
by sorry

end remainder_theorem_l887_88711


namespace roots_sum_reciprocal_minus_one_l887_88771

theorem roots_sum_reciprocal_minus_one (b c : ℝ) : 
  b^2 - b - 1 = 0 → c^2 - c - 1 = 0 → b ≠ c → 1 / (1 - b) + 1 / (1 - c) = -1 := by
  sorry

end roots_sum_reciprocal_minus_one_l887_88771


namespace sine_function_period_l887_88734

/-- Given a sinusoidal function y = 2sin(ωx + φ) with ω > 0,
    if the maximum value 2 occurs at x = π/6 and
    the minimum value -2 occurs at x = 2π/3,
    then ω = 2. -/
theorem sine_function_period (ω φ : ℝ) (h_ω_pos : ω > 0) :
  (∀ x : ℝ, 2 * Real.sin (ω * x + φ) ≤ 2) ∧
  (2 * Real.sin (ω * (π / 6) + φ) = 2) ∧
  (2 * Real.sin (ω * (2 * π / 3) + φ) = -2) →
  ω = 2 := by
  sorry

end sine_function_period_l887_88734


namespace x_squared_plus_y_squared_l887_88715

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 120) : 
  x^2 + y^2 = 11980 / 121 := by
  sorry

end x_squared_plus_y_squared_l887_88715


namespace tan_double_angle_l887_88772

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (2 * α) = -4/3 := by
  sorry

end tan_double_angle_l887_88772


namespace expression_evaluation_l887_88795

theorem expression_evaluation (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (2*x + y)^2 + (x + y)*(x - y) - x^2 = -4 := by sorry

end expression_evaluation_l887_88795


namespace probability_of_selection_for_given_sizes_l887_88738

/-- Simple random sampling without replacement -/
structure SimpleRandomSampling where
  population_size : ℕ
  sample_size : ℕ
  sample_size_le_population : sample_size ≤ population_size

/-- The probability of an individual being selected in simple random sampling -/
def probability_of_selection (srs : SimpleRandomSampling) : ℚ :=
  srs.sample_size / srs.population_size

theorem probability_of_selection_for_given_sizes :
  ∀ (srs : SimpleRandomSampling),
    srs.population_size = 6 →
    srs.sample_size = 3 →
    probability_of_selection srs = 1/2 :=
by sorry

end probability_of_selection_for_given_sizes_l887_88738


namespace max_triangle_area_l887_88726

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 6 = 0

/-- Circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

/-- Point on circle C -/
def point_on_C (P : ℝ × ℝ) : Prop := circle_C P.1 P.2

/-- Intersection points of line l and circle C -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

/-- Area of triangle PAB -/
noncomputable def triangle_area (P A B : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Maximum area of triangle PAB -/
theorem max_triangle_area :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  ∃ max_area : ℝ, max_area = (27 * Real.sqrt 3) / 4 ∧
  ∀ P : ℝ × ℝ, point_on_C P → triangle_area P A B ≤ max_area :=
sorry

end max_triangle_area_l887_88726


namespace negation_of_all_squares_positive_l887_88731

theorem negation_of_all_squares_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, n^2 ≤ 0 := by sorry

end negation_of_all_squares_positive_l887_88731


namespace quadratic_inequality_solution_set_l887_88798

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3*x - 4 ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by sorry

end quadratic_inequality_solution_set_l887_88798


namespace a_upper_bound_l887_88747

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * a (n / 2) + 3 * a (n / 3) + 6 * a (n / 6)

theorem a_upper_bound : ∀ n : ℕ, a n ≤ 10 * n^2 + 1 := by
  sorry

end a_upper_bound_l887_88747


namespace sugar_calculation_l887_88777

/-- The total amount of sugar given the number of packs, weight per pack, and leftover sugar -/
def total_sugar (num_packs : ℕ) (weight_per_pack : ℕ) (leftover : ℕ) : ℕ :=
  num_packs * weight_per_pack + leftover

/-- Theorem: Given 12 packs of sugar weighing 250 grams each and 20 grams of leftover sugar,
    the total amount of sugar is 3020 grams -/
theorem sugar_calculation :
  total_sugar 12 250 20 = 3020 := by
  sorry

end sugar_calculation_l887_88777


namespace flag_paint_cost_l887_88740

/-- Calculates the cost of paint for a flag given its dimensions and paint properties -/
theorem flag_paint_cost (width height : ℝ) (paint_cost_per_quart : ℝ) (coverage_per_quart : ℝ) : 
  width = 5 → height = 4 → paint_cost_per_quart = 2 → coverage_per_quart = 4 → 
  (2 * width * height / coverage_per_quart) * paint_cost_per_quart = 20 := by
sorry


end flag_paint_cost_l887_88740


namespace moon_temperature_difference_l887_88730

/-- The temperature difference between day and night on the moon's surface. -/
def moonTemperatureDifference (dayTemp : ℝ) (nightTemp : ℝ) : ℝ :=
  dayTemp - nightTemp

/-- Theorem stating the temperature difference on the moon's surface. -/
theorem moon_temperature_difference :
  moonTemperatureDifference 127 (-183) = 310 := by
  sorry

end moon_temperature_difference_l887_88730


namespace nth_k_gonal_number_l887_88787

/-- The nth k-gonal number -/
def N (n k : ℕ) : ℚ :=
  ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n

/-- Theorem stating the properties of the nth k-gonal number -/
theorem nth_k_gonal_number (k : ℕ) (h : k ≥ 3) :
  ∀ n : ℕ, N n k = ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n ∧
  N 10 24 = 1000 := by sorry

end nth_k_gonal_number_l887_88787


namespace total_carpets_l887_88745

theorem total_carpets (house1 house2 house3 house4 : ℕ) : 
  house1 = 12 → 
  house2 = 20 → 
  house3 = 10 → 
  house4 = 2 * house3 → 
  house1 + house2 + house3 + house4 = 62 := by
  sorry

end total_carpets_l887_88745


namespace james_total_score_l887_88719

theorem james_total_score (field_goals : ℕ) (two_point_shots : ℕ) 
  (h1 : field_goals = 13) (h2 : two_point_shots = 20) : 
  field_goals * 3 + two_point_shots * 2 = 79 := by
sorry

end james_total_score_l887_88719


namespace rectangle_area_around_square_total_area_of_rectangles_l887_88748

theorem rectangle_area_around_square (l₁ l₂ : ℝ) : 
  l₁ + l₂ = 11 → 
  2 * (6 * l₁ + 6 * l₂) = 132 := by
  sorry

theorem total_area_of_rectangles : 
  ∀ (l₁ l₂ : ℝ), 
  (4 * (12 + l₁ + l₂) = 92) → 
  (2 * (6 * l₁ + 6 * l₂) = 132) := by
  sorry

end rectangle_area_around_square_total_area_of_rectangles_l887_88748


namespace laura_to_ken_ratio_l887_88759

/-- The number of tiles Don can paint per minute -/
def D : ℕ := 3

/-- The number of tiles Ken can paint per minute -/
def K : ℕ := D + 2

/-- The number of tiles Laura can paint per minute -/
def L : ℕ := 10

/-- The number of tiles Kim can paint per minute -/
def Kim : ℕ := L - 3

/-- The total number of tiles painted by all four people in 15 minutes -/
def total_tiles : ℕ := 375

/-- The theorem stating that the ratio of Laura's painting rate to Ken's painting rate is 2:1 -/
theorem laura_to_ken_ratio :
  (L : ℚ) / K = 2 / 1 ∧ 15 * (D + K + L + Kim) = total_tiles :=
by sorry

end laura_to_ken_ratio_l887_88759


namespace water_percentage_in_mixture_l887_88770

/-- Given two liquids with different water percentages, prove the water percentage in their mixture -/
theorem water_percentage_in_mixture 
  (water_percent_1 water_percent_2 : ℝ) 
  (parts_1 parts_2 : ℝ) 
  (h1 : water_percent_1 = 20)
  (h2 : water_percent_2 = 35)
  (h3 : parts_1 = 10)
  (h4 : parts_2 = 4) :
  (water_percent_1 / 100 * parts_1 + water_percent_2 / 100 * parts_2) / (parts_1 + parts_2) * 100 =
  (0.2 * 10 + 0.35 * 4) / (10 + 4) * 100 := by
  sorry

#eval (0.2 * 10 + 0.35 * 4) / (10 + 4) * 100

end water_percentage_in_mixture_l887_88770


namespace unique_positive_integer_solution_l887_88789

theorem unique_positive_integer_solution (m : ℤ) : 
  (∃! x : ℤ, x > 0 ∧ 6 * x^2 + 2 * (m - 13) * x + 12 - m = 0) ↔ m = 8 := by
  sorry

end unique_positive_integer_solution_l887_88789


namespace city_population_ratio_l887_88716

/-- The population of Lake View -/
def lake_view_population : ℕ := 24000

/-- The difference between Lake View and Seattle populations -/
def population_difference : ℕ := 4000

/-- The total population of the three cities -/
def total_population : ℕ := 56000

/-- The ratio of Boise's population to Seattle's population -/
def population_ratio : ℚ := 3 / 5

theorem city_population_ratio :
  ∃ (boise seattle : ℕ),
    boise + seattle + lake_view_population = total_population ∧
    lake_view_population = seattle + population_difference ∧
    population_ratio = boise / seattle := by
  sorry

end city_population_ratio_l887_88716


namespace sqrt_sum_reciprocals_l887_88774

theorem sqrt_sum_reciprocals : Real.sqrt ((1 : ℝ) / 25 + 1 / 36) = Real.sqrt 61 / 30 := by
  sorry

end sqrt_sum_reciprocals_l887_88774


namespace min_value_ab_l887_88775

theorem min_value_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (heq : a * b + 2 = 2 * (a + b)) :
  ∃ (min : ℝ), min = 6 + 4 * Real.sqrt 2 ∧ a * b ≥ min := by
sorry

end min_value_ab_l887_88775


namespace florist_roses_l887_88721

/-- A problem about a florist's roses -/
theorem florist_roses (initial : ℕ) (sold : ℕ) (final : ℕ) (picked : ℕ) : 
  initial = 11 → sold = 2 → final = 41 → picked = final - (initial - sold) → picked = 32 := by
  sorry

end florist_roses_l887_88721


namespace optimal_strategy_is_down_l887_88784

/-- Represents the direction of movement on the escalator -/
inductive Direction
  | Up
  | Down

/-- Represents the state of Petya and his hat on the escalators -/
structure EscalatorState where
  petyaPosition : ℝ  -- Position of Petya (0 = bottom, 1 = top)
  hatPosition : ℝ    -- Position of the hat (0 = bottom, 1 = top)
  petyaSpeed : ℝ     -- Petya's movement speed
  escalatorSpeed : ℝ  -- Speed of the escalator

/-- Calculates the time for Petya to reach his hat -/
def timeToReachHat (state : EscalatorState) (direction : Direction) : ℝ :=
  sorry

/-- Theorem stating that moving downwards is the optimal strategy -/
theorem optimal_strategy_is_down (state : EscalatorState) :
  state.petyaPosition = 0.5 →
  state.hatPosition = 1 →
  state.petyaSpeed > state.escalatorSpeed →
  state.petyaSpeed < 2 * state.escalatorSpeed →
  timeToReachHat state Direction.Down < timeToReachHat state Direction.Up :=
sorry

#check optimal_strategy_is_down

end optimal_strategy_is_down_l887_88784


namespace median_and_mean_of_set_l887_88782

theorem median_and_mean_of_set (m : ℝ) (h : m + 4 = 16) :
  let S : Finset ℝ := {m, m + 2, m + 4, m + 11, m + 18}
  (S.sum id) / S.card = 19 := by
sorry

end median_and_mean_of_set_l887_88782


namespace birthday_candles_cost_l887_88700

/-- Calculates the total cost of blue and green candles given the ratio and number of red candles -/
def total_cost_blue_green_candles (ratio_red blue green : ℕ) (num_red : ℕ) (cost_blue cost_green : ℕ) : ℕ :=
  let units_per_ratio := num_red / ratio_red
  let num_blue := units_per_ratio * blue
  let num_green := units_per_ratio * green
  num_blue * cost_blue + num_green * cost_green

/-- Theorem stating that the total cost of blue and green candles is $333 given the problem conditions -/
theorem birthday_candles_cost : 
  total_cost_blue_green_candles 5 3 7 45 3 4 = 333 := by
  sorry

end birthday_candles_cost_l887_88700


namespace three_chords_when_sixty_degrees_l887_88706

/-- Represents a configuration of concentric circles with tangent chords -/
structure ConcentricCirclesWithChords where
  /-- The measure of the angle formed by two adjacent chords at their intersection on the larger circle -/
  angle : ℝ
  /-- The number of chords needed to form a closed polygon -/
  num_chords : ℕ

/-- Theorem stating that when the angle between chords is 60°, exactly 3 chords are needed -/
theorem three_chords_when_sixty_degrees (config : ConcentricCirclesWithChords) :
  config.angle = 60 → config.num_chords = 3 :=
by sorry

end three_chords_when_sixty_degrees_l887_88706


namespace triangle_side_value_l887_88788

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.b^2 - t.c^2 + 2*t.a = 0 ∧ Real.tan t.C / Real.tan t.B = 3

theorem triangle_side_value (t : Triangle) (h : TriangleConditions t) : t.a = 4 := by
  sorry

end triangle_side_value_l887_88788


namespace difference_set_Q_P_l887_88793

-- Define the sets P and Q
def P : Set ℝ := {x | 1 - 2/x < 0}
def Q : Set ℝ := {x | |x - 2| < 1}

-- Define the difference set
def difference_set (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem difference_set_Q_P : 
  difference_set Q P = {x | 2 ≤ x ∧ x < 3} := by sorry

end difference_set_Q_P_l887_88793


namespace regular_ticket_price_l887_88729

/-- Calculates the price of each regular ticket given the initial savings,
    VIP ticket information, number of regular tickets, and remaining money. -/
theorem regular_ticket_price
  (initial_savings : ℕ)
  (vip_ticket_count : ℕ)
  (vip_ticket_price : ℕ)
  (regular_ticket_count : ℕ)
  (remaining_money : ℕ)
  (h1 : initial_savings = 500)
  (h2 : vip_ticket_count = 2)
  (h3 : vip_ticket_price = 100)
  (h4 : regular_ticket_count = 3)
  (h5 : remaining_money = 150)
  (h6 : initial_savings ≥ vip_ticket_count * vip_ticket_price + remaining_money) :
  (initial_savings - (vip_ticket_count * vip_ticket_price + remaining_money)) / regular_ticket_count = 50 :=
by sorry

end regular_ticket_price_l887_88729


namespace product_xyz_equals_negative_two_l887_88776

theorem product_xyz_equals_negative_two
  (x y z : ℝ)
  (h1 : x + 2 / y = 2)
  (h2 : y + 2 / z = 2) :
  x * y * z = -2 := by
sorry

end product_xyz_equals_negative_two_l887_88776


namespace sqrt_sum_inequality_l887_88724

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end sqrt_sum_inequality_l887_88724


namespace sammys_offer_per_record_l887_88717

theorem sammys_offer_per_record (total_records : ℕ) 
  (bryans_offer_high : ℕ) (bryans_offer_low : ℕ) (profit_difference : ℕ) :
  total_records = 200 →
  bryans_offer_high = 6 →
  bryans_offer_low = 1 →
  profit_difference = 100 →
  (total_records / 2 * bryans_offer_high + total_records / 2 * bryans_offer_low + profit_difference) / total_records = 4 := by
  sorry

end sammys_offer_per_record_l887_88717


namespace parent_age_problem_l887_88708

/-- Given the conditions about the relationship between a parent's age and their daughter's age,
    prove that the parent's current age is 40 years. -/
theorem parent_age_problem (Y D : ℕ) : 
  Y = 4 * D →                 -- You are 4 times your daughter's age today
  Y - 7 = 11 * (D - 7) →      -- 7 years earlier, you were 11 times her age
  Y = 40                      -- Your current age is 40
:= by sorry

end parent_age_problem_l887_88708


namespace z_in_first_quadrant_l887_88783

/-- The complex number i -/
def i : ℂ := Complex.I

/-- The given equation (1-2i)z = 5 -/
def given_equation (z : ℂ) : Prop := (1 - 2*i) * z = 5

/-- A complex number is in the first quadrant if its real and imaginary parts are both positive -/
def in_first_quadrant (z : ℂ) : Prop := 0 < z.re ∧ 0 < z.im

/-- 
If (1-2i)z = 5, then z is in the first quadrant of the complex plane
-/
theorem z_in_first_quadrant (z : ℂ) (h : given_equation z) : in_first_quadrant z :=
sorry

end z_in_first_quadrant_l887_88783


namespace count_possible_denominators_all_denominators_divide_999_fraction_denominator_in_possible_set_seven_possible_denominators_l887_88797

/-- Represents a three-digit number abc --/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_is_digit : a < 10
  b_is_digit : b < 10
  c_is_digit : c < 10
  not_all_nines : ¬(a = 9 ∧ b = 9 ∧ c = 9)
  not_all_zeros : ¬(a = 0 ∧ b = 0 ∧ c = 0)

/-- Converts a ThreeDigitNumber to its decimal value --/
def toDecimal (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The denominator of the fraction representation of 0.abc̅ --/
def denominator : Nat := 999

/-- The set of possible denominators for 0.abc̅ in lowest terms --/
def possibleDenominators : Finset Nat :=
  {3, 9, 27, 37, 111, 333, 999}

/-- Theorem stating that there are exactly 7 possible denominators --/
theorem count_possible_denominators :
    (possibleDenominators.card : Nat) = 7 := by sorry

/-- Theorem stating that all elements in possibleDenominators are factors of 999 --/
theorem all_denominators_divide_999 :
    ∀ d ∈ possibleDenominators, denominator % d = 0 := by sorry

/-- Theorem stating that for any ThreeDigitNumber, its fraction representation
    has a denominator in possibleDenominators --/
theorem fraction_denominator_in_possible_set (n : ThreeDigitNumber) :
    ∃ d ∈ possibleDenominators,
      (toDecimal n).gcd denominator = (denominator / d) := by sorry

/-- Main theorem proving that there are exactly 7 possible denominators --/
theorem seven_possible_denominators :
    ∃! (s : Finset Nat),
      (∀ n : ThreeDigitNumber,
        ∃ d ∈ s, (toDecimal n).gcd denominator = (denominator / d)) ∧
      s.card = 7 := by sorry

end count_possible_denominators_all_denominators_divide_999_fraction_denominator_in_possible_set_seven_possible_denominators_l887_88797


namespace spot_horn_proportion_is_half_l887_88769

/-- Represents the proportion of spotted females and horned males -/
def spot_horn_proportion (total_cows : ℕ) (female_to_male_ratio : ℕ) (spotted_horned_difference : ℕ) : ℚ :=
  let male_cows := total_cows / (female_to_male_ratio + 1)
  let female_cows := female_to_male_ratio * male_cows
  (spotted_horned_difference : ℚ) / (female_cows - male_cows)

/-- Theorem stating the proportion of spotted females and horned males -/
theorem spot_horn_proportion_is_half :
  spot_horn_proportion 300 2 50 = 1/2 := by
  sorry

end spot_horn_proportion_is_half_l887_88769


namespace nadia_mistakes_l887_88712

/-- Represents Nadia's piano playing statistics -/
structure PianoStats where
  mistakes_per_40_notes : ℕ
  notes_per_minute : ℕ
  playing_time : ℕ

/-- Calculates the number of mistakes Nadia makes given her piano playing statistics -/
def calculate_mistakes (stats : PianoStats) : ℕ :=
  let total_notes := stats.notes_per_minute * stats.playing_time
  let blocks_of_40 := total_notes / 40
  blocks_of_40 * stats.mistakes_per_40_notes

/-- Theorem stating that Nadia makes 36 mistakes in 8 minutes of playing -/
theorem nadia_mistakes (stats : PianoStats)
  (h1 : stats.mistakes_per_40_notes = 3)
  (h2 : stats.notes_per_minute = 60)
  (h3 : stats.playing_time = 8) :
  calculate_mistakes stats = 36 := by
  sorry


end nadia_mistakes_l887_88712


namespace rectangle_perimeter_width_ratio_l887_88714

theorem rectangle_perimeter_width_ratio 
  (area : ℝ) (length : ℝ) (width : ℝ) (perimeter : ℝ) :
  area = 150 →
  length = 15 →
  area = length * width →
  perimeter = 2 * (length + width) →
  perimeter / width = 5 := by
sorry

end rectangle_perimeter_width_ratio_l887_88714


namespace gcd_factorial_seven_eight_l887_88751

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcd_factorial_seven_eight_l887_88751


namespace total_money_value_l887_88749

-- Define the number of nickels (and quarters)
def num_coins : ℕ := 40

-- Define the value of a nickel in cents
def nickel_value : ℕ := 5

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the conversion rate from cents to dollars
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem total_money_value : 
  (num_coins * nickel_value + num_coins * quarter_value) / cents_per_dollar = 12 := by
  sorry

end total_money_value_l887_88749


namespace problem_solution_l887_88767

theorem problem_solution (x n : ℕ) (h1 : x = 9^n - 1) (h2 : Odd n) 
  (h3 : (Nat.factors x).length = 3) (h4 : 61 ∈ Nat.factors x) : x = 59048 := by
  sorry

end problem_solution_l887_88767


namespace sin_720_equals_0_l887_88750

theorem sin_720_equals_0 (n : ℤ) (h1 : -90 ≤ n ∧ n ≤ 90) (h2 : Real.sin (n * π / 180) = Real.sin (720 * π / 180)) : n = 0 := by
  sorry

end sin_720_equals_0_l887_88750


namespace ice_cream_scoops_l887_88703

/-- The number of ice cream scoops served to a family at Ice Cream Palace -/
def total_scoops (single_cone waffle_bowl banana_split double_cone : ℕ) : ℕ :=
  single_cone + waffle_bowl + banana_split + double_cone

/-- Theorem: Given the conditions of the ice cream orders, the total number of scoops served is 10 -/
theorem ice_cream_scoops :
  ∀ (single_cone waffle_bowl banana_split double_cone : ℕ),
    single_cone = 1 →
    banana_split = 3 * single_cone →
    waffle_bowl = banana_split + 1 →
    double_cone = 2 →
    total_scoops single_cone waffle_bowl banana_split double_cone = 10 :=
by
  sorry

#check ice_cream_scoops

end ice_cream_scoops_l887_88703


namespace jenny_weight_capacity_l887_88791

/-- Represents the recycling problem Jenny faces --/
structure RecyclingProblem where
  bottle_weight : ℕ
  can_weight : ℕ
  num_cans : ℕ
  bottle_price : ℕ
  can_price : ℕ
  total_earnings : ℕ

/-- Calculates the total weight Jenny can carry --/
def total_weight (p : RecyclingProblem) : ℕ :=
  let num_bottles := (p.total_earnings - p.num_cans * p.can_price) / p.bottle_price
  num_bottles * p.bottle_weight + p.num_cans * p.can_weight

/-- Theorem stating that Jenny can carry 100 ounces --/
theorem jenny_weight_capacity :
  ∃ (p : RecyclingProblem),
    p.bottle_weight = 6 ∧
    p.can_weight = 2 ∧
    p.num_cans = 20 ∧
    p.bottle_price = 10 ∧
    p.can_price = 3 ∧
    p.total_earnings = 160 ∧
    total_weight p = 100 := by
  sorry


end jenny_weight_capacity_l887_88791


namespace dance_class_theorem_l887_88720

theorem dance_class_theorem (U : Finset ℕ) (A B : Finset ℕ) : 
  Finset.card U = 40 →
  Finset.card A = 18 →
  Finset.card B = 22 →
  Finset.card (A ∩ B) = 10 →
  Finset.card (U \ (A ∪ B)) = 10 := by
  sorry

end dance_class_theorem_l887_88720


namespace chips_in_bag_is_24_l887_88701

/-- The number of chips in a bag, given the calorie and cost information --/
def chips_in_bag (calories_per_chip : ℕ) (cost_per_bag : ℕ) (total_calories : ℕ) (total_cost : ℕ) : ℕ :=
  (total_calories / calories_per_chip) / (total_cost / cost_per_bag)

/-- Theorem stating that there are 24 chips in a bag --/
theorem chips_in_bag_is_24 :
  chips_in_bag 10 2 480 4 = 24 := by
  sorry

end chips_in_bag_is_24_l887_88701


namespace rectangle_circle_equality_l887_88704

/-- The length of a rectangle with width 3 units whose perimeter equals the circumference of a circle with radius 5 units is 5π - 3. -/
theorem rectangle_circle_equality (l : ℝ) : 
  (2 * (l + 3) = 2 * π * 5) → l = 5 * π - 3 := by
  sorry

end rectangle_circle_equality_l887_88704


namespace barbaras_savings_l887_88756

/-- Calculates the current savings given the total cost, weekly allowance, and remaining weeks to save. -/
def currentSavings (totalCost : ℕ) (weeklyAllowance : ℕ) (remainingWeeks : ℕ) : ℕ :=
  totalCost - (weeklyAllowance * remainingWeeks)

/-- Proves that given the specific conditions, Barbara's current savings is $20. -/
theorem barbaras_savings :
  let watchCost : ℕ := 100
  let weeklyAllowance : ℕ := 5
  let remainingWeeks : ℕ := 16
  currentSavings watchCost weeklyAllowance remainingWeeks = 20 := by
  sorry

end barbaras_savings_l887_88756


namespace problem_1_l887_88710

theorem problem_1 : 2 * Real.tan (π / 3) - |Real.sqrt 3 - 2| - 3 * Real.sqrt 3 + (1 / 3)⁻¹ = 1 := by
  sorry

end problem_1_l887_88710


namespace result_is_fifty_l887_88707

-- Define the original number
def x : ℝ := 150

-- Define the percentage
def percentage : ℝ := 0.60

-- Define the subtracted value
def subtracted : ℝ := 40

-- Theorem to prove
theorem result_is_fifty : percentage * x - subtracted = 50 := by
  sorry

end result_is_fifty_l887_88707


namespace solution_set_part1_min_value_part2_l887_88733

-- Define the function f
def f (x a b : ℝ) : ℝ := |2*x + a| + |x - b|

-- Part 1
theorem solution_set_part1 :
  ∀ x : ℝ, f x (-2) 1 < 6 ↔ -1 < x ∧ x < 3 := by sorry

-- Part 2
theorem min_value_part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x : ℝ, f x a b = 1 ∧ ∀ y : ℝ, f y a b ≥ 1) →
  (∀ c d : ℝ, c > 0 → d > 0 → 2/c + 1/d ≥ 4) ∧
  (∃ e g : ℝ, e > 0 ∧ g > 0 ∧ 2/e + 1/g = 4) := by sorry

end solution_set_part1_min_value_part2_l887_88733


namespace prism_pyramid_sum_l887_88796

/-- A shape formed by adding a pyramid to one square face of a rectangular prism -/
structure PrismPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_faces : ℕ
  pyramid_edges : ℕ
  pyramid_vertices : ℕ

/-- The sum of faces, edges, and vertices of the PrismPyramid -/
def total_sum (pp : PrismPyramid) : ℕ :=
  (pp.prism_faces - 1 + pp.pyramid_faces) + 
  (pp.prism_edges + pp.pyramid_edges) + 
  (pp.prism_vertices + pp.pyramid_vertices)

/-- Theorem stating that the total sum is 34 -/
theorem prism_pyramid_sum :
  ∀ (pp : PrismPyramid), 
    pp.prism_faces = 6 ∧ 
    pp.prism_edges = 12 ∧ 
    pp.prism_vertices = 8 ∧
    pp.pyramid_faces = 4 ∧
    pp.pyramid_edges = 4 ∧
    pp.pyramid_vertices = 1 →
    total_sum pp = 34 := by
  sorry

end prism_pyramid_sum_l887_88796


namespace log_inequality_implies_sum_nonnegative_l887_88728

theorem log_inequality_implies_sum_nonnegative (x y : ℝ) :
  (Real.log 3 / Real.log 2)^x + (Real.log 5 / Real.log 3)^y ≥ 
  (Real.log 2 / Real.log 3)^y + (Real.log 3 / Real.log 5)^x →
  x + y ≥ 0 := by
  sorry

end log_inequality_implies_sum_nonnegative_l887_88728


namespace complex_square_pure_imaginary_l887_88764

theorem complex_square_pure_imaginary (a : ℝ) : 
  let z : ℂ := a + 3*I
  (∃ b : ℝ, z^2 = b*I ∧ b ≠ 0) → (a = 3 ∨ a = -3) :=
by sorry

end complex_square_pure_imaginary_l887_88764


namespace determine_m_l887_88799

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + m
def g (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*m

-- State the theorem
theorem determine_m : ∃ m : ℝ, 2 * (f m 3) = 3 * (g m 3) ∧ m = 0 := by
  sorry

end determine_m_l887_88799


namespace smallest_five_digit_in_pascal_l887_88732

/-- Pascal's triangle function -/
def pascal (n k : ℕ) : ℕ := sorry

/-- A number is five-digit if it's between 10000 and 99999 inclusive -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_five_digit_in_pascal :
  ∃ (n k : ℕ), pascal n k = 10000 ∧
  (∀ (m l : ℕ), pascal m l < 10000 ∨ (pascal m l = 10000 ∧ m ≥ n)) :=
sorry

end smallest_five_digit_in_pascal_l887_88732


namespace perpendicular_lines_a_value_l887_88744

/-- Given two lines l₁: ax + y + 1 = 0 and l₂: x - 2y + 1 = 0,
    if they are perpendicular, then a = 2 -/
theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ x y, ax + y + 1 = 0 ∧ x - 2*y + 1 = 0) →
  (∀ x₁ y₁ x₂ y₂, ax₁ + y₁ + 1 = 0 ∧ x₁ - 2*y₁ + 1 = 0 ∧
                   ax₂ + y₂ + 1 = 0 ∧ x₂ - 2*y₂ + 1 = 0 →
                   (x₂ - x₁) * ((y₂ - y₁) / (x₂ - x₁)) = -1) →
  a = 2 :=
by sorry

end perpendicular_lines_a_value_l887_88744


namespace modulo_thirteen_residue_l887_88781

theorem modulo_thirteen_residue : (247 + 5 * 39 + 7 * 143 + 4 * 15) % 13 = 8 := by
  sorry

end modulo_thirteen_residue_l887_88781
