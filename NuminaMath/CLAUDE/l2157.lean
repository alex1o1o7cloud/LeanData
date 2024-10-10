import Mathlib

namespace divisibility_statements_l2157_215782

theorem divisibility_statements : 
  (∃ n : ℤ, 25 = 5 * n) ∧ 
  (∃ n : ℤ, 209 = 19 * n) ∧ 
  ¬(∃ n : ℤ, 63 = 19 * n) ∧
  (∃ n : ℤ, 140 = 7 * n) ∧
  (∃ n : ℤ, 90 = 30 * n) ∧
  (∃ n : ℤ, 34 = 17 * n) ∧
  (∃ n : ℤ, 68 = 17 * n) := by
  sorry

#check divisibility_statements

end divisibility_statements_l2157_215782


namespace number_equation_solution_l2157_215790

theorem number_equation_solution : ∃ x : ℝ, (3 * x - 6 = 2 * x) ∧ (x = 6) := by
  sorry

end number_equation_solution_l2157_215790


namespace william_napkins_before_l2157_215793

/-- The number of napkins William had before receiving napkins from Olivia and Amelia. -/
def napkins_before : ℕ := sorry

/-- The number of napkins Olivia gave to William. -/
def olivia_napkins : ℕ := 10

/-- The number of napkins Amelia gave to William. -/
def amelia_napkins : ℕ := 2 * olivia_napkins

/-- The total number of napkins William has now. -/
def total_napkins : ℕ := 45

theorem william_napkins_before :
  napkins_before = total_napkins - (olivia_napkins + amelia_napkins) :=
by sorry

end william_napkins_before_l2157_215793


namespace minimum_value_theorem_l2157_215791

theorem minimum_value_theorem (x y m : ℝ) :
  y ≥ 1 →
  y ≤ 2 * x - 1 →
  x + y ≤ m →
  (∀ x' y' : ℝ, y' ≥ 1 → y' ≤ 2 * x' - 1 → x' + y' ≤ m → x - y ≤ x' - y') →
  x - y = 0 →
  m = 2 :=
by sorry

end minimum_value_theorem_l2157_215791


namespace max_y_value_l2157_215737

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -8) : 
  y ≤ 27 ∧ ∃ (x' y' : ℤ), x' * y' + 7 * x' + 6 * y' = -8 ∧ y' = 27 := by
  sorry

end max_y_value_l2157_215737


namespace water_current_speed_l2157_215707

/-- The speed of a water current given swimmer's speed and time against current -/
theorem water_current_speed (swimmer_speed : ℝ) (distance : ℝ) (time : ℝ) :
  swimmer_speed = 4 →
  distance = 5 →
  time = 2.5 →
  ∃ (current_speed : ℝ), 
    time = distance / (swimmer_speed - current_speed) ∧
    current_speed = 2 := by
  sorry

end water_current_speed_l2157_215707


namespace smallest_n_congruence_l2157_215731

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 3 * n ≡ 1356 [ZMOD 22]) → n ≥ 12 :=
sorry

end smallest_n_congruence_l2157_215731


namespace expansion_coefficient_implies_k_value_l2157_215745

theorem expansion_coefficient_implies_k_value (k : ℕ+) :
  (15 * k ^ 4 : ℕ) < 120 → k = 1 := by
  sorry

end expansion_coefficient_implies_k_value_l2157_215745


namespace modulus_of_complex_number_l2157_215777

theorem modulus_of_complex_number :
  let z : ℂ := -1 + 3 * Complex.I
  Complex.abs z = Real.sqrt 10 := by sorry

end modulus_of_complex_number_l2157_215777


namespace steves_commute_l2157_215715

/-- The distance from Steve's house to work -/
def distance : ℝ := by sorry

/-- Steve's speed on the way to work -/
def speed_to_work : ℝ := by sorry

/-- Steve's speed on the way back from work -/
def speed_from_work : ℝ := 14

/-- The total time Steve spends on the roads -/
def total_time : ℝ := 6

theorem steves_commute :
  (speed_from_work = 2 * speed_to_work) →
  (distance / speed_to_work + distance / speed_from_work = total_time) →
  distance = 28 := by sorry

end steves_commute_l2157_215715


namespace min_value_theorem_l2157_215752

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  x + 81 / x ≥ 18 ∧ (x + 81 / x = 18 ↔ x = 9) := by sorry

end min_value_theorem_l2157_215752


namespace weight_of_N2O3_l2157_215704

/-- The molar mass of nitrogen in g/mol -/
def molar_mass_N : ℝ := 14.01

/-- The molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- The number of moles of N2O3 -/
def moles_N2O3 : ℝ := 7

/-- The molar mass of N2O3 in g/mol -/
def molar_mass_N2O3 : ℝ := 2 * molar_mass_N + 3 * molar_mass_O

/-- The weight of N2O3 in grams -/
def weight_N2O3 : ℝ := moles_N2O3 * molar_mass_N2O3

theorem weight_of_N2O3 : weight_N2O3 = 532.14 := by
  sorry

end weight_of_N2O3_l2157_215704


namespace intersection_equality_l2157_215783

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.cos (Real.arccos p.1)}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.arccos (Real.cos p.1)}

-- Define the intersection set
def intersection_set : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 ∧ -1 ≤ p.1 ∧ p.1 ≤ 1}

-- Theorem statement
theorem intersection_equality : A ∩ B = intersection_set := by
  sorry

end intersection_equality_l2157_215783


namespace initial_hay_bales_l2157_215738

theorem initial_hay_bales (better_quality_cost previous_cost : ℚ) 
  (cost_difference : ℚ) : 
  better_quality_cost = 18 →
  previous_cost = 15 →
  cost_difference = 210 →
  ∃ x : ℚ, x = 10 ∧ 2 * better_quality_cost * x - previous_cost * x = cost_difference :=
by sorry

end initial_hay_bales_l2157_215738


namespace corner_stationery_sales_proportion_l2157_215747

theorem corner_stationery_sales_proportion :
  let total_sales_percent : ℝ := 100
  let markers_percent : ℝ := 25
  let notebooks_percent : ℝ := 47
  total_sales_percent - (markers_percent + notebooks_percent) = 28 := by
sorry

end corner_stationery_sales_proportion_l2157_215747


namespace equation_solutions_l2157_215756

theorem equation_solutions (x : ℝ) :
  x ≠ 2 → x ≠ 4 →
  ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) = 
   (x - 2) * (x - 4) * (x - 2)) ↔ 
  (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end equation_solutions_l2157_215756


namespace stratified_sampling_middle_school_l2157_215750

/-- Represents the number of students in a school -/
structure School :=
  (students : ℕ)

/-- Represents a sampling strategy -/
structure Sampling :=
  (total_population : ℕ)
  (sample_size : ℕ)
  (schools : Vector School 3)

/-- Checks if the number of students in schools forms an arithmetic sequence -/
def is_arithmetic_sequence (schools : Vector School 3) : Prop :=
  schools[1].students - schools[0].students = schools[2].students - schools[1].students

/-- Theorem: In a stratified sampling of 120 students from 1500 students 
    distributed in an arithmetic sequence across 3 schools, 
    the number of students sampled from the middle school (B) is 40 -/
theorem stratified_sampling_middle_school 
  (sampling : Sampling) 
  (h1 : sampling.total_population = 1500)
  (h2 : sampling.sample_size = 120)
  (h3 : is_arithmetic_sequence sampling.schools)
  : (sampling.sample_size / 3 : ℕ) = 40 := by
  sorry

end stratified_sampling_middle_school_l2157_215750


namespace mutually_exclusive_necessary_not_sufficient_l2157_215765

open Set

universe u

variable {Ω : Type u} [MeasurableSpace Ω]
variable (A₁ A₂ : Set Ω)

def mutually_exclusive (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅

def complementary (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅ ∧ A₁ ∪ A₂ = univ

theorem mutually_exclusive_necessary_not_sufficient :
  (complementary A₁ A₂ → mutually_exclusive A₁ A₂) ∧
  ¬(mutually_exclusive A₁ A₂ → complementary A₁ A₂) := by sorry

end mutually_exclusive_necessary_not_sufficient_l2157_215765


namespace share_of_a_l2157_215743

theorem share_of_a (total : ℝ) (a b c : ℝ) : 
  total = 500 →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a = 125 := by
sorry

end share_of_a_l2157_215743


namespace three_by_three_min_cuts_l2157_215799

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a straight-line cut on the grid -/
inductive Cut
  | Vertical : ℕ → Cut
  | Horizontal : ℕ → Cut

/-- Defines the minimum number of cuts required to divide a grid into unit squares -/
def min_cuts (g : Grid) : ℕ := sorry

/-- Theorem stating that a 3x3 grid requires exactly 4 cuts to be divided into unit squares -/
theorem three_by_three_min_cuts :
  ∀ (g : Grid), g.size = 3 → min_cuts g = 4 := by sorry

end three_by_three_min_cuts_l2157_215799


namespace min_ab_perpendicular_lines_l2157_215734

/-- Given two perpendicular lines and b > 0, the minimum value of ab is 2 -/
theorem min_ab_perpendicular_lines (b : ℝ) (a : ℝ) (h : b > 0) :
  (∃ x y, (b^2 + 1) * x + a * y + 2 = 0) ∧ 
  (∃ x y, x - b^2 * y - 1 = 0) ∧
  ((b^2 + 1) * (1 / b^2) = -1) →
  (∀ c, ab ≥ c → c ≤ 2) ∧ (∃ d, ab = d ∧ d = 2) :=
by sorry

end min_ab_perpendicular_lines_l2157_215734


namespace chef_dinner_meals_l2157_215714

/-- Calculates the number of meals prepared for dinner given lunch and dinner information -/
def meals_prepared_for_dinner (lunch_prepared : ℕ) (lunch_sold : ℕ) (dinner_total : ℕ) : ℕ :=
  dinner_total - (lunch_prepared - lunch_sold)

/-- Proves that the chef prepared 5 meals for dinner -/
theorem chef_dinner_meals :
  meals_prepared_for_dinner 17 12 10 = 5 := by
  sorry

end chef_dinner_meals_l2157_215714


namespace fraction_problem_l2157_215762

theorem fraction_problem (a b m : ℚ) : 
  (2 * (1/2) - b = 0) →  -- Fraction is undefined when x = 0.5
  ((-2 + a) / (2 * (-2) - b) = 0) →  -- Fraction equals 0 when x = -2
  ((m + a) / (2 * m - b) = 1) →  -- Fraction equals 1 when x = m
  m = 3 := by sorry

end fraction_problem_l2157_215762


namespace two_valid_m_values_l2157_215706

theorem two_valid_m_values : 
  ∃! (s : Finset ℕ), 
    (∀ m ∈ s, m > 0 ∧ (3087 : ℤ) ∣ (m^2 - 3)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (3087 : ℤ) ∣ (m^2 - 3) → m ∈ s) ∧ 
    s.card = 2 := by
  sorry

end two_valid_m_values_l2157_215706


namespace monday_temp_value_l2157_215792

/-- The average temperature for a week -/
def average_temp : ℝ := 99

/-- The number of days in a week -/
def num_days : ℕ := 7

/-- The temperatures for 6 days of the week -/
def known_temps : List ℝ := [99.1, 98.7, 99.3, 99.8, 99, 98.9]

/-- The temperature on Monday -/
def monday_temp : ℝ := num_days * average_temp - known_temps.sum

theorem monday_temp_value : monday_temp = 98.2 := by sorry

end monday_temp_value_l2157_215792


namespace community_avg_age_l2157_215785

-- Define the ratio of women to men
def women_to_men_ratio : ℚ := 7 / 5

-- Define the average age of women
def avg_age_women : ℝ := 30

-- Define the average age of men
def avg_age_men : ℝ := 35

-- Theorem statement
theorem community_avg_age :
  let total_population := women_to_men_ratio + 1
  let weighted_age_sum := women_to_men_ratio * avg_age_women + avg_age_men
  weighted_age_sum / total_population = 385 / 12 :=
by sorry

end community_avg_age_l2157_215785


namespace sin_50_plus_sqrt3_tan_10_equals_one_l2157_215735

theorem sin_50_plus_sqrt3_tan_10_equals_one :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end sin_50_plus_sqrt3_tan_10_equals_one_l2157_215735


namespace pet_food_ratio_l2157_215754

/-- Represents the amounts of pet food in kilograms -/
structure PetFood where
  dog : ℕ
  cat : ℕ
  bird : ℕ

/-- The total amount of pet food -/
def total_food (pf : PetFood) : ℕ := pf.dog + pf.cat + pf.bird

/-- The ratio of pet food types -/
def food_ratio (pf : PetFood) : (ℕ × ℕ × ℕ) :=
  let gcd := Nat.gcd pf.dog (Nat.gcd pf.cat pf.bird)
  (pf.dog / gcd, pf.cat / gcd, pf.bird / gcd)

theorem pet_food_ratio : 
  let bought := PetFood.mk 15 10 5
  let final := PetFood.mk 40 15 5
  let initial := PetFood.mk (final.dog - bought.dog) (final.cat - bought.cat) (final.bird - bought.bird)
  total_food final = 60 →
  food_ratio final = (8, 3, 1) := by
  sorry

end pet_food_ratio_l2157_215754


namespace university_packaging_volume_l2157_215736

/-- The minimum volume needed to package the university's collection given the box dimensions, cost per box, and minimum amount spent. -/
theorem university_packaging_volume
  (box_length : ℝ)
  (box_width : ℝ)
  (box_height : ℝ)
  (cost_per_box : ℝ)
  (min_amount_spent : ℝ)
  (h_box_length : box_length = 20)
  (h_box_width : box_width = 20)
  (h_box_height : box_height = 12)
  (h_cost_per_box : cost_per_box = 0.5)
  (h_min_amount_spent : min_amount_spent = 200) :
  (min_amount_spent / cost_per_box) * (box_length * box_width * box_height) = 1920000 :=
by sorry

end university_packaging_volume_l2157_215736


namespace power_of_power_l2157_215700

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end power_of_power_l2157_215700


namespace nectarines_per_box_l2157_215723

theorem nectarines_per_box (num_crates : ℕ) (oranges_per_crate : ℕ) (num_boxes : ℕ) (total_fruits : ℕ) :
  num_crates = 12 →
  oranges_per_crate = 150 →
  num_boxes = 16 →
  total_fruits = 2280 →
  (total_fruits - num_crates * oranges_per_crate) / num_boxes = 30 :=
by sorry

end nectarines_per_box_l2157_215723


namespace unique_base_solution_l2157_215741

def base_to_decimal (n : ℕ) (b : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * b^2 + tens * b + ones

theorem unique_base_solution :
  ∃! (c : ℕ), c > 0 ∧ base_to_decimal 243 c + base_to_decimal 156 c = base_to_decimal 421 c :=
by sorry

end unique_base_solution_l2157_215741


namespace consecutive_integer_averages_l2157_215732

theorem consecutive_integer_averages (a b : ℤ) (h_positive : a > 0) : 
  ((7 * a + 21) / 7 = b) → 
  ((7 * b + 21) / 7 = a + 6) := by
sorry

end consecutive_integer_averages_l2157_215732


namespace l_shaped_tiling_exists_l2157_215772

/-- An L-shaped piece consisting of three squares -/
inductive LPiece
| mk : LPiece

/-- A square grid of side length 2^n -/
def Square (n : ℕ) := Fin (2^n) × Fin (2^n)

/-- A cell in the square grid -/
def Cell (n : ℕ) := Square n

/-- A tiling of the square grid using L-shaped pieces -/
def Tiling (n : ℕ) := Square n → Option LPiece

/-- Predicate to check if a tiling is valid -/
def is_valid_tiling (n : ℕ) (t : Tiling n) (removed : Cell n) : Prop :=
  ∀ (c : Cell n), c ≠ removed → ∃ (piece : LPiece), t c = some piece

/-- The main theorem: for all n, there exists a valid tiling of a 2^n x 2^n square
    with one cell removed using L-shaped pieces -/
theorem l_shaped_tiling_exists (n : ℕ) :
  ∀ (removed : Cell n), ∃ (t : Tiling n), is_valid_tiling n t removed :=
sorry

end l_shaped_tiling_exists_l2157_215772


namespace function_fixed_point_l2157_215784

theorem function_fixed_point (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end function_fixed_point_l2157_215784


namespace geometric_sequence_sum_l2157_215780

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 1 + a 2 = 40) →
  (a 3 + a 4 = 60) →
  (a 7 + a 8 = 135) :=
by sorry

end geometric_sequence_sum_l2157_215780


namespace population_trend_decreasing_l2157_215786

theorem population_trend_decreasing 
  (k : ℝ) 
  (h1 : -1 < k) 
  (h2 : k < 0) 
  (P : ℝ) 
  (hP : P > 0) :
  ∀ n : ℕ, ∀ m : ℕ, n < m → P * (1 + k)^n > P * (1 + k)^m :=
sorry

end population_trend_decreasing_l2157_215786


namespace probability_point_in_circle_l2157_215779

/-- The probability that a randomly selected point from a square with side length 6
    is within a circle of radius 2 centered at the origin is π/9 -/
theorem probability_point_in_circle (s : ℝ) (r : ℝ) : 
  s = 6 → r = 2 → (π * r^2) / (s^2) = π / 9 := by
  sorry

end probability_point_in_circle_l2157_215779


namespace first_triangle_isosceles_l2157_215729

theorem first_triangle_isosceles (α β γ δ ε : ℝ) :
  α + β + γ = 180 →
  α + β = δ →
  β + γ = ε →
  0 < α ∧ 0 < β ∧ 0 < γ →
  0 < δ ∧ 0 < ε →
  ∃ (θ : ℝ), (α = θ ∧ γ = θ) ∨ (α = θ ∧ β = θ) ∨ (β = θ ∧ γ = θ) :=
by sorry

end first_triangle_isosceles_l2157_215729


namespace max_difference_of_five_integers_l2157_215722

theorem max_difference_of_five_integers (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℝ) / 5 = 50 →
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e →
  e ≤ 58 →
  e - a ≤ 34 :=
by sorry

end max_difference_of_five_integers_l2157_215722


namespace smallest_cut_length_for_triangle_l2157_215778

theorem smallest_cut_length_for_triangle (a b c : ℕ) (ha : a = 12) (hb : b = 18) (hc : c = 20) :
  ∃ (x : ℕ), x = 10 ∧
  (∀ (y : ℕ), y < x → (a - y) + (b - y) > (c - y)) ∧
  (a - x) + (b - x) ≤ (c - x) :=
by sorry

end smallest_cut_length_for_triangle_l2157_215778


namespace sum_of_integers_l2157_215776

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 4)
  (eq4 : d - a + b = 3)
  (eq5 : a + b + c - d = 10) : 
  a + b + c + d = 16 := by
  sorry

end sum_of_integers_l2157_215776


namespace set_operation_equality_l2157_215759

def U : Finset Nat := {1,2,3,4,5,6,7}
def A : Finset Nat := {2,4,5,7}
def B : Finset Nat := {3,4,5}

theorem set_operation_equality : (U \ A) ∪ (U \ B) = {1,2,3,6,7} := by
  sorry

end set_operation_equality_l2157_215759


namespace inequality_solution_set_l2157_215771

theorem inequality_solution_set : 
  {x : ℝ | 2 ≥ (1 / (x - 1))} = Set.Iic 1 ∪ Set.Ici (3/2) :=
by sorry

end inequality_solution_set_l2157_215771


namespace amusement_park_visitors_l2157_215775

/-- Amusement park visitor count problem -/
theorem amusement_park_visitors :
  let morning_visitors : ℕ := 473
  let noon_departures : ℕ := 179
  let afternoon_visitors : ℕ := 268
  let total_visitors : ℕ := morning_visitors + afternoon_visitors
  let current_visitors : ℕ := morning_visitors - noon_departures + afternoon_visitors
  (total_visitors = 741) ∧ (current_visitors = 562) := by
  sorry

end amusement_park_visitors_l2157_215775


namespace topological_minor_theorem_l2157_215797

-- Define the average degree of a graph
def average_degree (G : Graph) : ℝ := sorry

-- Define what it means for a graph to contain another graph as a topological minor
def contains_topological_minor (G H : Graph) : Prop := sorry

-- Define the complete graph on r vertices
def complete_graph (r : ℕ) : Graph := sorry

theorem topological_minor_theorem :
  ∃ (c : ℝ), c = 10 ∧
  ∀ (r : ℕ) (G : Graph),
    average_degree G ≥ c * r^2 →
    contains_topological_minor G (complete_graph r) :=
sorry

end topological_minor_theorem_l2157_215797


namespace function_range_in_unit_interval_l2157_215763

theorem function_range_in_unit_interval (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ z : ℝ, 0 ≤ f z ∧ f z ≤ 1 := by
  sorry

end function_range_in_unit_interval_l2157_215763


namespace f_at_negative_one_l2157_215748

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 150*x + c

-- State the theorem
theorem f_at_negative_one (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c (-1) = 3733.25 := by
sorry


end f_at_negative_one_l2157_215748


namespace geometric_sequence_sum_l2157_215711

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (3 * a 2^2 - 10 * a 2 + 3 = 0) →
  (3 * a 6^2 - 10 * a 6 + 3 = 0) →
  (1 / a 2 + 1 / a 6 + a 4^2 = 13/3) :=
by sorry

end geometric_sequence_sum_l2157_215711


namespace horner_rule_v2_l2157_215769

def horner_polynomial (x : ℚ) : ℚ := 1 + 2*x + x^2 - 3*x^3 + 2*x^4

def horner_v2 (x : ℚ) : ℚ :=
  let v1 := 2*x^3 - 3*x^2 + x
  v1 * x + 2

theorem horner_rule_v2 :
  horner_v2 (-1) = -4 :=
by sorry

end horner_rule_v2_l2157_215769


namespace solution_set_of_inequality_l2157_215702

open Real

theorem solution_set_of_inequality (f : ℝ → ℝ) (f_diff : Differentiable ℝ f) :
  (f 0 = 2) →
  (∀ x, f x + deriv f x > 1) →
  (∀ x, (exp x * f x > exp x + 1) ↔ x > 0) :=
by sorry

end solution_set_of_inequality_l2157_215702


namespace age_problem_l2157_215717

theorem age_problem (mehki_age jordyn_age certain_age : ℕ) : 
  mehki_age = jordyn_age + 10 →
  jordyn_age = 2 * certain_age →
  mehki_age = 22 →
  certain_age = 6 := by
  sorry

end age_problem_l2157_215717


namespace car_speed_problem_l2157_215719

/-- Proves that given a car traveling 75% of a trip at 50 mph and the remaining 25% at speed s,
    if the average speed for the entire trip is 50 mph, then s must equal 50 mph. -/
theorem car_speed_problem (D : ℝ) (s : ℝ) (h1 : D > 0) (h2 : s > 0) : 
  (0.75 * D / 50 + 0.25 * D / s) = D / 50 → s = 50 := by
  sorry

end car_speed_problem_l2157_215719


namespace circle_standard_equation_l2157_215798

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the points A and B
def A : ℝ × ℝ := (1, -5)
def B : ℝ × ℝ := (2, -2)

-- Define the line equation
def line_equation (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0

-- Define the circle equation
def circle_equation (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_standard_equation :
  ∃ (c : Circle),
    circle_equation c A ∧
    circle_equation c B ∧
    line_equation c.center ∧
    c.center = (-3, -2) ∧
    c.radius = 5 :=
by sorry

end circle_standard_equation_l2157_215798


namespace domain_v_correct_l2157_215739

/-- The domain of v(x, y) = 1/√(x + y) where x and y are real numbers -/
def domain_v : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 > -p.1}

/-- The function v(x, y) = 1/√(x + y) -/
noncomputable def v (p : ℝ × ℝ) : ℝ :=
  1 / Real.sqrt (p.1 + p.2)

theorem domain_v_correct :
  ∀ p : ℝ × ℝ, p ∈ domain_v ↔ ∃ z : ℝ, v p = z :=
by sorry

end domain_v_correct_l2157_215739


namespace sum_of_squares_of_roots_l2157_215740

/-- Given a quadratic equation 10x^2 + 15x - 25 = 0, 
    the sum of the squares of its roots is equal to 29/4 -/
theorem sum_of_squares_of_roots : 
  let a : ℚ := 10
  let b : ℚ := 15
  let c : ℚ := -25
  let x₁ : ℚ := (-b + (b^2 - 4*a*c).sqrt) / (2*a)
  let x₂ : ℚ := (-b - (b^2 - 4*a*c).sqrt) / (2*a)
  x₁^2 + x₂^2 = 29/4 := by
sorry


end sum_of_squares_of_roots_l2157_215740


namespace intersection_points_theorem_l2157_215794

theorem intersection_points_theorem :
  let roots : Set ℝ := {1, 2}
  let eq_A : ℝ → ℝ → Prop := λ x y ↦ (y = x^2 ∨ y = 3*x)
  let eq_B : ℝ → ℝ → Prop := λ x y ↦ (y = x^2 - 3*x + 2 ∨ y = 2)
  let eq_C : ℝ → ℝ → Prop := λ x y ↦ (y = x ∨ y = x - 2)
  let eq_D : ℝ → ℝ → Prop := λ x y ↦ (y = x^2 - 3*x + 3 ∨ y = 3)
  (∀ x y, eq_A x y → x ∉ roots) ∧
  (∀ x y, eq_B x y → x ∉ roots) ∧
  (¬∃ x y, eq_C x y) ∧
  (∀ x y, eq_D x y → x ∉ roots) := by
  sorry


end intersection_points_theorem_l2157_215794


namespace mod_equivalence_2023_l2157_215728

theorem mod_equivalence_2023 : ∃! n : ℕ, n ≤ 6 ∧ n ≡ -2023 [ZMOD 7] ∧ n = 0 := by
  sorry

end mod_equivalence_2023_l2157_215728


namespace f_neg_a_eq_zero_l2157_215764

noncomputable def f (x : ℝ) : ℝ := x * Real.log (Real.exp (2 * x) + 1) - x^2 + 1

theorem f_neg_a_eq_zero (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end f_neg_a_eq_zero_l2157_215764


namespace pure_imaginary_fraction_l2157_215712

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (b : ℝ), (a + Complex.I) / (1 + Complex.I) = Complex.I * b) → a = -1 := by
  sorry

end pure_imaginary_fraction_l2157_215712


namespace max_servings_is_56_l2157_215787

/-- Represents the recipe requirements for one serving of salad -/
structure Recipe where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the warehouse -/
structure Warehouse where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of servings that can be made -/
def max_servings (recipe : Recipe) (warehouse : Warehouse) : ℕ :=
  min
    (warehouse.cucumbers / recipe.cucumbers)
    (min
      (warehouse.tomatoes / recipe.tomatoes)
      (min
        (warehouse.brynza / recipe.brynza)
        (warehouse.peppers / recipe.peppers)))

/-- Theorem: The maximum number of servings that can be made is 56 -/
theorem max_servings_is_56 :
  let recipe := Recipe.mk 2 2 75 1
  let warehouse := Warehouse.mk 117 116 4200 60
  max_servings recipe warehouse = 56 := by
  sorry

#eval max_servings (Recipe.mk 2 2 75 1) (Warehouse.mk 117 116 4200 60)

end max_servings_is_56_l2157_215787


namespace largest_common_term_l2157_215789

/-- The largest common term of two arithmetic sequences -/
theorem largest_common_term : ∃ (n m : ℕ), 
  138 = 2 + 4 * n ∧ 
  138 = 5 + 5 * m ∧ 
  138 ≤ 150 ∧ 
  ∀ (k l : ℕ), (2 + 4 * k = 5 + 5 * l) → (2 + 4 * k ≤ 150) → (2 + 4 * k ≤ 138) :=
sorry

end largest_common_term_l2157_215789


namespace base_nine_to_ten_l2157_215730

theorem base_nine_to_ten : 
  (3 * 9^4 + 9 * 9^3 + 4 * 9^2 + 5 * 9^1 + 7 * 9^0) = 26620 := by
  sorry

end base_nine_to_ten_l2157_215730


namespace triangle_inequality_example_l2157_215701

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_example : can_form_triangle 3 4 5 := by
  sorry

end triangle_inequality_example_l2157_215701


namespace problem_1_problem_2_problem_3_problem_4_l2157_215749

-- Problem 1
theorem problem_1 : (1) - 1^2 + 16 / (-4)^2 * (-3 - 1) = -5 := by sorry

-- Problem 2
theorem problem_2 : 5 * (5/8) - 2 * (-5/8) - 7 * (5/8) = 0 := by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) : x - 3*y - (-3*x + 4*y) = 4*x - 7*y := by sorry

-- Problem 4
theorem problem_4 (a b : ℝ) : 3*a - 4*(a - 3/2*b) - 2*(4*b + 5*a) = -11*a - 2*b := by sorry

end problem_1_problem_2_problem_3_problem_4_l2157_215749


namespace cookies_eaten_l2157_215705

theorem cookies_eaten (original : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  original = 18 → remaining = 9 → eaten = original - remaining → eaten = 9 := by
  sorry

end cookies_eaten_l2157_215705


namespace trivia_team_size_l2157_215713

theorem trivia_team_size :
  ∀ (original_members : ℕ),
  (original_members ≥ 2) →
  (4 * (original_members - 2) = 20) →
  original_members = 7 :=
by
  sorry

end trivia_team_size_l2157_215713


namespace sqrt3_plus_sqrt2_inverse_of_sqrt3_minus_sqrt2_l2157_215774

theorem sqrt3_plus_sqrt2_inverse_of_sqrt3_minus_sqrt2 :
  (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = 1 := by
  sorry

end sqrt3_plus_sqrt2_inverse_of_sqrt3_minus_sqrt2_l2157_215774


namespace perimeter_of_figure_c_l2157_215720

/-- Given a large rectangle made up of 20 identical small rectangles,
    this theorem proves that if the perimeter of figure A (6x2 small rectangles)
    and figure B (4x6 small rectangles) are both 56 cm,
    then the perimeter of figure C (2x6 small rectangles) is 40 cm. -/
theorem perimeter_of_figure_c (x y : ℝ) 
  (h1 : 6 * x + 2 * y = 56)  -- Perimeter of figure A
  (h2 : 4 * x + 6 * y = 56)  -- Perimeter of figure B
  : 2 * x + 6 * y = 40 :=    -- Perimeter of figure C
by sorry

end perimeter_of_figure_c_l2157_215720


namespace jake_has_seven_peaches_l2157_215755

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 19

/-- The number of peaches Jake has fewer than Steven -/
def jake_fewer_than_steven : ℕ := 12

/-- The number of peaches Jake has more than Jill -/
def jake_more_than_jill : ℕ := 72

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - jake_fewer_than_steven

theorem jake_has_seven_peaches : jake_peaches = 7 := by
  sorry

end jake_has_seven_peaches_l2157_215755


namespace inequality_solution_l2157_215760

theorem inequality_solution (x : ℝ) : (x + 3) * (x - 1) < 0 ↔ -3 < x ∧ x < 1 := by
  sorry

end inequality_solution_l2157_215760


namespace expression_equals_101_15_closest_integer_is_6_l2157_215796

-- Define the expression
def expression : ℚ := (4 * 10^150 + 4 * 10^152) / (3 * 10^151 + 3 * 10^151)

-- Theorem stating that the expression equals 101/15
theorem expression_equals_101_15 : expression = 101 / 15 := by sorry

-- Function to find the closest integer to a rational number
def closest_integer (q : ℚ) : ℤ := 
  ⌊q + 1/2⌋

-- Theorem stating that the closest integer to the expression is 6
theorem closest_integer_is_6 : closest_integer expression = 6 := by sorry

end expression_equals_101_15_closest_integer_is_6_l2157_215796


namespace final_amount_in_euros_l2157_215766

/-- Represents the number of coins of each type -/
structure CoinCollection where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ
  half_dollars : ℕ
  one_dollar_coins : ℕ

/-- Calculates the total value of a coin collection in dollars -/
def collection_value (c : CoinCollection) : ℚ :=
  c.quarters * (1/4) + c.dimes * (1/10) + c.nickels * (1/20) + 
  c.pennies * (1/100) + c.half_dollars * (1/2) + c.one_dollar_coins

/-- Rob's initial coin collection -/
def initial_collection : CoinCollection := {
  quarters := 7,
  dimes := 3,
  nickels := 5,
  pennies := 12,
  half_dollars := 3,
  one_dollar_coins := 2
}

/-- Removes one coin of each type from the collection -/
def remove_one_each (c : CoinCollection) : CoinCollection := {
  quarters := c.quarters - 1,
  dimes := c.dimes - 1,
  nickels := c.nickels - 1,
  pennies := c.pennies - 1,
  half_dollars := c.half_dollars - 1,
  one_dollar_coins := c.one_dollar_coins - 1
}

/-- Exchanges three nickels for two dimes -/
def exchange_nickels_for_dimes (c : CoinCollection) : CoinCollection := {
  c with
  nickels := c.nickels - 3,
  dimes := c.dimes + 2
}

/-- Exchanges a half-dollar for a quarter and two dimes -/
def exchange_half_dollar (c : CoinCollection) : CoinCollection := {
  c with
  half_dollars := c.half_dollars - 1,
  quarters := c.quarters + 1,
  dimes := c.dimes + 2
}

/-- Exchanges a one-dollar coin for fifty pennies -/
def exchange_dollar_for_pennies (c : CoinCollection) : CoinCollection := {
  c with
  one_dollar_coins := c.one_dollar_coins - 1,
  pennies := c.pennies + 50
}

/-- Converts dollars to euros -/
def dollars_to_euros (dollars : ℚ) : ℚ :=
  dollars * (85/100)

/-- The main theorem stating the final amount in euros -/
theorem final_amount_in_euros : 
  dollars_to_euros (collection_value (
    exchange_dollar_for_pennies (
      exchange_half_dollar (
        exchange_nickels_for_dimes (
          remove_one_each initial_collection
        )
      )
    )
  )) = 2.9835 := by
  sorry


end final_amount_in_euros_l2157_215766


namespace betty_morning_flies_l2157_215726

/-- The number of flies Betty caught in the morning -/
def morning_flies : ℕ := 5

/-- The number of flies a frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies Betty caught in the afternoon -/
def afternoon_flies : ℕ := 6

/-- The number of flies that escaped -/
def escaped_flies : ℕ := 1

/-- The number of additional flies Betty needs -/
def additional_flies_needed : ℕ := 4

theorem betty_morning_flies :
  morning_flies = 5 :=
by
  sorry

end betty_morning_flies_l2157_215726


namespace box_volume_l2157_215773

theorem box_volume (sheet_length sheet_width cut_length : ℝ) 
  (h1 : sheet_length = 46)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 8) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 4800 := by
  sorry

end box_volume_l2157_215773


namespace soccer_team_combinations_l2157_215727

theorem soccer_team_combinations (n : ℕ) (k : ℕ) (h1 : n = 16) (h2 : k = 7) :
  Nat.choose n k = 11440 := by
  sorry

end soccer_team_combinations_l2157_215727


namespace quadratic_roots_range_l2157_215758

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   x₁ ≠ x₂ ∧
   x₁^2 + (m+2)*x₁ + m + 5 = 0 ∧
   x₂^2 + (m+2)*x₂ + m + 5 = 0) →
  -5 < m ∧ m ≤ -4 := by
sorry

end quadratic_roots_range_l2157_215758


namespace clock_time_sum_l2157_215703

/-- Represents time on a 12-hour digital clock -/
structure ClockTime where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

def addTime (start : ClockTime) (hours minutes seconds : Nat) : ClockTime :=
  let totalSeconds := start.hours * 3600 + start.minutes * 60 + start.seconds +
                      hours * 3600 + minutes * 60 + seconds
  let newSeconds := totalSeconds % 86400  -- 24 hours in seconds
  { hours := (newSeconds / 3600) % 12,
    minutes := (newSeconds % 3600) / 60,
    seconds := newSeconds % 60 }

def sumDigits (time : ClockTime) : Nat :=
  time.hours + time.minutes + time.seconds

theorem clock_time_sum (startTime : ClockTime) :
  let endTime := addTime startTime 189 58 52
  sumDigits endTime = 122 := by
  sorry

end clock_time_sum_l2157_215703


namespace additional_daily_intake_l2157_215708

/-- Proves that given a total milk consumption goal and a time frame, 
    the additional daily intake required can be calculated. -/
theorem additional_daily_intake 
  (total_milk : ℝ) 
  (weeks : ℝ) 
  (current_daily : ℝ) 
  (h1 : total_milk = 105) 
  (h2 : weeks = 3) 
  (h3 : current_daily = 3) : 
  ∃ (additional : ℝ), 
    additional = (total_milk / (weeks * 7)) - current_daily ∧ 
    additional = 2 := by
  sorry

end additional_daily_intake_l2157_215708


namespace smallest_multiple_with_100_divisors_l2157_215733

/-- The number of positive integral divisors of n -/
def divisor_count (n : ℕ) : ℕ := sorry

/-- n is a multiple of m -/
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_multiple_with_100_divisors :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n → ¬(is_multiple m 100 ∧ divisor_count m = 100)) ∧
    is_multiple n 100 ∧
    divisor_count n = 100 ∧
    n / 100 = 324 / 10 :=
sorry

end smallest_multiple_with_100_divisors_l2157_215733


namespace nested_fraction_evaluation_l2157_215746

theorem nested_fraction_evaluation :
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end nested_fraction_evaluation_l2157_215746


namespace shopkeeper_total_cards_l2157_215770

/-- The number of cards in a standard deck of playing cards -/
def standard_deck_size : ℕ := 52

/-- The number of complete decks the shopkeeper has -/
def complete_decks : ℕ := 6

/-- The number of additional cards the shopkeeper has -/
def additional_cards : ℕ := 7

/-- Theorem: The total number of cards the shopkeeper has is 319 -/
theorem shopkeeper_total_cards : 
  complete_decks * standard_deck_size + additional_cards = 319 := by
  sorry

end shopkeeper_total_cards_l2157_215770


namespace rolling_semicircle_distance_l2157_215795

/-- The distance traveled by the center of a rolling semi-circle -/
theorem rolling_semicircle_distance (r : ℝ) (h : r = 4 / Real.pi) :
  2 * Real.pi * r / 2 = 8 :=
by sorry

end rolling_semicircle_distance_l2157_215795


namespace anchuria_laws_theorem_l2157_215710

variables (K N M : ℕ) (p : ℝ)

/-- The probability that exactly M laws are included in the Concept -/
def prob_M_laws_included : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- The expected number of laws included in the Concept -/
def expected_laws_included : ℝ :=
  K * (1 - (1 - p)^N)

/-- Theorem stating the correctness of the probability and expectation calculations -/
theorem anchuria_laws_theorem (h1 : 0 ≤ p) (h2 : p ≤ 1) (h3 : M ≤ K) :
  (prob_M_laws_included K N M p = (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)) ∧
  (expected_laws_included K N p = K * (1 - (1 - p)^N)) := by
  sorry

end anchuria_laws_theorem_l2157_215710


namespace least_n_factorial_divisible_by_10080_l2157_215757

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem least_n_factorial_divisible_by_10080 :
  ∀ n : ℕ, n > 0 → (is_divisible (factorial n) 10080 → n ≥ 7) ∧
  (is_divisible (factorial 7) 10080) :=
sorry

end least_n_factorial_divisible_by_10080_l2157_215757


namespace inverse_sum_mod_11_l2157_215751

theorem inverse_sum_mod_11 : 
  (((2⁻¹ : ZMod 11) + (6⁻¹ : ZMod 11) + (10⁻¹ : ZMod 11))⁻¹ : ZMod 11) = 8 := by
  sorry

end inverse_sum_mod_11_l2157_215751


namespace inequalities_and_range_l2157_215709

theorem inequalities_and_range :
  (∀ x : ℝ, x > 1 → 2 * Real.log x < x - 1/x) ∧
  (∀ a : ℝ, a > 0 → (∀ t : ℝ, t > 0 → (1 + a/t) * Real.log (1 + t) > a) ↔ 0 < a ∧ a ≤ 2) ∧
  ((9/10 : ℝ)^19 < 1/Real.exp 2) :=
by sorry

end inequalities_and_range_l2157_215709


namespace hyperbola_properties_l2157_215721

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 6 = 1

-- Define the asymptote equation
def is_asymptote (m : ℝ) : Prop := ∀ x y : ℝ, hyperbola x y → (y = m * x ∨ y = -m * x)

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := ∃ a c : ℝ, a > 0 ∧ c > 0 ∧ e = c / a ∧ ∀ x y : ℝ, hyperbola x y → x^2 / a^2 - y^2 / (c^2 - a^2) = 1

-- Theorem statement
theorem hyperbola_properties : is_asymptote (Real.sqrt 2) ∧ eccentricity (Real.sqrt 3) :=
sorry

end hyperbola_properties_l2157_215721


namespace triangle_side_range_l2157_215718

/-- Given a triangle ABC with side lengths a, b, and c, prove that if |a+b-6|+(a-b+4)^2=0, then 4 < c < 6 -/
theorem triangle_side_range (a b c : ℝ) (h : |a+b-6|+(a-b+4)^2=0) : 4 < c ∧ c < 6 := by
  sorry

end triangle_side_range_l2157_215718


namespace joshes_investment_l2157_215767

/-- Proves that the initial investment is $2000 given the conditions of Josh's investment scenario -/
theorem joshes_investment
  (initial_wallet : ℝ)
  (final_wallet : ℝ)
  (stock_increase : ℝ)
  (h1 : initial_wallet = 300)
  (h2 : final_wallet = 2900)
  (h3 : stock_increase = 0.3)
  : ∃ (investment : ℝ), 
    investment = 2000 ∧ 
    final_wallet = initial_wallet + investment * (1 + stock_increase) :=
by sorry

end joshes_investment_l2157_215767


namespace walnut_distribution_l2157_215744

/-- The total number of walnuts -/
def total_walnuts : ℕ := 55

/-- The number of walnuts in the first pile -/
def first_pile : ℕ := 7

/-- The number of walnuts in each of the other piles -/
def other_piles : ℕ := 12

/-- The number of piles -/
def num_piles : ℕ := 5

theorem walnut_distribution :
  (num_piles - 1) * other_piles + first_pile = total_walnuts ∧
  ∃ (equal_walnuts : ℕ), equal_walnuts * num_piles = total_walnuts :=
by sorry

end walnut_distribution_l2157_215744


namespace f_k_even_iff_l2157_215724

/-- The number of valid coloring schemes for n points on a circle with at least one red point in any k consecutive points. -/
def f_k (k n : ℕ) : ℕ := sorry

/-- Theorem stating the necessary and sufficient conditions for f_k(n) to be even. -/
theorem f_k_even_iff (n k : ℕ) (h1 : n > k) (h2 : k ≥ 2) :
  Even (f_k k n) ↔ Even k ∧ (k + 1 ∣ n) := by sorry

end f_k_even_iff_l2157_215724


namespace volleyball_matches_l2157_215768

theorem volleyball_matches (a : ℕ) : 
  (3 / 5 : ℚ) * a = (11 / 20 : ℚ) * ((7 / 6 : ℚ) * a) → a = 24 := by
  sorry

end volleyball_matches_l2157_215768


namespace uniform_price_calculation_l2157_215716

/-- Represents the price of the uniform in Rupees -/
def uniform_price : ℝ := 200

/-- Represents the full year service pay in Rupees -/
def full_year_pay : ℝ := 800

/-- Represents the actual service duration in months -/
def actual_service : ℝ := 9

/-- Represents the full year service duration in months -/
def full_year : ℝ := 12

/-- Represents the actual payment received in Rupees -/
def actual_payment : ℝ := 400

theorem uniform_price_calculation :
  uniform_price = full_year_pay * (actual_service / full_year) - actual_payment :=
by sorry

end uniform_price_calculation_l2157_215716


namespace symmetric_function_properties_l2157_215753

def f (x m : ℝ) : ℝ := 2 * |x| + |2*x - m|

theorem symmetric_function_properties (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x : ℝ, f x m = f (2 - x) m) :
  (m = 4) ∧ 
  (∀ x : ℝ, f x m ≥ 4) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = m → 1/a + 4/b ≥ 9/4) := by
  sorry

end symmetric_function_properties_l2157_215753


namespace product_of_eight_consecutive_odd_numbers_divisible_by_ten_l2157_215761

theorem product_of_eight_consecutive_odd_numbers_divisible_by_ten (n : ℕ) (h : Odd n) :
  ∃ k : ℕ, (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) * (n + 14)) = 10 * k :=
by
  sorry

#check product_of_eight_consecutive_odd_numbers_divisible_by_ten

end product_of_eight_consecutive_odd_numbers_divisible_by_ten_l2157_215761


namespace polynomial_remainder_l2157_215781

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^3 - 18 * x^2 + 24 * x - 26) % (4 * x - 8) = 14 := by
  sorry

end polynomial_remainder_l2157_215781


namespace polynomial_coefficient_sum_l2157_215725

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 2 := by
sorry

end polynomial_coefficient_sum_l2157_215725


namespace line_parallel_perpendicular_to_plane_l2157_215742

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_parallel_perpendicular_to_plane 
  (a b : Line) (α : Plane) :
  parallel a b → perpendicular b α → perpendicular a α :=
sorry

end line_parallel_perpendicular_to_plane_l2157_215742


namespace system_solution_and_range_l2157_215788

theorem system_solution_and_range (a x y : ℝ) : 
  (2 * x + y = 5 * a ∧ x - 3 * y = -a + 7) →
  (x = 2 * a + 1 ∧ y = a - 2) ∧
  (x ≥ 0 ∧ y < 0 ↔ -1/2 ≤ a ∧ a < 2) :=
by sorry

end system_solution_and_range_l2157_215788
