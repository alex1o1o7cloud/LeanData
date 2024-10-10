import Mathlib

namespace victors_marks_percentage_l3039_303914

/-- Given that Victor scored 240 marks out of a maximum of 300 marks,
    prove that the percentage of marks he got is 80%. -/
theorem victors_marks_percentage (marks_obtained : ℝ) (maximum_marks : ℝ) 
    (h1 : marks_obtained = 240)
    (h2 : maximum_marks = 300) :
    (marks_obtained / maximum_marks) * 100 = 80 := by
  sorry

end victors_marks_percentage_l3039_303914


namespace circle_equation_proof_l3039_303925

/-- The standard equation of a circle with center (h, k) and radius r is (x-h)^2 + (y-k)^2 = r^2 -/
def StandardCircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Prove that for a circle with center (2, -1) and radius 3, the standard equation is (x-2)^2 + (y+1)^2 = 9 -/
theorem circle_equation_proof :
  ∀ (x y : ℝ), StandardCircleEquation 2 (-1) 3 x y ↔ (x - 2)^2 + (y + 1)^2 = 9 := by
  sorry


end circle_equation_proof_l3039_303925


namespace total_games_played_l3039_303982

-- Define the parameters
def win_percentage : ℚ := 50 / 100
def games_won : ℕ := 70

-- State the theorem
theorem total_games_played : ℕ := by
  -- The proof goes here
  sorry

-- The goal to prove
#check total_games_played = 140

end total_games_played_l3039_303982


namespace april_cookie_spending_l3039_303992

/-- Calculates the total amount spent on cookies in April given the following conditions:
  * April has 30 days
  * On even days, 3 chocolate chip cookies and 2 sugar cookies are bought
  * On odd days, 4 oatmeal cookies and 1 snickerdoodle cookie are bought
  * Prices: chocolate chip $18, sugar $22, oatmeal $15, snickerdoodle $25
-/
theorem april_cookie_spending : 
  let days_in_april : ℕ := 30
  let even_days : ℕ := days_in_april / 2
  let odd_days : ℕ := days_in_april / 2
  let choc_chip_price : ℕ := 18
  let sugar_price : ℕ := 22
  let oatmeal_price : ℕ := 15
  let snickerdoodle_price : ℕ := 25
  let even_day_cost : ℕ := 3 * choc_chip_price + 2 * sugar_price
  let odd_day_cost : ℕ := 4 * oatmeal_price + 1 * snickerdoodle_price
  let total_cost : ℕ := even_days * even_day_cost + odd_days * odd_day_cost
  total_cost = 2745 := by
sorry


end april_cookie_spending_l3039_303992


namespace solve_equation_l3039_303922

theorem solve_equation : ∃ x : ℝ, (12 : ℝ) ^ x * 6^2 / 432 = 144 ∧ x = 3 := by
  sorry

end solve_equation_l3039_303922


namespace triangle_properties_l3039_303981

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.sin t.A = t.a * Real.sin (2 * t.B))
  (h2 : t.b = Real.sqrt 10)
  (h3 : t.a + t.c = t.a * t.c) :
  t.B = π / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = (5 * Real.sqrt 3) / 4 := by
  sorry

end triangle_properties_l3039_303981


namespace retailer_profit_percentage_l3039_303971

/-- Calculates the overall profit percentage for a retailer selling three items -/
theorem retailer_profit_percentage
  (radio_purchase : ℝ) (tv_purchase : ℝ) (speaker_purchase : ℝ)
  (radio_overhead : ℝ) (tv_overhead : ℝ) (speaker_overhead : ℝ)
  (radio_selling : ℝ) (tv_selling : ℝ) (speaker_selling : ℝ)
  (h1 : radio_purchase = 225)
  (h2 : tv_purchase = 4500)
  (h3 : speaker_purchase = 1500)
  (h4 : radio_overhead = 30)
  (h5 : tv_overhead = 200)
  (h6 : speaker_overhead = 100)
  (h7 : radio_selling = 300)
  (h8 : tv_selling = 5400)
  (h9 : speaker_selling = 1800) :
  let total_cp := radio_purchase + tv_purchase + speaker_purchase +
                  radio_overhead + tv_overhead + speaker_overhead
  let total_sp := radio_selling + tv_selling + speaker_selling
  let profit := total_sp - total_cp
  let profit_percentage := (profit / total_cp) * 100
  abs (profit_percentage - 14.42) < 0.01 := by
sorry


end retailer_profit_percentage_l3039_303971


namespace target_breaking_permutations_l3039_303957

theorem target_breaking_permutations :
  let total_targets : ℕ := 10
  let column_a_targets : ℕ := 4
  let column_b_targets : ℕ := 4
  let column_c_targets : ℕ := 2
  (column_a_targets + column_b_targets + column_c_targets = total_targets) →
  (Nat.factorial total_targets) / 
  (Nat.factorial column_a_targets * Nat.factorial column_b_targets * Nat.factorial column_c_targets) = 5040 := by
  sorry

end target_breaking_permutations_l3039_303957


namespace sine_function_period_l3039_303958

/-- 
Given a function y = A sin(Bx + C) + D, where A, B, C, and D are constants,
if the graph covers two periods in an interval of 4π, then B = 1.
-/
theorem sine_function_period (A B C D : ℝ) : 
  (∃ (a b : ℝ), b - a = 4 * π ∧ 
    (∀ x ∈ Set.Icc a b, ∃ k : ℤ, A * Real.sin (B * x + C) + D = A * Real.sin (B * (x + 2 * k * π / B) + C) + D)) →
  B = 1 := by
sorry

end sine_function_period_l3039_303958


namespace triangle_max_area_l3039_303975

theorem triangle_max_area (x y : ℝ) (h : x + y = 418) :
  ⌊(1/2 : ℝ) * x * y⌋ ≤ 21840 ∧ ∃ (x' y' : ℝ), x' + y' = 418 ∧ ⌊(1/2 : ℝ) * x' * y'⌋ = 21840 :=
sorry

end triangle_max_area_l3039_303975


namespace rent_share_ratio_l3039_303964

theorem rent_share_ratio (purity_share : ℚ) (rose_share : ℚ) (total_rent : ℚ) :
  rose_share = 1800 →
  total_rent = 5400 →
  total_rent = 5 * purity_share + purity_share + rose_share →
  rose_share / purity_share = 3 := by
  sorry

end rent_share_ratio_l3039_303964


namespace aluminum_carbonate_weight_l3039_303995

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The number of moles of Aluminum carbonate -/
def moles : ℝ := 5

/-- The molecular weight of Aluminum carbonate (Al2(CO3)3) in g/mol -/
def Al2CO3_3_weight : ℝ := 2 * Al_weight + 3 * C_weight + 9 * O_weight

/-- The total weight of the given moles of Aluminum carbonate in grams -/
def total_weight : ℝ := moles * Al2CO3_3_weight

theorem aluminum_carbonate_weight : total_weight = 1169.95 := by sorry

end aluminum_carbonate_weight_l3039_303995


namespace trig_equation_solution_range_l3039_303935

theorem trig_equation_solution_range :
  ∀ m : ℝ, 
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y + Real.sin y ^ 2 + m - 4 = 0) ↔ 
  (0 ≤ m ∧ m ≤ 8) :=
by sorry

end trig_equation_solution_range_l3039_303935


namespace sqrt_equation_solution_l3039_303993

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 10 → x = 105 := by
  sorry

end sqrt_equation_solution_l3039_303993


namespace largest_m_bound_l3039_303926

theorem largest_m_bound (x y z t : ℕ+) (h1 : x + y = z + t) (h2 : 2 * x * y = z * t) (h3 : x ≥ y) :
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ 
  (∀ (m' : ℝ), (∀ (x' y' z' t' : ℕ+), 
    x' + y' = z' + t' → 2 * x' * y' = z' * t' → x' ≥ y' → 
    (x' : ℝ) / (y' : ℝ) ≥ m') → m' ≤ m) := by
  sorry

end largest_m_bound_l3039_303926


namespace bicycle_price_increase_l3039_303946

theorem bicycle_price_increase (initial_price : ℝ) (first_increase : ℝ) (second_increase : ℝ) :
  initial_price = 220 →
  first_increase = 0.08 →
  second_increase = 0.10 →
  let price_after_first := initial_price * (1 + first_increase)
  let final_price := price_after_first * (1 + second_increase)
  final_price = 261.36 := by
sorry

end bicycle_price_increase_l3039_303946


namespace beta_still_water_speed_l3039_303966

/-- Represents a boat with its speed in still water -/
structure Boat where
  speed : ℝ

/-- Represents the river with its current speed -/
structure River where
  currentSpeed : ℝ

/-- Represents a journey on the river -/
inductive Direction
  | Upstream
  | Downstream

def effectiveSpeed (b : Boat) (r : River) (d : Direction) : ℝ :=
  match d with
  | Direction.Upstream => b.speed + r.currentSpeed
  | Direction.Downstream => b.speed - r.currentSpeed

theorem beta_still_water_speed 
  (alpha : Boat)
  (beta : Boat)
  (river : River)
  (h1 : alpha.speed = 56)
  (h2 : beta.speed = 52)
  (h3 : river.currentSpeed = 4)
  (h4 : effectiveSpeed alpha river Direction.Upstream / effectiveSpeed beta river Direction.Downstream = 5 / 4)
  (h5 : effectiveSpeed alpha river Direction.Downstream / effectiveSpeed beta river Direction.Upstream = 4 / 5) :
  beta.speed = 61 := by
  sorry

end beta_still_water_speed_l3039_303966


namespace base_prime_441_l3039_303903

/-- Base prime representation of a natural number --/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- 441 in base prime representation --/
theorem base_prime_441 : base_prime_repr 441 = [0, 2, 2, 0] := by
  sorry

end base_prime_441_l3039_303903


namespace partial_fraction_decomposition_l3039_303997

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
  (6 * x^2 + 3 * x) / ((x - 4) * (x - 2)^3) =
  13.5 / (x - 4) + (-27) / (x - 2) + (-15) / (x - 2)^3 := by
  sorry

end partial_fraction_decomposition_l3039_303997


namespace m_values_l3039_303932

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem m_values : 
  {m : ℝ | B m ⊆ A} = {1/3, -1/2} := by sorry

end m_values_l3039_303932


namespace little_john_money_distribution_l3039_303952

/-- Calculates the amount Little John gave to each of his two friends -/
def money_given_to_each_friend (initial_amount : ℚ) (spent_on_sweets : ℚ) (amount_left : ℚ) : ℚ :=
  (initial_amount - spent_on_sweets - amount_left) / 2

/-- Proves that Little John gave $2.20 to each of his two friends -/
theorem little_john_money_distribution :
  money_given_to_each_friend 10.50 2.25 3.85 = 2.20 := by
  sorry

#eval money_given_to_each_friend 10.50 2.25 3.85

end little_john_money_distribution_l3039_303952


namespace smallest_sum_of_reciprocals_l3039_303938

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 → (x + y : ℕ) ≤ (a + b : ℕ)) → 
  (x + y : ℕ) = 49 := by
sorry

end smallest_sum_of_reciprocals_l3039_303938


namespace share_calculation_l3039_303954

theorem share_calculation (total : ℚ) (a b c : ℚ) 
  (h1 : total = 510)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) : b = 90 := by
  sorry

end share_calculation_l3039_303954


namespace determinant_solution_l3039_303965

/-- Given a ≠ 0 and b ≠ 0, the solution to the determinant equation is (3b^2 + ab) / (a + b) -/
theorem determinant_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x := (3 * b^2 + a * b) / (a + b)
  (x + a) * ((x + b) * (2 * x) - x * (2 * x)) -
  x * (x * (2 * x) - x * (2 * x + a + b)) +
  x * (x * (2 * x) - (x + b) * (2 * x + a + b)) = 0 := by
  sorry

#check determinant_solution

end determinant_solution_l3039_303965


namespace geometric_sequence_condition_l3039_303949

/-- A sequence of three real numbers is geometric if the ratio between consecutive terms is constant. -/
def IsGeometricSequence (x y z : ℝ) : Prop :=
  y / x = z / y

/-- The statement that a = ±6 is equivalent to the sequence 4, a, 9 being geometric. -/
theorem geometric_sequence_condition :
  ∀ a : ℝ, IsGeometricSequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by sorry

end geometric_sequence_condition_l3039_303949


namespace books_left_l3039_303963

theorem books_left (initial_books given_away : ℝ) 
  (h1 : initial_books = 54.0)
  (h2 : given_away = 23.0) : 
  initial_books - given_away = 31.0 := by
sorry

end books_left_l3039_303963


namespace absolute_value_equality_implies_midpoint_l3039_303978

theorem absolute_value_equality_implies_midpoint (x : ℚ) :
  |x - 2| = |x - 5| → x = 7/2 := by
  sorry

end absolute_value_equality_implies_midpoint_l3039_303978


namespace ellipse_major_axis_length_l3039_303989

/-- The length of the major axis of an ellipse with given foci and tangent to y-axis -/
theorem ellipse_major_axis_length :
  let f1 : ℝ × ℝ := (15, 10)
  let f2 : ℝ × ℝ := (35, 40)
  let ellipse := {p : ℝ × ℝ | ∃ k, dist p f1 + dist p f2 = k}
  let tangent_to_y_axis := ∃ y, (0, y) ∈ ellipse
  let major_axis_length := Real.sqrt 3400
  tangent_to_y_axis →
  ∃ a b : ℝ × ℝ, a ∈ ellipse ∧ b ∈ ellipse ∧ dist a b = major_axis_length :=
by sorry


end ellipse_major_axis_length_l3039_303989


namespace oranges_per_box_l3039_303970

def total_oranges : ℕ := 45
def num_boxes : ℕ := 9

theorem oranges_per_box : total_oranges / num_boxes = 5 := by
  sorry

end oranges_per_box_l3039_303970


namespace certain_chinese_book_l3039_303909

def total_books : ℕ := 12
def chinese_books : ℕ := 10
def math_books : ℕ := 2
def drawn_books : ℕ := 3

theorem certain_chinese_book :
  ∀ (drawn : Finset ℕ),
    drawn.card = drawn_books →
    drawn ⊆ Finset.range total_books →
    ∃ (book : ℕ), book ∈ drawn ∧ book < chinese_books :=
sorry

end certain_chinese_book_l3039_303909


namespace total_shot_cost_l3039_303900

-- Define the given conditions
def pregnant_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def shots_per_puppy : ℕ := 2
def cost_per_shot : ℕ := 5

-- Define the theorem
theorem total_shot_cost : 
  pregnant_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = 120 := by
  sorry

end total_shot_cost_l3039_303900


namespace trapezoid_area_is_12_sqrt_5_l3039_303939

/-- Represents a trapezoid with given measurements -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculates the area of a trapezoid given its measurements -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that a trapezoid with the given measurements has an area of 12√5 -/
theorem trapezoid_area_is_12_sqrt_5 :
  ∀ t : Trapezoid,
    t.base1 = 3 ∧
    t.base2 = 6 ∧
    t.diagonal1 = 7 ∧
    t.diagonal2 = 8 →
    area t = 12 * Real.sqrt 5 := by
  sorry

end trapezoid_area_is_12_sqrt_5_l3039_303939


namespace surface_area_of_problem_lshape_l3039_303943

/-- Represents the L-shaped structure formed by unit cubes -/
structure LShape where
  base_length : Nat
  top_length : Nat
  top_start_position : Nat

/-- Calculates the surface area of the L-shaped structure -/
def surface_area (l : LShape) : Nat :=
  let base_visible_top := l.base_length - l.top_length
  let base_visible_sides := 2 * l.base_length
  let base_visible_ends := 2
  let top_visible_top := l.top_length
  let top_visible_sides := 2 * l.top_length
  let top_visible_ends := 2
  base_visible_top + base_visible_sides + base_visible_ends +
  top_visible_top + top_visible_sides + top_visible_ends

/-- The specific L-shaped structure described in the problem -/
def problem_lshape : LShape :=
  { base_length := 10
    top_length := 5
    top_start_position := 5 }

theorem surface_area_of_problem_lshape :
  surface_area problem_lshape = 45 := by
  sorry

end surface_area_of_problem_lshape_l3039_303943


namespace panda_weekly_consumption_l3039_303913

/-- The total bamboo consumption for a group of pandas in a week -/
def panda_bamboo_consumption 
  (small_pandas : ℕ) 
  (big_pandas : ℕ) 
  (small_daily_consumption : ℕ) 
  (big_daily_consumption : ℕ) : ℕ :=
  ((small_pandas * small_daily_consumption + big_pandas * big_daily_consumption) * 7)

/-- Theorem: The total bamboo consumption for 4 small pandas and 5 big pandas in a week is 2100 pounds -/
theorem panda_weekly_consumption : 
  panda_bamboo_consumption 4 5 25 40 = 2100 := by
  sorry

end panda_weekly_consumption_l3039_303913


namespace minimal_sequence_property_l3039_303906

def F_p (p : ℕ) : Set (ℕ → ℕ) :=
  {a | ∀ n > 0, a (n + 1) = (p + 1) * a n - p * a (n - 1) ∧ ∀ n, a n ≥ 0}

def minimal_sequence (p : ℕ) (n : ℕ) : ℕ :=
  (p^n - 1) / (p - 1)

theorem minimal_sequence_property (p : ℕ) (hp : p > 1) :
  minimal_sequence p ∈ F_p p ∧
  ∀ b ∈ F_p p, ∀ n, minimal_sequence p n ≤ b n :=
by sorry

end minimal_sequence_property_l3039_303906


namespace no_natural_solution_l3039_303910

theorem no_natural_solution :
  ¬∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 2 * m^2 := by
  sorry

end no_natural_solution_l3039_303910


namespace valid_param_iff_l3039_303919

/-- A parameterization of a line in 2D space -/
structure LineParam where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- The line equation y = 2x - 4 -/
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

/-- A parameterization is valid for the line y = 2x - 4 -/
def is_valid_param (p : LineParam) : Prop :=
  line_eq p.x₀ p.y₀ ∧ p.dy = 2 * p.dx

theorem valid_param_iff (p : LineParam) :
  is_valid_param p ↔ 
  (∀ t : ℝ, line_eq (p.x₀ + t * p.dx) (p.y₀ + t * p.dy)) :=
sorry

end valid_param_iff_l3039_303919


namespace unique_solution_3x_4y_5z_l3039_303904

theorem unique_solution_3x_4y_5z :
  ∀ (x y z : ℕ), 3^x + 4^y = 5^z → (x = 2 ∧ y = 2 ∧ z = 2) := by
  sorry

end unique_solution_3x_4y_5z_l3039_303904


namespace complex_exp_seven_pi_over_two_eq_i_l3039_303960

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem complex_exp_seven_pi_over_two_eq_i :
  cexp (Complex.I * (7 * Real.pi / 2)) = Complex.I :=
by sorry

end complex_exp_seven_pi_over_two_eq_i_l3039_303960


namespace jimmys_cabin_friends_l3039_303920

def hostel_stay_days : ℕ := 3
def hostel_cost_per_night : ℕ := 15
def cabin_stay_days : ℕ := 2
def cabin_cost_per_night : ℕ := 45
def total_lodging_cost : ℕ := 75

theorem jimmys_cabin_friends :
  ∃ (n : ℕ), 
    hostel_stay_days * hostel_cost_per_night + 
    cabin_stay_days * (cabin_cost_per_night / (n + 1)) = total_lodging_cost ∧
    n = 2 := by
  sorry

end jimmys_cabin_friends_l3039_303920


namespace bird_count_difference_l3039_303984

theorem bird_count_difference (monday_count tuesday_count wednesday_count : ℕ) : 
  monday_count = 70 →
  tuesday_count = monday_count / 2 →
  monday_count + tuesday_count + wednesday_count = 148 →
  wednesday_count - tuesday_count = 8 := by
sorry

end bird_count_difference_l3039_303984


namespace sequence_constraint_l3039_303983

/-- An arithmetic sequence of four real numbers -/
structure ArithmeticSequence (x a₁ a₂ y : ℝ) : Prop where
  diff₁ : a₁ - x = a₂ - a₁
  diff₂ : a₂ - a₁ = y - a₂

/-- A geometric sequence of four real numbers -/
structure GeometricSequence (x b₁ b₂ y : ℝ) : Prop where
  ratio₁ : x ≠ 0
  ratio₂ : b₁ / x = b₂ / b₁
  ratio₃ : b₂ / b₁ = y / b₂

theorem sequence_constraint (x a₁ a₂ y b₁ b₂ : ℝ) 
  (h₁ : ArithmeticSequence x a₁ a₂ y) (h₂ : GeometricSequence x b₁ b₂ y) : 
  x ≥ 4 := by sorry

end sequence_constraint_l3039_303983


namespace davids_math_marks_l3039_303955

/-- Represents the marks obtained in each subject -/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average marks -/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology) / 5

/-- Theorem: Given David's marks in other subjects and his average, his Mathematics marks must be 65 -/
theorem davids_math_marks (m : Marks) : 
  m.english = 51 → 
  m.physics = 82 → 
  m.chemistry = 67 → 
  m.biology = 85 → 
  average m = 70 → 
  m.mathematics = 65 := by
  sorry

#eval average { english := 51, mathematics := 65, physics := 82, chemistry := 67, biology := 85 }

end davids_math_marks_l3039_303955


namespace simplify_expression_l3039_303998

theorem simplify_expression (b : ℚ) (h : b = 2) :
  (15 * b^4 - 45 * b^3) / (75 * b^2) = -2/5 := by
  sorry

end simplify_expression_l3039_303998


namespace stratified_sampling_young_employees_l3039_303931

theorem stratified_sampling_young_employees 
  (total_employees : ℕ) 
  (young_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 200) 
  (h2 : young_employees = 120) 
  (h3 : sample_size = 25) :
  ↑sample_size * (↑young_employees / ↑total_employees) = 15 :=
by sorry

end stratified_sampling_young_employees_l3039_303931


namespace speed_ratio_is_two_sevenths_l3039_303990

/-- Two objects A and B moving uniformly along perpendicular paths -/
structure MovingObjects where
  vA : ℝ  -- Speed of object A
  vB : ℝ  -- Speed of object B

/-- The conditions of the problem -/
def satisfies_conditions (obj : MovingObjects) : Prop :=
  ∃ (t₁ t₂ : ℝ),
    t₁ > 0 ∧ t₂ > t₁ ∧
    (obj.vA * t₁)^2 = (750 - obj.vB * t₁)^2 ∧
    (obj.vA * t₂)^2 = (750 - obj.vB * t₂)^2 ∧
    t₂ - t₁ = 6 ∧
    t₁ = 3

/-- The theorem statement -/
theorem speed_ratio_is_two_sevenths (obj : MovingObjects) 
  (h : satisfies_conditions obj) : 
  obj.vA / obj.vB = 2 / 7 := by
  sorry

end speed_ratio_is_two_sevenths_l3039_303990


namespace min_sum_squares_l3039_303921

/-- A line intercepted by a circle with a given chord length -/
structure LineCircleIntersection where
  /-- Coefficient of x in the line equation -/
  a : ℝ
  /-- Coefficient of y in the line equation -/
  b : ℝ
  /-- The line equation: ax + 2by - 4 = 0 -/
  line_eq : ∀ (x y : ℝ), a * x + 2 * b * y - 4 = 0
  /-- The circle equation: x^2 + y^2 + 4x - 2y + 1 = 0 -/
  circle_eq : ∀ (x y : ℝ), x^2 + y^2 + 4*x - 2*y + 1 = 0
  /-- The chord length of the intersection is 4 -/
  chord_length : ℝ
  chord_length_eq : chord_length = 4

/-- The minimum value of a^2 + b^2 for a LineCircleIntersection is 2 -/
theorem min_sum_squares (lci : LineCircleIntersection) : 
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 + b^2 ≥ m) ∧ m = 2 :=
sorry

end min_sum_squares_l3039_303921


namespace susan_walk_distance_l3039_303944

/-- Given two people walking together for a total of 15 miles, where one person walks 3 miles less
    than the other, prove that the person who walked more covered 9 miles. -/
theorem susan_walk_distance (susan_distance erin_distance : ℝ) :
  susan_distance + erin_distance = 15 →
  erin_distance = susan_distance - 3 →
  susan_distance = 9 := by
sorry

end susan_walk_distance_l3039_303944


namespace order_of_ab_squared_a_ab_l3039_303927

theorem order_of_ab_squared_a_ab (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) : 
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end order_of_ab_squared_a_ab_l3039_303927


namespace inequality_equivalence_l3039_303945

theorem inequality_equivalence (b c : ℝ) : 
  (∀ x : ℝ, |2*x - 3| < 5 ↔ -x^2 + b*x + c > 0) → b + c = 7 := by
  sorry

end inequality_equivalence_l3039_303945


namespace merchant_profit_l3039_303962

theorem merchant_profit (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
  markup_percentage = 0.40 →
  discount_percentage = 0.10 →
  let marked_price := cost_price * (1 + markup_percentage)
  let selling_price := marked_price * (1 - discount_percentage)
  let profit := selling_price - cost_price
  let profit_percentage := profit / cost_price
  profit_percentage = 0.26 := by
  sorry

end merchant_profit_l3039_303962


namespace solve_linear_equation_l3039_303996

theorem solve_linear_equation :
  ∃ x : ℚ, -3 * x - 10 = 4 * x + 5 ∧ x = -15 / 7 := by
  sorry

end solve_linear_equation_l3039_303996


namespace remainder_x2023_plus_1_l3039_303953

theorem remainder_x2023_plus_1 (x : ℂ) : 
  (x^2023 + 1) % (x^6 - x^4 + x^2 - 1) = -x^3 + 1 := by sorry

end remainder_x2023_plus_1_l3039_303953


namespace students_play_both_calculation_l3039_303947

/-- Represents the number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

theorem students_play_both_calculation :
  students_play_both 450 325 175 50 = 100 := by
  sorry

end students_play_both_calculation_l3039_303947


namespace no_real_roots_l3039_303907

theorem no_real_roots : ¬ ∃ x : ℝ, x^2 - x + 2 = 0 := by sorry

end no_real_roots_l3039_303907


namespace apple_tree_difference_l3039_303994

theorem apple_tree_difference (ava_trees lily_trees total_trees : ℕ) : 
  ava_trees = 9 → 
  total_trees = 15 → 
  lily_trees = total_trees - ava_trees → 
  ava_trees - lily_trees = 3 := by
sorry

end apple_tree_difference_l3039_303994


namespace martian_calendar_months_l3039_303918

/-- Represents the number of days in a Martian month -/
inductive MartianMonth
  | long : MartianMonth  -- 100 days
  | short : MartianMonth -- 77 days

/-- Calculates the number of days in a Martian month -/
def daysInMonth (m : MartianMonth) : Nat :=
  match m with
  | MartianMonth.long => 100
  | MartianMonth.short => 77

/-- Represents a Martian calendar year -/
structure MartianYear where
  months : List MartianMonth
  total_days : Nat
  total_days_eq : total_days = List.sum (months.map daysInMonth)

/-- The theorem to be proved -/
theorem martian_calendar_months (year : MartianYear) 
    (h : year.total_days = 5882) : year.months.length = 74 := by
  sorry

#check martian_calendar_months

end martian_calendar_months_l3039_303918


namespace binomial_coefficient_ratio_l3039_303917

theorem binomial_coefficient_ratio (n : ℕ) : 4^n / 2^n = 64 → n = 6 := by
  sorry

end binomial_coefficient_ratio_l3039_303917


namespace cherry_pie_degrees_l3039_303940

/-- Calculates the number of degrees for cherry pie in a pie chart given the class preferences. -/
theorem cherry_pie_degrees (total_students : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ)
  (h1 : total_students = 48)
  (h2 : chocolate = 15)
  (h3 : apple = 10)
  (h4 : blueberry = 9)
  (h5 : (total_students - (chocolate + apple + blueberry)) % 2 = 0) :
  (((total_students - (chocolate + apple + blueberry)) / 2 : ℚ) / total_students) * 360 = 52.5 := by
  sorry

#eval ((7 : ℚ) / 48) * 360  -- Should output 52.5

end cherry_pie_degrees_l3039_303940


namespace method2_more_profitable_above_15000_l3039_303985

/-- Profit calculation for Method 1 (end of month) -/
def profit_method1 (x : ℝ) : ℝ := 0.3 * x - 900

/-- Profit calculation for Method 2 (beginning of month with reinvestment) -/
def profit_method2 (x : ℝ) : ℝ := 0.26 * x

/-- Theorem stating that Method 2 is more profitable when x > 15000 -/
theorem method2_more_profitable_above_15000 (x : ℝ) (h : x > 15000) :
  profit_method2 x > profit_method1 x :=
sorry

end method2_more_profitable_above_15000_l3039_303985


namespace rectangle_side_ratio_l3039_303950

theorem rectangle_side_ratio (a b c d : ℝ) (h : a / c = b / d ∧ a / c = 4 / 5) : b / d = 4 / 5 := by
  sorry

end rectangle_side_ratio_l3039_303950


namespace parabola_properties_l3039_303968

/-- Parabola properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- Points on the parabola -/
structure ParabolaPoints (p : Parabola) where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h₁ : y₁ = p.a * x₁^2 + p.b * x₁
  h₂ : y₂ = p.a * x₂^2 + p.b * x₂
  h₃ : x₁ < x₂
  h₄ : x₁ + x₂ = 2

/-- Theorem about parabola properties -/
theorem parabola_properties (p : Parabola) (pts : ParabolaPoints p)
  (h₁ : p.a * 3^2 + p.b * 3 = 3) :
  (p.b = 1 - 3 * p.a) ∧
  (pts.y₁ = pts.y₂ → p.a = 1) ∧
  (pts.y₁ < pts.y₂ → 0 < p.a ∧ p.a < 1) := by
  sorry

end parabola_properties_l3039_303968


namespace vector_sum_magnitude_l3039_303911

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) :
  Real.sqrt (((a.1 + 2 * b.1) ^ 2) + ((a.2 + 2 * b.2) ^ 2)) = 2 * Real.sqrt 3 := by
  sorry

end vector_sum_magnitude_l3039_303911


namespace max_sum_at_five_l3039_303924

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The theorem stating when S_n reaches its maximum value -/
theorem max_sum_at_five (seq : ArithmeticSequence)
    (h1 : seq.a 1 + seq.a 5 + seq.a 9 = 6)
    (h2 : seq.S 11 = -11) :
    ∃ n_max : ℕ, n_max = 5 ∧ ∀ n : ℕ, seq.S n ≤ seq.S n_max := by
  sorry

end max_sum_at_five_l3039_303924


namespace employee_age_when_hired_l3039_303908

/-- Rule of 70 provision for retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1986

/-- The year the employee became eligible for retirement -/
def retirement_year : ℕ := 2006

/-- The age of the employee when hired -/
def age_when_hired : ℕ := 50

theorem employee_age_when_hired :
  rule_of_70 age_when_hired (retirement_year - hire_year) ∧
  age_when_hired = 70 - (retirement_year - hire_year) := by
  sorry

end employee_age_when_hired_l3039_303908


namespace initial_distance_correct_l3039_303959

/-- The initial distance between Seonghyeon and Jisoo -/
def initial_distance : ℝ := 1200

/-- The distance Seonghyeon ran towards Jisoo -/
def distance_towards : ℝ := 200

/-- The distance Seonghyeon ran in the opposite direction -/
def distance_away : ℝ := 1000

/-- The final distance between Seonghyeon and Jisoo -/
def final_distance : ℝ := 2000

/-- Theorem stating that the initial distance is correct given the conditions -/
theorem initial_distance_correct : 
  initial_distance - distance_towards + distance_away = final_distance :=
by sorry

end initial_distance_correct_l3039_303959


namespace steves_coins_l3039_303930

theorem steves_coins (total_coins : ℕ) (nickel_value dime_value : ℕ) (swap_increase : ℕ) :
  total_coins = 30 →
  nickel_value = 5 →
  dime_value = 10 →
  swap_increase = 120 →
  ∃ (nickels dimes : ℕ),
    nickels + dimes = total_coins ∧
    dimes * dime_value + nickels * nickel_value + swap_increase = nickels * dime_value + dimes * nickel_value →
    dimes * dime_value + nickels * nickel_value = 165 :=
by sorry

end steves_coins_l3039_303930


namespace weight_distribution_problem_l3039_303979

theorem weight_distribution_problem :
  ∃! (a b c : ℕ), a + b + c = 100 ∧ a + 10 * b + 50 * c = 500 ∧ (a, b, c) = (60, 39, 1) := by
  sorry

end weight_distribution_problem_l3039_303979


namespace complement_intersection_problem_l3039_303902

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_intersection_problem : (Mᶜ ∩ N) = {-3, -4} := by
  sorry

end complement_intersection_problem_l3039_303902


namespace orthogonality_iff_k_eq_4_l3039_303969

/-- Two unit vectors with an angle of 60° between them -/
structure UnitVectorPair :=
  (e₁ e₂ : ℝ × ℝ)
  (unit_e₁ : e₁.1 ^ 2 + e₁.2 ^ 2 = 1)
  (unit_e₂ : e₂.1 ^ 2 + e₂.2 ^ 2 = 1)
  (angle_60 : e₁.1 * e₂.1 + e₁.2 * e₂.2 = 1/2)

/-- The orthogonality condition -/
def orthogonality (v : UnitVectorPair) (k : ℝ) : Prop :=
  (2 * v.e₁.1 - k * v.e₂.1) * v.e₁.1 + (2 * v.e₁.2 - k * v.e₂.2) * v.e₁.2 = 0

/-- The main theorem -/
theorem orthogonality_iff_k_eq_4 (v : UnitVectorPair) :
  orthogonality v 4 ∧ (∀ k : ℝ, orthogonality v k → k = 4) :=
sorry

end orthogonality_iff_k_eq_4_l3039_303969


namespace tangent_line_to_circle_l3039_303980

/-- A line is tangent to a circle if and only if the distance from the center of the circle to the line equals the radius of the circle. -/
axiom line_tangent_to_circle_iff_distance_eq_radius {a b c : ℝ} {x₀ y₀ r : ℝ} :
  (∀ x y, (x - x₀)^2 + (y - y₀)^2 = r^2 → a*x + b*y + c ≠ 0) ↔
  |a*x₀ + b*y₀ + c| / Real.sqrt (a^2 + b^2) = r

/-- The theorem to be proved -/
theorem tangent_line_to_circle (m : ℝ) :
  m > 0 →
  (∀ x y, (x - 3)^2 + (y - 4)^2 = 4 → 3*x - 4*y - m ≠ 0) →
  m = 3 := by
  sorry

end tangent_line_to_circle_l3039_303980


namespace smallest_prime_divisor_of_sum_l3039_303905

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^15 + 11^21) ∧ ∀ q, Nat.Prime q → q ∣ (3^15 + 11^21) → p ≤ q :=
by sorry

end smallest_prime_divisor_of_sum_l3039_303905


namespace isosceles_triangle_midpoint_property_l3039_303988

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the properties of the isosceles triangle
def IsIsosceles (t : Triangle) (a : ℝ) : Prop :=
  dist t.X t.Y = a ∧ dist t.Y t.Z = a

-- Define point M on XZ
def PointOnXZ (t : Triangle) (M : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ M = (1 - k) • t.X + k • t.Z

-- Define the midpoint property of M
def IsMidpoint (t : Triangle) (M : ℝ × ℝ) : Prop :=
  dist t.Y M = dist M t.Z

-- Define the sum of distances property
def SumOfDistances (t : Triangle) (M : ℝ × ℝ) (a : ℝ) : Prop :=
  dist t.X M + dist M t.Z = 2 * a

-- Main theorem
theorem isosceles_triangle_midpoint_property
  (t : Triangle) (M : ℝ × ℝ) (a : ℝ) :
  IsIsosceles t a →
  PointOnXZ t M →
  IsMidpoint t M →
  SumOfDistances t M a →
  dist t.Y M = a / 2 :=
sorry

end isosceles_triangle_midpoint_property_l3039_303988


namespace arithmetic_sequence_a8_l3039_303923

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 = 2 → a 4 = 6 → a 8 = 14 := by
sorry

end arithmetic_sequence_a8_l3039_303923


namespace binomial_12_9_l3039_303936

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end binomial_12_9_l3039_303936


namespace hemisphere_cylinder_surface_area_l3039_303934

/-- The total surface area of a hemisphere (excluding its base) and cylinder side surface -/
theorem hemisphere_cylinder_surface_area (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  2 * π * r * h + 2 * π * r^2 = 150 * π := by sorry

end hemisphere_cylinder_surface_area_l3039_303934


namespace squares_in_50th_ring_l3039_303974

/-- The number of squares in the nth ring of a square pattern -/
def squares_in_ring (n : ℕ) : ℕ := 4 * n + 4

/-- The number of squares in the 50th ring is 204 -/
theorem squares_in_50th_ring : squares_in_ring 50 = 204 := by
  sorry

end squares_in_50th_ring_l3039_303974


namespace hyperbola_equation_l3039_303991

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Existence of the hyperbola
  (∃ x : ℝ, x = 2 ∧ x^2 / a^2 - 0^2 / b^2 = 1) →  -- Right vertex at (2, 0)
  (∃ c : ℝ, c / a = 3/2) →  -- Eccentricity is 3/2
  (∃ x y : ℝ, y^2 = 8*x) →  -- Existence of the parabola
  (∀ x y : ℝ, x^2 / 4 - y^2 / 5 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end hyperbola_equation_l3039_303991


namespace median_name_length_l3039_303972

/-- Represents the distribution of name lengths -/
structure NameLengthDistribution where
  three_letter : Nat
  four_letter : Nat
  five_letter : Nat
  six_letter : Nat
  seven_letter : Nat

/-- The median of a list of natural numbers -/
def median (l : List Nat) : Nat :=
  sorry

/-- Generates a list of name lengths based on the distribution -/
def generateNameLengths (d : NameLengthDistribution) : List Nat :=
  sorry

theorem median_name_length (d : NameLengthDistribution) :
  d.three_letter = 6 →
  d.four_letter = 5 →
  d.five_letter = 2 →
  d.six_letter = 4 →
  d.seven_letter = 4 →
  d.three_letter + d.four_letter + d.five_letter + d.six_letter + d.seven_letter = 21 →
  median (generateNameLengths d) = 4 := by
  sorry

end median_name_length_l3039_303972


namespace jerry_hawk_feathers_l3039_303986

/-- The number of hawk feathers Jerry found -/
def hawk_feathers : ℕ := 6

/-- The number of eagle feathers Jerry found -/
def eagle_feathers : ℕ := 17 * hawk_feathers

/-- The total number of feathers Jerry initially had -/
def total_feathers : ℕ := hawk_feathers + eagle_feathers

/-- The number of feathers Jerry had after giving 10 to his sister -/
def feathers_after_giving : ℕ := total_feathers - 10

/-- The number of feathers Jerry had after selling half of the remaining feathers -/
def feathers_after_selling : ℕ := feathers_after_giving / 2

theorem jerry_hawk_feathers :
  hawk_feathers = 6 ∧
  eagle_feathers = 17 * hawk_feathers ∧
  total_feathers = hawk_feathers + eagle_feathers ∧
  feathers_after_giving = total_feathers - 10 ∧
  feathers_after_selling = feathers_after_giving / 2 ∧
  feathers_after_selling = 49 :=
by sorry

end jerry_hawk_feathers_l3039_303986


namespace initial_manager_percentage_l3039_303933

/-- The initial percentage of managers in a room with 200 employees,
    given that 99.99999999999991 managers leave and the resulting
    percentage is 98%, is approximately 99%. -/
theorem initial_manager_percentage :
  let total_employees : ℕ := 200
  let managers_who_left : ℝ := 99.99999999999991
  let final_percentage : ℝ := 98
  let initial_percentage : ℝ := 
    ((managers_who_left + (final_percentage / 100) * (total_employees - managers_who_left)) / total_employees) * 100
  ∀ ε > 0, |initial_percentage - 99| < ε :=
by sorry


end initial_manager_percentage_l3039_303933


namespace min_a_value_l3039_303967

-- Define the function representing the left side of the inequality
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|

-- State the theorem
theorem min_a_value (a : ℝ) : 
  (∃ x : ℝ, f x a ≤ 8) → 
  (∀ b : ℝ, (∃ x : ℝ, f x b ≤ 8) → a ≤ b) → 
  a = -9 := by
sorry

end min_a_value_l3039_303967


namespace no_valid_schedule_l3039_303961

theorem no_valid_schedule : ¬∃ (a b : ℕ+), (29 ∣ a) ∧ (32 ∣ b) ∧ (a + b = 29 * 32) := by
  sorry

end no_valid_schedule_l3039_303961


namespace pencil_rows_l3039_303929

/-- Given a total number of pencils and the number of pencils per row,
    calculate the number of complete rows that can be formed. -/
def calculate_rows (total_pencils : ℕ) (pencils_per_row : ℕ) : ℕ :=
  total_pencils / pencils_per_row

/-- Theorem stating that 30 pencils arranged in rows of 5 will form 6 complete rows -/
theorem pencil_rows : calculate_rows 30 5 = 6 := by
  sorry

end pencil_rows_l3039_303929


namespace units_digit_of_powers_l3039_303973

theorem units_digit_of_powers : 
  (31^2020 % 10 = 1) ∧ (37^2020 % 10 = 1) := by
  sorry

end units_digit_of_powers_l3039_303973


namespace dragons_games_count_l3039_303948

theorem dragons_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (60 * initial_games) / 100 →
    ∃ (total_games : ℕ),
      total_games = initial_games + 11 ∧
      (initial_wins + 8 : ℚ) / total_games = 55 / 100 ∧
      total_games = 50 :=
by sorry

end dragons_games_count_l3039_303948


namespace salt_solution_dilution_l3039_303937

def initial_solution_volume : ℝ := 40
def initial_salt_concentration : ℝ := 0.15
def target_salt_concentration : ℝ := 0.10
def water_added : ℝ := 20

theorem salt_solution_dilution :
  let initial_salt_amount : ℝ := initial_solution_volume * initial_salt_concentration
  let final_solution_volume : ℝ := initial_solution_volume + water_added
  let final_salt_concentration : ℝ := initial_salt_amount / final_solution_volume
  final_salt_concentration = target_salt_concentration := by sorry

end salt_solution_dilution_l3039_303937


namespace fourteen_binary_l3039_303928

/-- The binary representation of a natural number -/
def binary_repr (n : ℕ) : List Bool :=
  sorry

theorem fourteen_binary : binary_repr 14 = [true, true, true, false] := by
  sorry

end fourteen_binary_l3039_303928


namespace value_of_a_minus_b_l3039_303987

theorem value_of_a_minus_b (a b : ℤ) 
  (eq1 : 3015 * a + 3019 * b = 3023) 
  (eq2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := by
sorry

end value_of_a_minus_b_l3039_303987


namespace problem_statement_l3039_303912

theorem problem_statement (a : ℝ) (h : a^2 + a - 1 = 0) :
  2 * a^2 + 2 * a + 2008 = 2010 := by
  sorry

end problem_statement_l3039_303912


namespace number_from_percentages_l3039_303976

theorem number_from_percentages (x : ℝ) : 
  0.15 * 0.30 * 0.50 * x = 126 → x = 5600 := by
  sorry

end number_from_percentages_l3039_303976


namespace range_of_a_l3039_303901

theorem range_of_a (a : ℝ) (h_a_pos : a > 0) : 
  (∀ m : ℝ, (3 * a < m ∧ m < 4 * a) → (1 < m ∧ m < 3/2)) →
  (1/3 ≤ a ∧ a ≤ 3/8) :=
by sorry

end range_of_a_l3039_303901


namespace area_ratio_octagon_quadrilateral_l3039_303941

/-- Regular octagon with vertices ABCDEFGH -/
structure RegularOctagon where
  area : ℝ

/-- Quadrilateral ACEG within the regular octagon -/
structure Quadrilateral where
  area : ℝ

/-- Theorem stating that the ratio of the quadrilateral area to the octagon area is √2/2 -/
theorem area_ratio_octagon_quadrilateral (octagon : RegularOctagon) (quad : Quadrilateral) :
  quad.area / octagon.area = Real.sqrt 2 / 2 := by
  sorry

end area_ratio_octagon_quadrilateral_l3039_303941


namespace water_amount_is_150_l3039_303956

/-- Represents the ratios of bleach, detergent, and water in a solution --/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- The original ratio of the solution --/
def original_ratio : SolutionRatio := ⟨4, 40, 100⟩

/-- The altered ratio after tripling bleach to detergent and halving detergent to water --/
def altered_ratio : SolutionRatio :=
  let b := original_ratio.bleach * 3
  let d := original_ratio.detergent
  let w := original_ratio.water / 2
  ⟨b, d, w⟩

/-- The amount of detergent in the altered solution --/
def altered_detergent_amount : ℚ := 60

/-- Calculates the amount of water in the altered solution --/
def water_amount : ℚ :=
  altered_detergent_amount * (altered_ratio.water / altered_ratio.detergent)

/-- Theorem stating that the amount of water in the altered solution is 150 liters --/
theorem water_amount_is_150 : water_amount = 150 := by sorry

end water_amount_is_150_l3039_303956


namespace win_sector_area_l3039_303977

/-- Given a circular spinner with radius 15 cm and a probability of winning of 1/3,
    the area of the WIN sector is 75π square centimeters. -/
theorem win_sector_area (radius : ℝ) (win_prob : ℝ) (win_area : ℝ) : 
  radius = 15 → 
  win_prob = 1/3 → 
  win_area = win_prob * π * radius^2 →
  win_area = 75 * π := by
sorry

end win_sector_area_l3039_303977


namespace infinitely_many_n_with_bounded_prime_divisors_l3039_303915

theorem infinitely_many_n_with_bounded_prime_divisors :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ 
  (∀ n ∈ S, ∀ p : ℕ, Prime p → p ∣ (n^2 + n + 1) → p ≤ Real.sqrt n) :=
sorry

end infinitely_many_n_with_bounded_prime_divisors_l3039_303915


namespace exists_m_for_second_quadrant_l3039_303942

/-- A point is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P -/
def x_coord (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point P -/
def y_coord : ℝ := 3

theorem exists_m_for_second_quadrant : 
  ∃ m : ℝ, m < 1 ∧ is_in_second_quadrant (x_coord m) y_coord :=
sorry

end exists_m_for_second_quadrant_l3039_303942


namespace tom_bob_sticker_ratio_l3039_303999

def bob_stickers : ℕ := 12

theorem tom_bob_sticker_ratio :
  ∃ (tom_stickers : ℕ),
    tom_stickers = bob_stickers ∧
    tom_stickers / bob_stickers = 1 := by
  sorry

end tom_bob_sticker_ratio_l3039_303999


namespace bookseller_sales_l3039_303951

/-- Bookseller's monthly sales problem -/
theorem bookseller_sales 
  (b1 b2 b3 b4 : ℕ) 
  (h1 : b1 + b2 + b3 = 45)
  (h2 : b4 = (3 * (b1 + b2)) / 4)
  (h3 : (b1 + b2 + b3 + b4) / 4 = 18) :
  b3 = 9 ∧ b1 + b2 = 36 ∧ b4 = 27 := by
  sorry

end bookseller_sales_l3039_303951


namespace part1_part2_l3039_303916

-- Define the conditions p and q as functions of x and m
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0

def q (x m : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part 1
theorem part1 (m : ℝ) (h : m > 0) :
  (∀ x, p x → q x m) → m ≥ 4 := by sorry

-- Part 2
theorem part2 (x : ℝ) :
  (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) →
  (x ∈ Set.Icc (-3) (-2) ∪ Set.Ioc 6 7) := by sorry

end part1_part2_l3039_303916
