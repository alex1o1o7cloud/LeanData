import Mathlib

namespace remainder_problem_l3389_338969

theorem remainder_problem (n : ℤ) (h : n % 7 = 2) : (5 * n + 9) % 7 = 5 := by
  sorry

end remainder_problem_l3389_338969


namespace max_value_of_f_l3389_338972

-- Define the function we want to maximize
def f (x : ℤ) : ℝ := 5 - |6 * x - 80|

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℤ), f x ≤ m :=
sorry

end max_value_of_f_l3389_338972


namespace moon_permutations_l3389_338976

def word_length : ℕ := 4
def repeated_letter_count : ℕ := 2

theorem moon_permutations :
  (word_length.factorial) / (repeated_letter_count.factorial) = 12 := by
  sorry

end moon_permutations_l3389_338976


namespace negation_of_existence_squared_greater_than_power_of_two_l3389_338906

theorem negation_of_existence_squared_greater_than_power_of_two :
  (¬ ∃ (n : ℕ+), n.val ^ 2 > 2 ^ n.val) ↔ (∀ (n : ℕ+), n.val ^ 2 ≤ 2 ^ n.val) := by
  sorry

end negation_of_existence_squared_greater_than_power_of_two_l3389_338906


namespace equation_equivalence_l3389_338918

theorem equation_equivalence (x y : ℝ) : 2 * x - y = 3 ↔ y = 2 * x - 3 := by sorry

end equation_equivalence_l3389_338918


namespace equation_solution_l3389_338916

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) :=
  sorry

end equation_solution_l3389_338916


namespace perpendicular_iff_a_eq_neg_five_or_one_l3389_338973

def line1 (a : ℝ) (x y : ℝ) : Prop := (2*a + 1)*x + (a + 5)*y - 6 = 0

def line2 (a : ℝ) (x y : ℝ) : Prop := (a + 5)*x + (a - 4)*y + 1 = 0

def perpendicular (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, line1 a x1 y1 ∧ line2 a x2 y2 →
    (2*a + 1)*(a + 5) + (a + 5)*(a - 4) = 0

theorem perpendicular_iff_a_eq_neg_five_or_one :
  ∀ a : ℝ, perpendicular a ↔ a = -5 ∨ a = 1 := by sorry

end perpendicular_iff_a_eq_neg_five_or_one_l3389_338973


namespace largest_B_for_divisibility_by_4_l3389_338964

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_single_digit (n : ℕ) : Prop := n ≤ 9

def seven_digit_number (B X : ℕ) : ℕ := 4000000 + 100000 * B + 6000 + 792 * 10 + X

theorem largest_B_for_divisibility_by_4 :
  ∃ (B : ℕ), is_single_digit B ∧
  (∃ (X : ℕ), is_single_digit X ∧ is_divisible_by_4 (seven_digit_number B X)) ∧
  ∀ (B' : ℕ), is_single_digit B' →
    (∃ (X : ℕ), is_single_digit X ∧ is_divisible_by_4 (seven_digit_number B' X)) →
    B' ≤ B :=
by sorry

end largest_B_for_divisibility_by_4_l3389_338964


namespace least_prime_factor_of_5_5_minus_5_3_l3389_338962

theorem least_prime_factor_of_5_5_minus_5_3 :
  Nat.minFac (5^5 - 5^3) = 2 := by
  sorry

end least_prime_factor_of_5_5_minus_5_3_l3389_338962


namespace triangle_inequality_l3389_338915

theorem triangle_inequality (a b x : ℝ) : 
  (a = 3 ∧ b = 5) → (2 < x ∧ x < 8) → 
  (a + b > x ∧ b + x > a ∧ x + a > b) := by
  sorry

end triangle_inequality_l3389_338915


namespace f_max_value_l3389_338900

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n : ℚ) / ((n + 32 : ℕ) * S (n + 1))

theorem f_max_value :
  (∀ n : ℕ, f n ≤ 1 / 50) ∧ (∃ n : ℕ, f n = 1 / 50) := by sorry

end f_max_value_l3389_338900


namespace intersection_complement_equality_l3389_338989

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {3, 7, 9}
def B : Set ℕ := {1, 9}

theorem intersection_complement_equality : A ∩ (U \ B) = {3, 7} := by
  sorry

end intersection_complement_equality_l3389_338989


namespace complex_number_location_l3389_338904

theorem complex_number_location : 
  let i : ℂ := Complex.I
  (1 + i) / (1 - i) + i ^ 2012 = 1 + i := by sorry

end complex_number_location_l3389_338904


namespace max_defective_items_l3389_338941

theorem max_defective_items 
  (N M n : ℕ) 
  (h1 : M ≤ N) 
  (h2 : n ≤ N) : 
  ∃ X : ℕ, X ≤ min M n ∧ 
  ∀ Y : ℕ, Y ≤ M ∧ Y ≤ n → Y ≤ X :=
sorry

end max_defective_items_l3389_338941


namespace kyle_earnings_theorem_l3389_338984

/-- Calculates the money Kyle makes from selling his remaining baked goods --/
def kyle_earnings (initial_cookies : ℕ) (initial_brownies : ℕ) 
                  (kyle_cookies_eaten : ℕ) (kyle_brownies_eaten : ℕ)
                  (mom_cookies_eaten : ℕ) (mom_brownies_eaten : ℕ)
                  (cookie_price : ℚ) (brownie_price : ℚ) : ℚ :=
  let remaining_cookies := initial_cookies - (kyle_cookies_eaten + mom_cookies_eaten)
  let remaining_brownies := initial_brownies - (kyle_brownies_eaten + mom_brownies_eaten)
  (remaining_cookies : ℚ) * cookie_price + (remaining_brownies : ℚ) * brownie_price

/-- Theorem stating Kyle's earnings from selling all remaining baked goods --/
theorem kyle_earnings_theorem : 
  kyle_earnings 60 32 2 2 1 2 1 (3/2) = 99 := by
  sorry

end kyle_earnings_theorem_l3389_338984


namespace carpet_area_calculation_l3389_338913

theorem carpet_area_calculation (room_length room_width wardrobe_side feet_per_yard : ℝ) 
  (h1 : room_length = 18)
  (h2 : room_width = 12)
  (h3 : wardrobe_side = 3)
  (h4 : feet_per_yard = 3) : 
  (room_length * room_width - wardrobe_side * wardrobe_side) / (feet_per_yard * feet_per_yard) = 23 := by
  sorry

end carpet_area_calculation_l3389_338913


namespace linear_equation_exponent_l3389_338987

/-- If x^(m+1) - 2 = 1 is a linear equation with respect to x, then m = 0 -/
theorem linear_equation_exponent (m : ℕ) : 
  (∀ x, ∃ a b : ℝ, x^(m+1) - 2 = a*x + b) → m = 0 := by
  sorry

end linear_equation_exponent_l3389_338987


namespace right_triangle_circumscribed_circle_perimeter_l3389_338968

theorem right_triangle_circumscribed_circle_perimeter 
  (r : ℝ) (h : ℝ) (a b : ℝ) :
  r = 4 →
  h = 26 →
  a^2 + b^2 = h^2 →
  a * b = 4 * (a + b + h) →
  a + b + h = 60 :=
by sorry

end right_triangle_circumscribed_circle_perimeter_l3389_338968


namespace current_at_12_ohms_l3389_338957

/-- A battery with voltage 48V and current-resistance relationship I = 48 / R -/
structure Battery where
  voltage : ℝ
  current : ℝ → ℝ
  resistance : ℝ
  h_voltage : voltage = 48
  h_current : ∀ R, current R = 48 / R

/-- When resistance is 12Ω, the current is 4A -/
theorem current_at_12_ohms (b : Battery) (h : b.resistance = 12) : 
  b.current b.resistance = 4 := by
  sorry

end current_at_12_ohms_l3389_338957


namespace matrix_value_equation_l3389_338922

theorem matrix_value_equation (x : ℝ) : 
  (3 * x) * (4 * x) - 2 * (2 * x) = 6 ↔ x = -1/3 ∨ x = 3/2 := by
  sorry

end matrix_value_equation_l3389_338922


namespace privateer_overtakes_at_1730_l3389_338995

/-- Represents the chase between a privateer and a merchantman -/
structure ChaseScenario where
  initial_distance : ℝ
  initial_time : ℕ  -- represented in minutes since midnight
  initial_privateer_speed : ℝ
  initial_merchantman_speed : ℝ
  initial_chase_duration : ℝ
  new_speed_ratio : ℚ

/-- Calculates the time when the privateer overtakes the merchantman -/
def overtake_time (scenario : ChaseScenario) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem privateer_overtakes_at_1730 : 
  let scenario : ChaseScenario := {
    initial_distance := 10,
    initial_time := 11 * 60 + 45,  -- 11:45 a.m. in minutes
    initial_privateer_speed := 11,
    initial_merchantman_speed := 8,
    initial_chase_duration := 2,
    new_speed_ratio := 17 / 15
  }
  overtake_time scenario = 17 * 60 + 30  -- 5:30 p.m. in minutes
:= by sorry


end privateer_overtakes_at_1730_l3389_338995


namespace tan_half_angle_fourth_quadrant_l3389_338966

theorem tan_half_angle_fourth_quadrant (α : Real) :
  (α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) →  -- α is in the fourth quadrant
  (Real.sin α + Real.cos α = 1 / 5) →               -- given condition
  Real.tan (α / 2) = -1 / 3 := by                   -- conclusion to prove
sorry


end tan_half_angle_fourth_quadrant_l3389_338966


namespace non_negative_sequence_l3389_338956

theorem non_negative_sequence (a : Fin 100 → ℝ) 
  (h1 : ∀ i : Fin 98, a i - 2 * a (i + 1) + a (i + 2) ≤ 0)
  (h2 : a 0 = a 99)
  (h3 : a 0 ≥ 0) : 
  ∀ i : Fin 100, a i ≥ 0 := by
sorry

end non_negative_sequence_l3389_338956


namespace division_theorem_l3389_338980

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = 176 → 
  divisor = 19 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder → 
  quotient = 9 := by
sorry

end division_theorem_l3389_338980


namespace cats_problem_l3389_338955

/-- The number of cats owned by the certain person -/
def person_cats (melanie_cats : ℕ) (annie_cats : ℕ) : ℕ :=
  3 * annie_cats

theorem cats_problem (melanie_cats : ℕ) (annie_cats : ℕ) 
  (h1 : melanie_cats = 2 * annie_cats)
  (h2 : melanie_cats = 60) :
  person_cats melanie_cats annie_cats = 90 := by
  sorry

end cats_problem_l3389_338955


namespace min_value_bn_Sn_l3389_338935

def S (n : ℕ) : ℚ := n / (n + 1)

def b (n : ℕ) : ℤ := n - 8

theorem min_value_bn_Sn :
  ∃ (min : ℚ), min = -4 ∧
  ∀ (n : ℕ), n ≥ 1 → (b n : ℚ) * S n ≥ min :=
sorry

end min_value_bn_Sn_l3389_338935


namespace time_to_paint_one_room_l3389_338914

/-- Given a painting job with a total number of rooms, rooms already painted,
    and time to paint the remaining rooms, calculate the time to paint one room. -/
theorem time_to_paint_one_room
  (total_rooms : ℕ)
  (painted_rooms : ℕ)
  (time_for_remaining : ℕ)
  (h1 : total_rooms = 10)
  (h2 : painted_rooms = 8)
  (h3 : time_for_remaining = 16)
  (h4 : painted_rooms < total_rooms) :
  (time_for_remaining : ℚ) / (total_rooms - painted_rooms : ℚ) = 8 := by
  sorry

end time_to_paint_one_room_l3389_338914


namespace negative_fraction_range_l3389_338990

theorem negative_fraction_range (x : ℝ) : (-5 : ℝ) / (2 - x) < 0 → x < 2 := by
  sorry

end negative_fraction_range_l3389_338990


namespace cost_per_book_l3389_338953

theorem cost_per_book (total_books : ℕ) (total_spent : ℕ) (h1 : total_books = 14) (h2 : total_spent = 224) :
  total_spent / total_books = 16 := by
sorry

end cost_per_book_l3389_338953


namespace tan_sum_angle_l3389_338981

theorem tan_sum_angle (α β : Real) : 
  α ∈ Set.Ioo 0 (Real.pi / 2) →
  β ∈ Set.Ioo 0 (Real.pi / 2) →
  2 * Real.tan α = Real.sin (2 * β) / (Real.sin β + Real.sin β ^ 2) →
  Real.tan (2 * α + β + Real.pi / 3) = -Real.sqrt 3 / 3 := by
  sorry

end tan_sum_angle_l3389_338981


namespace prob_same_color_is_31_105_l3389_338932

def blue_marbles : ℕ := 4
def yellow_marbles : ℕ := 5
def black_marbles : ℕ := 6
def total_marbles : ℕ := blue_marbles + yellow_marbles + black_marbles

def prob_same_color : ℚ :=
  (blue_marbles * (blue_marbles - 1) + yellow_marbles * (yellow_marbles - 1) + black_marbles * (black_marbles - 1)) /
  (total_marbles * (total_marbles - 1))

theorem prob_same_color_is_31_105 : prob_same_color = 31 / 105 := by
  sorry

end prob_same_color_is_31_105_l3389_338932


namespace combined_years_is_75_l3389_338926

/-- The combined total of years taught by Virginia, Adrienne, and Dennis -/
def combinedYears (adrienneYears virginiaYears dennisYears : ℕ) : ℕ :=
  adrienneYears + virginiaYears + dennisYears

/-- Theorem stating the combined total of years taught is 75 -/
theorem combined_years_is_75 :
  ∀ (adrienneYears virginiaYears dennisYears : ℕ),
    virginiaYears = adrienneYears + 9 →
    virginiaYears = dennisYears - 9 →
    dennisYears = 34 →
    combinedYears adrienneYears virginiaYears dennisYears = 75 := by
  sorry


end combined_years_is_75_l3389_338926


namespace infinitely_many_solutions_l3389_338911

/-- The equation 3(5 + dx) = 15x + 15 has infinitely many solutions for x if and only if d = 5 -/
theorem infinitely_many_solutions (d : ℝ) : 
  (∀ x, 3 * (5 + d * x) = 15 * x + 15) ↔ d = 5 := by sorry

end infinitely_many_solutions_l3389_338911


namespace alpha_plus_beta_equals_negative_55_l3389_338947

theorem alpha_plus_beta_equals_negative_55 :
  ∀ α β : ℝ, 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 50*x + 621) / (x^2 + 75*x - 2016)) →
  α + β = -55 :=
by sorry

end alpha_plus_beta_equals_negative_55_l3389_338947


namespace mason_father_age_l3389_338963

/-- Mason's age -/
def mason_age : ℕ := 20

/-- Sydney's age -/
def sydney_age : ℕ := mason_age + 6

/-- Mason's father's age -/
def father_age : ℕ := sydney_age + 6

theorem mason_father_age : father_age = 26 := by
  sorry

end mason_father_age_l3389_338963


namespace no_4digit_square_abba_palindromes_l3389_338988

/-- A function that checks if a number is a 4-digit square --/
def is_4digit_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, m * m = n

/-- A function that checks if a number is a palindrome with two different middle digits (abba form) --/
def is_abba_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    a ≠ b ∧
    n = a * 1000 + b * 100 + b * 10 + a

/-- The main theorem stating that there are no 4-digit squares that are abba palindromes --/
theorem no_4digit_square_abba_palindromes :
  ¬ ∃ n : ℕ, is_4digit_square n ∧ is_abba_palindrome n :=
sorry

end no_4digit_square_abba_palindromes_l3389_338988


namespace hyperbola_properties_l3389_338928

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → (x, y) ∈ foci ∨ (x, y) ∉ foci) ∧
  (∃ a b c : ℝ, a^2 = 4 ∧ b^2 = 12 ∧ c^2 = a^2 + b^2 ∧ eccentricity = c / a) :=
sorry

end hyperbola_properties_l3389_338928


namespace inequality_and_equality_conditions_l3389_338978

theorem inequality_and_equality_conditions (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  (x * y - 10)^2 ≥ 64 ∧ 
  (∃ (a b : ℝ), (a * b - 10)^2 = 64 ∧ (a + 1) * (b + 2) = 8) ∧
  (∀ (a b : ℝ), (a * b - 10)^2 = 64 ∧ (a + 1) * (b + 2) = 8 → (a = 1 ∧ b = 2) ∨ (a = -3 ∧ b = -6)) :=
by sorry

end inequality_and_equality_conditions_l3389_338978


namespace circle_areas_sum_l3389_338923

/-- The sum of the areas of an infinite series of circles with radii following
    the geometric sequence 1/√(2^(n-1)) is equal to 2π. -/
theorem circle_areas_sum : 
  let radius (n : ℕ) := (1 : ℝ) / Real.sqrt (2 ^ (n - 1))
  let area (n : ℕ) := Real.pi * (radius n) ^ 2
  (∑' n, area n) = 2 * Real.pi := by
  sorry

end circle_areas_sum_l3389_338923


namespace fill_jug_completely_l3389_338949

/-- The capacity of the jug in milliliters -/
def jug_capacity : ℕ := 800

/-- The capacity of a small container in milliliters -/
def container_capacity : ℕ := 48

/-- The minimum number of small containers needed to fill the jug completely -/
def min_containers : ℕ := 17

theorem fill_jug_completely :
  min_containers = (jug_capacity + container_capacity - 1) / container_capacity ∧
  min_containers * container_capacity ≥ jug_capacity ∧
  (min_containers - 1) * container_capacity < jug_capacity := by
  sorry

end fill_jug_completely_l3389_338949


namespace solve_for_a_l3389_338910

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (a : ℝ) : Prop :=
  (a * i) / (1 - i) = -1 + i

-- Theorem statement
theorem solve_for_a : ∃ (a : ℝ), equation a ∧ a = 2 := by
  sorry

end solve_for_a_l3389_338910


namespace environmental_policy_support_percentage_l3389_338905

theorem environmental_policy_support_percentage : 
  let total_surveyed : ℕ := 150 + 850
  let men_surveyed : ℕ := 150
  let women_surveyed : ℕ := 850
  let men_support_percentage : ℚ := 70 / 100
  let women_support_percentage : ℚ := 75 / 100
  let men_supporters : ℚ := men_surveyed * men_support_percentage
  let women_supporters : ℚ := women_surveyed * women_support_percentage
  let total_supporters : ℚ := men_supporters + women_supporters
  let overall_support_percentage : ℚ := total_supporters / total_surveyed * 100
  overall_support_percentage = 743 / 10 := by sorry

end environmental_policy_support_percentage_l3389_338905


namespace new_person_weight_l3389_338967

theorem new_person_weight (initial_total : ℝ) (h1 : initial_total > 0) : 
  let initial_avg := initial_total / 5
  let new_avg := initial_avg + 4
  let new_total := new_avg * 5
  new_total - (initial_total - 50) = 70 := by
  sorry

end new_person_weight_l3389_338967


namespace star_def_star_diff_neg_star_special_case_l3389_338998

-- Define the ☆ operation
def star (a b : ℝ) : ℝ := 3 * a + b

-- Theorem 1: Definition of ☆ operation
theorem star_def (a b : ℝ) : star a b = 3 * a + b := by sorry

-- Theorem 2: If a < b, then a☆b - b☆a < 0
theorem star_diff_neg {a b : ℝ} (h : a < b) : star a b - star b a < 0 := by sorry

-- Theorem 3: If a☆(-2b) = 4, then [3(a-b)]☆(3a+b) = 16
theorem star_special_case {a b : ℝ} (h : star a (-2*b) = 4) : 
  star (3*(a-b)) (3*a+b) = 16 := by sorry

end star_def_star_diff_neg_star_special_case_l3389_338998


namespace whale_first_hour_consumption_l3389_338921

/-- Represents the whale's plankton consumption pattern --/
structure WhaleConsumption where
  duration : Nat
  hourlyIncrease : Nat
  totalConsumption : Nat
  sixthHourConsumption : Nat

/-- Calculates the first hour's consumption given the whale's consumption pattern --/
def firstHourConsumption (w : WhaleConsumption) : Nat :=
  w.sixthHourConsumption - (w.hourlyIncrease * 5)

/-- Theorem stating that for the given whale consumption pattern, 
    the first hour's consumption is 38 kilos --/
theorem whale_first_hour_consumption 
  (w : WhaleConsumption) 
  (h1 : w.duration = 9)
  (h2 : w.hourlyIncrease = 3)
  (h3 : w.totalConsumption = 450)
  (h4 : w.sixthHourConsumption = 53) : 
  firstHourConsumption w = 38 := by
  sorry

#eval firstHourConsumption ⟨9, 3, 450, 53⟩

end whale_first_hour_consumption_l3389_338921


namespace problem_solution_l3389_338938

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Define the theorem
theorem problem_solution :
  -- Part 1
  (∀ x ∈ Set.Icc 1 3, g 1 x ∈ Set.Icc 0 4) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, g a x ∈ Set.Icc 0 4) → a = 1) ∧
  
  -- Part 2
  (∀ k : ℝ, (∀ x : ℝ, x ≥ 1 → g 1 (2^x) - k * 4^x ≥ 0) ↔ k ≤ 1/4) ∧
  
  -- Part 3
  (∀ k : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (g 1 (|2^x₁ - 1|) / |2^x₁ - 1| + k * (2 / |2^x₁ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₂ - 1|) / |2^x₂ - 1| + k * (2 / |2^x₂ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₃ - 1|) / |2^x₃ - 1| + k * (2 / |2^x₃ - 1|) - 3*k = 0)) ↔
   k > 0) := by
  sorry

end problem_solution_l3389_338938


namespace discounted_soda_price_70_cans_l3389_338930

/-- Calculate the price of discounted soda cans -/
def discounted_soda_price (regular_price : ℚ) (discount_percent : ℚ) (num_cans : ℕ) : ℚ :=
  let discounted_price := regular_price * (1 - discount_percent)
  let full_cases := num_cans / 24
  let remaining_cans := num_cans % 24
  discounted_price * (↑full_cases * 24 + ↑remaining_cans)

/-- The price of 70 cans of soda with a regular price of $0.55 and 25% discount in 24-can cases is $28.875 -/
theorem discounted_soda_price_70_cans :
  discounted_soda_price (55/100) (25/100) 70 = 28875/1000 :=
sorry

end discounted_soda_price_70_cans_l3389_338930


namespace two_correct_relations_l3389_338909

theorem two_correct_relations : 
  (0 ∈ ({0} : Set ℕ)) ∧ 
  ((∅ : Set ℕ) ⊆ {0}) ∧ 
  ¬({0, 1} ⊆ ({(0, 1)} : Set (ℕ × ℕ))) ∧ 
  ∀ a b : ℕ, {(a, b)} ≠ ({(b, a)} : Set (ℕ × ℕ)) := by
  sorry

end two_correct_relations_l3389_338909


namespace largest_multiple_under_500_l3389_338982

theorem largest_multiple_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by sorry

end largest_multiple_under_500_l3389_338982


namespace hundredth_digit_is_two_l3389_338979

/-- The decimal representation of 7/26 has a repeating cycle of 9 digits -/
def decimal_cycle : Fin 9 → Nat
| 0 => 2
| 1 => 6
| 2 => 9
| 3 => 2
| 4 => 3
| 5 => 0
| 6 => 7
| 7 => 6
| 8 => 9

/-- The 100th digit after the decimal point in the decimal representation of 7/26 -/
def hundredth_digit : Nat :=
  decimal_cycle (100 % 9)

theorem hundredth_digit_is_two : hundredth_digit = 2 := by
  sorry

end hundredth_digit_is_two_l3389_338979


namespace max_value_of_equation_l3389_338927

theorem max_value_of_equation (x : ℝ) : 
  (x^2 - x - 30) / (x - 5) = 2 / (x + 6) → x ≤ Real.sqrt 38 :=
by
  sorry

end max_value_of_equation_l3389_338927


namespace jack_afternoon_emails_l3389_338975

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 4

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- The total number of emails Jack received in the afternoon and evening -/
def afternoon_evening_emails : ℕ := 13

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := afternoon_evening_emails - evening_emails

theorem jack_afternoon_emails :
  afternoon_emails = 5 := by sorry

end jack_afternoon_emails_l3389_338975


namespace men_complete_nine_units_l3389_338954

/-- The number of men in the committee -/
def num_men : ℕ := 250

/-- The number of women in the committee -/
def num_women : ℕ := 150

/-- The number of units completed per day when all men and women work -/
def total_units : ℕ := 12

/-- The number of units completed per day when only women work -/
def women_units : ℕ := 3

/-- The number of units completed per day by men -/
def men_units : ℕ := total_units - women_units

theorem men_complete_nine_units : men_units = 9 := by
  sorry

end men_complete_nine_units_l3389_338954


namespace leadership_selection_l3389_338997

/-- The number of ways to choose a president, vice-president, and committee from a group --/
def choose_leadership (total : ℕ) (committee_size : ℕ) : ℕ :=
  total * (total - 1) * (Nat.choose (total - 2) committee_size)

/-- The problem statement --/
theorem leadership_selection :
  choose_leadership 10 3 = 5040 := by
  sorry

end leadership_selection_l3389_338997


namespace water_fraction_in_first_container_l3389_338950

/-- Represents the amount of liquid in a container -/
structure Container where
  juice : ℚ
  water : ℚ

/-- The problem setup and operations -/
def liquidTransfer : Prop :=
  ∃ (initial1 initial2 after_first_transfer after_second_transfer final1 : Container),
    -- Initial setup
    initial1.juice = 5 ∧ initial1.water = 0 ∧
    initial2.juice = 0 ∧ initial2.water = 5 ∧
    
    -- First transfer (1/3 of juice from container 1 to 2)
    after_first_transfer.juice = initial1.juice - (initial1.juice / 3) ∧
    after_first_transfer.water = initial1.water ∧
    
    -- Second transfer (1/4 of mixture from container 2 back to 1)
    final1.juice = after_first_transfer.juice + 
      ((initial2.water + (initial1.juice / 3)) / 4) * ((initial1.juice / 3) / (initial2.water + (initial1.juice / 3))) ∧
    final1.water = ((initial2.water + (initial1.juice / 3)) / 4) * (initial2.water / (initial2.water + (initial1.juice / 3))) ∧
    
    -- Final result
    final1.water / (final1.juice + final1.water) = 3 / 13

/-- The main theorem to prove -/
theorem water_fraction_in_first_container : liquidTransfer := by
  sorry

end water_fraction_in_first_container_l3389_338950


namespace largest_digit_divisible_by_six_l3389_338961

theorem largest_digit_divisible_by_six :
  ∃ (M : ℕ), M < 10 ∧ 
  (∀ (n : ℕ), n < 10 → 6 ∣ (3190 * 10 + n) → n ≤ M) ∧
  (6 ∣ (3190 * 10 + M)) :=
by sorry

end largest_digit_divisible_by_six_l3389_338961


namespace red_ball_packs_l3389_338933

theorem red_ball_packs (total_balls : ℕ) (yellow_packs green_packs : ℕ) (balls_per_pack : ℕ) :
  total_balls = 399 →
  yellow_packs = 10 →
  green_packs = 8 →
  balls_per_pack = 19 →
  ∃ red_packs : ℕ, red_packs = 3 ∧ 
    total_balls = (red_packs + yellow_packs + green_packs) * balls_per_pack :=
by sorry

end red_ball_packs_l3389_338933


namespace sampling_inspection_correct_for_yeast_l3389_338970

/-- Represents a biological experimental technique --/
inductive BiologicalTechnique
| YeastCounting
| SoilAnimalRichness
| OnionRootMitosis
| FatIdentification

/-- Represents a method used in biological experiments --/
inductive ExperimentalMethod
| SamplingInspection
| MarkRecapture
| RinsingForDye
| HydrochloricAcidWashing

/-- Function that returns the correct method for a given technique --/
def correct_method (technique : BiologicalTechnique) : ExperimentalMethod :=
  match technique with
  | BiologicalTechnique.YeastCounting => ExperimentalMethod.SamplingInspection
  | _ => ExperimentalMethod.SamplingInspection  -- Placeholder for other techniques

/-- Theorem stating that the sampling inspection method is correct for yeast counting --/
theorem sampling_inspection_correct_for_yeast :
  correct_method BiologicalTechnique.YeastCounting = ExperimentalMethod.SamplingInspection :=
by sorry

end sampling_inspection_correct_for_yeast_l3389_338970


namespace inequality_condition_l3389_338994

theorem inequality_condition (p : ℝ) :
  (∀ x₁ x₂ x₃ : ℝ, x₁^2 + x₂^2 + x₃^2 ≥ p * (x₁ * x₂ + x₂ * x₃)) ↔ p ≤ Real.sqrt 2 :=
by sorry

end inequality_condition_l3389_338994


namespace chocolate_difference_l3389_338946

theorem chocolate_difference (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 12) 
  (h2 : nickel_chocolates = 3) : 
  robert_chocolates - nickel_chocolates = 9 := by
sorry

end chocolate_difference_l3389_338946


namespace no_prime_root_sum_29_l3389_338931

/-- A quadratic equation x^2 - 29x + k = 0 with prime roots -/
def has_prime_roots (k : ℤ) : Prop :=
  ∃ p q : ℕ, 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    (p : ℤ) + (q : ℤ) = 29 ∧ 
    (p : ℤ) * (q : ℤ) = k

/-- There are no integer values of k such that x^2 - 29x + k = 0 has two prime roots -/
theorem no_prime_root_sum_29 : ¬∃ k : ℤ, has_prime_roots k := by
  sorry

end no_prime_root_sum_29_l3389_338931


namespace min_prime_divisor_of_quadratic_l3389_338929

theorem min_prime_divisor_of_quadratic : 
  ∀ p : ℕ, Prime p → (∃ n : ℕ, p ∣ (n^2 + 7*n + 23)) → p ≥ 11 :=
by sorry

end min_prime_divisor_of_quadratic_l3389_338929


namespace house_transactions_result_l3389_338917

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  hasHouse : Bool

/-- Represents a house transaction between two people -/
def houseTransaction (buyer seller : FinancialState) (price : Int) : FinancialState × FinancialState :=
  (FinancialState.mk (buyer.cash - price) true, FinancialState.mk (seller.cash + price) false)

/-- The main theorem to prove -/
theorem house_transactions_result :
  let initialA := FinancialState.mk 15000 true
  let initialB := FinancialState.mk 16000 false
  let (a1, b1) := houseTransaction initialB initialA 16000
  let (a2, b2) := houseTransaction a1 b1 14000
  let (a3, b3) := houseTransaction b2 a2 17000
  a3.cash = 34000 ∧ b3.cash = -3000 := by
  sorry

#check house_transactions_result

end house_transactions_result_l3389_338917


namespace lcm_inequality_l3389_338999

/-- For any two positive integers n and m where n > m, 
    the sum of the least common multiples of (m,n) and (m+1,n+1) 
    is greater than or equal to (2nm)/√(n-m). -/
theorem lcm_inequality (m n : ℕ) (h1 : 0 < m) (h2 : m < n) :
  Nat.lcm m n + Nat.lcm (m + 1) (n + 1) ≥ 
  (2 * n * m : ℝ) / Real.sqrt (n - m : ℝ) := by
  sorry

end lcm_inequality_l3389_338999


namespace major_premise_identification_l3389_338944

theorem major_premise_identification (α : ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, |Real.sin x| ≤ 1)
  (h2 : m = Real.sin α)
  (h3 : |m| ≤ 1) :
  (∀ x : ℝ, |Real.sin x| ≤ 1) = (|Real.sin x| ≤ 1) := by
sorry

end major_premise_identification_l3389_338944


namespace impossible_circle_assignment_l3389_338974

-- Define the type for circles
def Circle := Fin 6

-- Define the connection relation between circles
def connected : Circle → Circle → Prop := sorry

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

-- Main theorem
theorem impossible_circle_assignment :
  ¬ ∃ (f : Circle → ℕ),
    (∀ i j : Circle, connected i j → (divides (f i) (f j) ∨ divides (f j) (f i))) ∧
    (∀ i j : Circle, ¬ connected i j → ¬ divides (f i) (f j) ∧ ¬ divides (f j) (f i)) :=
by sorry


end impossible_circle_assignment_l3389_338974


namespace systematic_sampling_524_l3389_338960

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  populationSize : Nat
  samplingInterval : Nat

/-- Checks if the sampling interval divides the population size evenly -/
def SystematicSampling.isValidInterval (s : SystematicSampling) : Prop :=
  s.populationSize % s.samplingInterval = 0

theorem systematic_sampling_524 :
  ∃ (s : SystematicSampling), s.populationSize = 524 ∧ s.samplingInterval = 4 ∧ s.isValidInterval :=
sorry

end systematic_sampling_524_l3389_338960


namespace roberto_salary_l3389_338945

/-- Calculates the final salary after two consecutive percentage increases -/
def final_salary (starting_salary : ℝ) (first_increase : ℝ) (second_increase : ℝ) : ℝ :=
  starting_salary * (1 + first_increase) * (1 + second_increase)

/-- Theorem: Roberto's final salary calculation -/
theorem roberto_salary : 
  final_salary 80000 0.4 0.2 = 134400 := by
  sorry

#eval final_salary 80000 0.4 0.2

end roberto_salary_l3389_338945


namespace berkeley_b_count_l3389_338985

def abraham_total : ℕ := 20
def abraham_b : ℕ := 12
def berkeley_total : ℕ := 30

theorem berkeley_b_count : ℕ := by
  -- Define berkeley_b as the number of students in Mrs. Berkeley's class who received a 'B'
  -- Prove that berkeley_b = 18 given the conditions
  sorry

#check berkeley_b_count

end berkeley_b_count_l3389_338985


namespace inequality_proof_l3389_338993

theorem inequality_proof (a b c d e p q : ℝ) 
  (hp_pos : 0 < p) 
  (hq_geq_p : p ≤ q)
  (ha : p ≤ a ∧ a ≤ q)
  (hb : p ≤ b ∧ b ≤ q)
  (hc : p ≤ c ∧ c ≤ q)
  (hd : p ≤ d ∧ d ≤ q)
  (he : p ≤ e ∧ e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 :=
by sorry

end inequality_proof_l3389_338993


namespace fish_value_is_three_and_three_quarters_l3389_338907

/-- Represents the value of one fish in terms of bags of rice -/
def fish_value (fish_to_bread : ℚ) (bread_to_rice : ℚ) : ℚ :=
  (fish_to_bread * bread_to_rice)⁻¹

/-- Theorem stating that one fish is worth 3¾ bags of rice given the trade rates -/
theorem fish_value_is_three_and_three_quarters :
  fish_value (4/5) 3 = 15/4 := by
  sorry

end fish_value_is_three_and_three_quarters_l3389_338907


namespace bookstore_inventory_calculation_l3389_338991

/-- Represents the inventory of a bookstore -/
structure BookInventory where
  fiction : ℕ
  nonFiction : ℕ
  children : ℕ

/-- Represents the sales figures for a day -/
structure DailySales where
  inStoreFiction : ℕ
  inStoreNonFiction : ℕ
  inStoreChildren : ℕ
  online : ℕ

/-- Calculate the total number of books in the inventory -/
def totalBooks (inventory : BookInventory) : ℕ :=
  inventory.fiction + inventory.nonFiction + inventory.children

/-- Calculate the total in-store sales -/
def totalInStoreSales (sales : DailySales) : ℕ :=
  sales.inStoreFiction + sales.inStoreNonFiction + sales.inStoreChildren

theorem bookstore_inventory_calculation 
  (initialInventory : BookInventory)
  (saturdaySales : DailySales)
  (sundayInStoreSalesMultiplier : ℕ)
  (sundayOnlineSalesIncrease : ℕ)
  (newShipment : ℕ)
  (h1 : totalBooks initialInventory = 743)
  (h2 : initialInventory.fiction = 520)
  (h3 : initialInventory.nonFiction = 123)
  (h4 : initialInventory.children = 100)
  (h5 : totalInStoreSales saturdaySales = 37)
  (h6 : saturdaySales.inStoreFiction = 15)
  (h7 : saturdaySales.inStoreNonFiction = 12)
  (h8 : saturdaySales.inStoreChildren = 10)
  (h9 : saturdaySales.online = 128)
  (h10 : sundayInStoreSalesMultiplier = 2)
  (h11 : sundayOnlineSalesIncrease = 34)
  (h12 : newShipment = 160)
  : totalBooks initialInventory - 
    (totalInStoreSales saturdaySales + saturdaySales.online) - 
    (sundayInStoreSalesMultiplier * totalInStoreSales saturdaySales + 
     saturdaySales.online + sundayOnlineSalesIncrease) + 
    newShipment = 502 := by
  sorry

end bookstore_inventory_calculation_l3389_338991


namespace system_solution_unique_l3389_338940

theorem system_solution_unique :
  ∃! (x y : ℝ), x + y = 15 ∧ x - y = 5 := by
  sorry

end system_solution_unique_l3389_338940


namespace smallest_gcd_yz_l3389_338912

theorem smallest_gcd_yz (x y z : ℕ+) 
  (hxy : Nat.gcd x.val y.val = 270)
  (hxz : Nat.gcd x.val z.val = 105) :
  ∃ (y' z' : ℕ+), 
    Nat.gcd y'.val z'.val = 15 ∧
    (∀ (y'' z'' : ℕ+), 
      Nat.gcd x.val y''.val = 270 → 
      Nat.gcd x.val z''.val = 105 → 
      Nat.gcd y''.val z''.val ≥ 15) :=
sorry

end smallest_gcd_yz_l3389_338912


namespace trigonometric_and_algebraic_identities_l3389_338986

theorem trigonometric_and_algebraic_identities :
  (2 * Real.sin (45 * π / 180) ^ 2 + Real.tan (60 * π / 180) * Real.tan (30 * π / 180) - Real.cos (60 * π / 180) = 3 / 2) ∧
  (Real.sqrt 12 - 2 * Real.cos (30 * π / 180) + (3 - Real.pi) ^ 0 + |1 - Real.sqrt 3| = 2 * Real.sqrt 3) := by
  sorry

end trigonometric_and_algebraic_identities_l3389_338986


namespace find_x_value_l3389_338952

theorem find_x_value (x : ℝ) (hx : x ≠ 0) 
  (h : x = (1/x) * (-x) + 3) : x = 2 := by
  sorry

end find_x_value_l3389_338952


namespace truck_capacity_l3389_338937

theorem truck_capacity (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 15.5) 
  (h2 : 5 * x + 6 * y = 35) : 
  3 * x + 5 * y = 24.5 := by
sorry

end truck_capacity_l3389_338937


namespace min_study_tools_l3389_338951

theorem min_study_tools (n : ℕ) : n^3 ≥ 366 ∧ (n-1)^3 < 366 → n = 8 := by
  sorry

end min_study_tools_l3389_338951


namespace sum_f_equals_1326_l3389_338943

/-- A lattice point is a point with integer coordinates -/
def is_lattice_point (p : ℤ × ℤ) : Prop := True

/-- f(n) is the number of lattice points on the segment from (0,0) to (n, n+3), excluding endpoints -/
def f (n : ℕ) : ℕ := 
  if n % 3 = 0 then 2 else 0

/-- The sum of f(n) from 1 to 1990 -/
def sum_f : ℕ := (Finset.range 1990).sum f

theorem sum_f_equals_1326 : sum_f = 1326 := by
  sorry

end sum_f_equals_1326_l3389_338943


namespace min_value_of_x2_plus_y2_l3389_338992

theorem min_value_of_x2_plus_y2 (x y : ℝ) : 
  (x - 1)^2 + y^2 = 16 → ∃ (m : ℝ), (∀ (a b : ℝ), (a - 1)^2 + b^2 = 16 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 9 := by
  sorry

end min_value_of_x2_plus_y2_l3389_338992


namespace coffee_stock_decaf_percentage_l3389_338902

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
theorem coffee_stock_decaf_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 70) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := (initial_stock * initial_decaf_percent / 100) +
                     (additional_stock * additional_decaf_percent / 100)
  total_decaf / total_stock * 100 = 30 := by
sorry

end coffee_stock_decaf_percentage_l3389_338902


namespace equation_positive_root_l3389_338924

theorem equation_positive_root (x m : ℝ) : 
  (2 / (x - 2) = 1 - m / (x - 2)) → 
  (x > 0) → 
  (m = -2) := by
sorry

end equation_positive_root_l3389_338924


namespace inverse_variation_problem_l3389_338971

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x = k / (y ^ 2)) 
  (h2 : 1 = k / (2 ^ 2)) (h3 : 0.1111111111111111 = k / (y ^ 2)) : y = 6 := by
  sorry

end inverse_variation_problem_l3389_338971


namespace weight_loss_challenge_l3389_338934

theorem weight_loss_challenge (initial_weight : ℝ) (x : ℝ) : 
  x > 0 →
  (initial_weight * (1 - x / 100 + 2 / 100)) / initial_weight = 1 - 11.26 / 100 →
  x = 13.26 :=
by sorry

end weight_loss_challenge_l3389_338934


namespace completing_square_transformation_l3389_338901

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
  sorry

end completing_square_transformation_l3389_338901


namespace smallest_number_of_eggs_l3389_338920

/-- Represents the number of eggs in a container --/
def EggsPerContainer := 12

/-- Represents the number of containers with fewer eggs --/
def FewerEggsContainers := 3

/-- Represents the number of eggs in containers with fewer eggs --/
def EggsInFewerEggsContainers := 10

/-- Calculates the total number of eggs given the number of containers --/
def totalEggs (numContainers : ℕ) : ℕ :=
  numContainers * EggsPerContainer - FewerEggsContainers * (EggsPerContainer - EggsInFewerEggsContainers)

theorem smallest_number_of_eggs :
  ∃ (n : ℕ), (n > 100 ∧ totalEggs n = 102 ∧ ∀ m, m > 100 → totalEggs m ≥ 102) := by
  sorry

end smallest_number_of_eggs_l3389_338920


namespace baseball_stats_l3389_338936

-- Define the total number of hits and the number of each type of hit
def total_hits : ℕ := 45
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 6

-- Define the number of singles
def singles : ℕ := total_hits - (home_runs + triples + doubles)

-- Define the percentage of singles
def singles_percentage : ℚ := (singles : ℚ) / (total_hits : ℚ) * 100

-- Theorem to prove
theorem baseball_stats :
  singles = 34 ∧ singles_percentage = 75.56 := by
  sorry

end baseball_stats_l3389_338936


namespace john_jane_difference_l3389_338948

/-- The width of the streets in Perfectville -/
def street_width : ℕ := 30

/-- The side length of a block in Perfectville -/
def block_side_length : ℕ := 500

/-- The side length of John's path -/
def john_path_side : ℕ := block_side_length + 2 * street_width

/-- The perimeter of Jane's path -/
def jane_perimeter : ℕ := 4 * block_side_length

/-- The perimeter of John's path -/
def john_perimeter : ℕ := 4 * john_path_side

theorem john_jane_difference :
  john_perimeter - jane_perimeter = 240 := by sorry

end john_jane_difference_l3389_338948


namespace units_digit_47_pow_25_l3389_338925

theorem units_digit_47_pow_25 : ∃ n : ℕ, 47^25 ≡ 7 [ZMOD 10] :=
sorry

end units_digit_47_pow_25_l3389_338925


namespace min_sum_xy_l3389_338919

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + x - y - 10 = 0) :
  x + y ≥ 6 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ + x₀ - y₀ - 10 = 0 ∧ x₀ + y₀ = 6 :=
by sorry

end min_sum_xy_l3389_338919


namespace cone_cylinder_volume_ratio_l3389_338965

/-- The ratio of the volume of a cone to the volume of a cylinder with specified dimensions -/
theorem cone_cylinder_volume_ratio :
  let cone_height : ℝ := 10
  let cylinder_height : ℝ := 30
  let radius : ℝ := 5
  let cone_volume := (1/3) * π * radius^2 * cone_height
  let cylinder_volume := π * radius^2 * cylinder_height
  cone_volume / cylinder_volume = 2/9 := by
sorry

end cone_cylinder_volume_ratio_l3389_338965


namespace runners_meeting_time_l3389_338942

/-- Represents a runner with their start time (in minutes after 7:00 AM) and lap duration -/
structure Runner where
  startTime : ℕ
  lapDuration : ℕ

/-- The earliest time (in minutes after 7:00 AM) when all runners meet at the starting point -/
def earliestMeetingTime (runners : List Runner) : ℕ :=
  sorry

/-- The problem statement -/
theorem runners_meeting_time :
  let kevin := Runner.mk 45 5
  let laura := Runner.mk 50 8
  let neil := Runner.mk 55 10
  let runners := [kevin, laura, neil]
  earliestMeetingTime runners = 95
  := by sorry

end runners_meeting_time_l3389_338942


namespace discount_calculation_l3389_338959

/-- Proves that a product with given cost and original prices, sold at a specific profit margin, 
    results in a particular discount percentage. -/
theorem discount_calculation (cost_price original_price : ℝ) 
  (profit_margin : ℝ) (discount_percentage : ℝ) : 
  cost_price = 200 → 
  original_price = 300 → 
  profit_margin = 0.05 →
  discount_percentage = 0.7 →
  (original_price * discount_percentage - cost_price) / cost_price = profit_margin :=
by
  sorry

#check discount_calculation

end discount_calculation_l3389_338959


namespace rectangle_diagonal_intersection_l3389_338977

/-- Given a rectangle with opposite vertices at (2, -3) and (14, 9),
    prove that its diagonals intersect at the point (8, 3). -/
theorem rectangle_diagonal_intersection :
  let a : ℝ × ℝ := (2, -3)
  let c : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((a.1 + c.1) / 2, (a.2 + c.2) / 2)
  midpoint = (8, 3) :=
by sorry

end rectangle_diagonal_intersection_l3389_338977


namespace gcd_of_powers_of_97_l3389_338983

theorem gcd_of_powers_of_97 : 
  Nat.Prime 97 → Nat.gcd (97^7 + 1) (97^7 + 97^3 + 1) = 1 := by
  sorry

end gcd_of_powers_of_97_l3389_338983


namespace problem_structure_surface_area_l3389_338903

/-- Represents the 3D structure composed of unit cubes -/
structure CubeStructure where
  base : Nat
  secondLayer : Nat
  column : Nat
  sideOne : Nat
  sideTwo : Nat

/-- Calculates the surface area of the given cube structure -/
def surfaceArea (s : CubeStructure) : Nat :=
  let frontBack := s.base + s.secondLayer + s.column + s.sideOne + s.sideTwo
  let top := (s.base - s.secondLayer) + s.secondLayer + 1 + s.sideOne + s.sideTwo
  let bottom := s.base
  2 * frontBack + top + bottom

/-- The specific cube structure described in the problem -/
def problemStructure : CubeStructure :=
  { base := 5
  , secondLayer := 3
  , column := 2
  , sideOne := 3
  , sideTwo := 2 }

/-- Theorem stating that the surface area of the problem structure is 62 -/
theorem problem_structure_surface_area :
  surfaceArea problemStructure = 62 := by sorry

end problem_structure_surface_area_l3389_338903


namespace sandras_sweets_l3389_338939

theorem sandras_sweets (saved : ℚ) (mother_gave : ℚ) (father_gave : ℚ)
  (candy_cost : ℚ) (jelly_bean_cost : ℚ) (candy_count : ℕ) (jelly_bean_count : ℕ)
  (remaining : ℚ) :
  saved = 10 →
  mother_gave = 4 →
  candy_cost = 1/2 →
  jelly_bean_cost = 1/5 →
  candy_count = 14 →
  jelly_bean_count = 20 →
  remaining = 11 →
  saved + mother_gave + father_gave = 
    candy_cost * candy_count + jelly_bean_cost * jelly_bean_count + remaining →
  father_gave / mother_gave = 2 := by
sorry

#eval (8 : ℚ) / (4 : ℚ) -- Expected output: 2

end sandras_sweets_l3389_338939


namespace unit_segments_bound_l3389_338908

/-- 
Given n distinct points in a plane, τ(n) represents the number of 
unit-length segments joining pairs of these points.
-/
def τ (n : ℕ) : ℕ := sorry

/-- 
Theorem: The number of unit-length segments joining pairs of n distinct 
points in a plane is at most n²/3.
-/
theorem unit_segments_bound (n : ℕ) : τ n ≤ n^2 / 3 := by
  sorry

end unit_segments_bound_l3389_338908


namespace factorization_equality_l3389_338996

theorem factorization_equality (x y : ℝ) : 
  (x + y)^2 + 4*(x - y)^2 - 4*(x^2 - y^2) = (x - 3*y)^2 := by sorry

end factorization_equality_l3389_338996


namespace paint_calculation_l3389_338958

/-- The amount of paint Joe uses given the initial amount and usage fractions -/
def paint_used (initial : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) : ℚ :=
  let first_week := first_week_fraction * initial
  let remaining := initial - first_week
  let second_week := second_week_fraction * remaining
  first_week + second_week

/-- Theorem stating that given 360 gallons of paint, if 2/3 is used in the first week
    and 1/5 of the remainder is used in the second week, the total amount of paint used is 264 gallons -/
theorem paint_calculation :
  paint_used 360 (2/3) (1/5) = 264 := by
  sorry

end paint_calculation_l3389_338958
