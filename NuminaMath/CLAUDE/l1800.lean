import Mathlib

namespace car_worth_calculation_l1800_180033

/-- Brendan's earnings and expenses in June -/
structure BrendanFinances where
  total_earnings : ℕ  -- Total earnings in June
  remaining_money : ℕ  -- Remaining money at the end of June
  car_worth : ℕ  -- Worth of the used car

/-- The worth of the car is the difference between total earnings and remaining money -/
theorem car_worth_calculation (b : BrendanFinances) 
  (h1 : b.total_earnings = 5000)
  (h2 : b.remaining_money = 1000)
  (h3 : b.car_worth = b.total_earnings - b.remaining_money) :
  b.car_worth = 4000 := by
  sorry

#check car_worth_calculation

end car_worth_calculation_l1800_180033


namespace lock_combinations_count_l1800_180026

/-- The number of digits on the lock -/
def n : ℕ := 4

/-- The number of possible digits (0 to 9) -/
def k : ℕ := 10

/-- The number of ways to select n digits from k possibilities in non-decreasing order -/
def lockCombinations : ℕ := (n + k - 1).choose (k - 1)

theorem lock_combinations_count : lockCombinations = 715 := by
  sorry

end lock_combinations_count_l1800_180026


namespace max_rectangle_area_l1800_180022

def perimeter : ℝ := 300
def min_length : ℝ := 80
def min_width : ℝ := 40

def rectangle_area (l w : ℝ) : ℝ := l * w

theorem max_rectangle_area :
  ∀ l w : ℝ,
  l ≥ min_length →
  w ≥ min_width →
  2 * l + 2 * w = perimeter →
  rectangle_area l w ≤ 5600 :=
by sorry

end max_rectangle_area_l1800_180022


namespace siblings_age_sum_l1800_180049

/-- The age difference between each sibling -/
def age_gap : ℕ := 5

/-- The current age of the eldest sibling -/
def eldest_age : ℕ := 20

/-- The number of years into the future we're calculating -/
def years_ahead : ℕ := 10

/-- The total age of three siblings born 'age_gap' years apart, 
    where the eldest is currently 'eldest_age' years old, 
    after 'years_ahead' years -/
def total_age (age_gap eldest_age years_ahead : ℕ) : ℕ :=
  (eldest_age + years_ahead) + 
  (eldest_age - age_gap + years_ahead) + 
  (eldest_age - 2 * age_gap + years_ahead)

theorem siblings_age_sum : 
  total_age age_gap eldest_age years_ahead = 75 := by
  sorry

end siblings_age_sum_l1800_180049


namespace percentage_difference_in_earnings_l1800_180017

def mike_hourly_rate : ℝ := 12
def phil_hourly_rate : ℝ := 6

theorem percentage_difference_in_earnings : 
  (mike_hourly_rate - phil_hourly_rate) / mike_hourly_rate * 100 = 50 := by
  sorry

end percentage_difference_in_earnings_l1800_180017


namespace jessy_jewelry_count_l1800_180039

def initial_necklaces : ℕ := 10
def initial_earrings : ℕ := 15
def bought_necklaces : ℕ := 10
def bought_earrings : ℕ := (2 * initial_earrings) / 3
def mother_gift_earrings : ℕ := bought_earrings / 5 + bought_earrings

def total_jewelry : ℕ := initial_necklaces + initial_earrings + bought_necklaces + bought_earrings + mother_gift_earrings

theorem jessy_jewelry_count : total_jewelry = 57 := by
  sorry

end jessy_jewelry_count_l1800_180039


namespace bryan_books_and_magazines_l1800_180002

theorem bryan_books_and_magazines (books_per_shelf : ℕ) (magazines_per_shelf : ℕ) (num_shelves : ℕ) :
  books_per_shelf = 23 →
  magazines_per_shelf = 61 →
  num_shelves = 29 →
  books_per_shelf * num_shelves + magazines_per_shelf * num_shelves = 2436 :=
by
  sorry

end bryan_books_and_magazines_l1800_180002


namespace total_weight_BaF2_is_1051_956_l1800_180070

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 18.998

/-- The number of moles of BaF2 -/
def moles_BaF2 : ℝ := 6

/-- The molecular weight of BaF2 in g/mol -/
def molecular_weight_BaF2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_F

/-- The total weight of BaF2 in grams -/
def total_weight_BaF2 : ℝ := molecular_weight_BaF2 * moles_BaF2

/-- Theorem stating that the total weight of 6 moles of BaF2 is 1051.956 g -/
theorem total_weight_BaF2_is_1051_956 : 
  total_weight_BaF2 = 1051.956 := by sorry

end total_weight_BaF2_is_1051_956_l1800_180070


namespace collinear_points_k_value_l1800_180003

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- Theorem: If (1,1), (-1,0), and (2,k) are collinear, then k = 3/2 -/
theorem collinear_points_k_value :
  collinear 1 1 (-1) 0 2 k → k = 3/2 := by
  sorry

end collinear_points_k_value_l1800_180003


namespace cube_side_ratio_l1800_180065

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 49 / 1 → a / b = 7 / 1 := by
  sorry

end cube_side_ratio_l1800_180065


namespace nine_chapters_equations_correct_l1800_180057

/-- Represents the scenario of cars and people as described in "The Nine Chapters on the Mathematical Art" problem --/
def nine_chapters_problem (x y : ℤ) : Prop :=
  (y = 2*x + 9) ∧ (y = 3*(x - 2))

/-- Theorem stating that the equations correctly represent the described scenario --/
theorem nine_chapters_equations_correct :
  ∀ x y : ℤ, 
    nine_chapters_problem x y →
    (y = 2*x + 9) ∧ 
    (y = 3*(x - 2)) ∧
    (x > 0) ∧ 
    (y > 0) := by
  sorry

end nine_chapters_equations_correct_l1800_180057


namespace integer_representation_l1800_180015

theorem integer_representation (N : ℕ+) : 
  ∃ (p q u v : ℤ), (N : ℤ) = p * q + u * v ∧ u - v = 2 * (p - q) := by
  sorry

end integer_representation_l1800_180015


namespace pairball_playing_time_l1800_180012

theorem pairball_playing_time (total_time : ℕ) (num_children : ℕ) (h1 : total_time = 120) (h2 : num_children = 6) : 
  (2 * total_time) / num_children = 40 :=
by sorry

end pairball_playing_time_l1800_180012


namespace smallest_multiple_of_nine_l1800_180073

theorem smallest_multiple_of_nine (x y : ℤ) 
  (hx : ∃ k : ℤ, x + 2 = 9 * k) 
  (hy : ∃ k : ℤ, y - 2 = 9 * k) : 
  (∃ n : ℕ, n > 0 ∧ ∃ k : ℤ, x^2 - x*y + y^2 + n = 9 * k) ∧ 
  (∀ m : ℕ, m > 0 → (∃ k : ℤ, x^2 - x*y + y^2 + m = 9 * k) → m ≥ 6) :=
by sorry

end smallest_multiple_of_nine_l1800_180073


namespace right_triangle_inequality_l1800_180047

/-- In a right-angled triangle with legs a and b, hypotenuse c, and altitude m
    corresponding to the hypotenuse, m + c > a + b -/
theorem right_triangle_inequality (a b c m : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Pythagorean theorem
  (h_altitude : a * b = c * m) -- Area equality
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ m > 0) : m + c > a + b := by
  sorry


end right_triangle_inequality_l1800_180047


namespace diophantine_equation_equivalence_l1800_180028

/-- Given non-square integers a and b, the existence of a non-trivial integer solution
    to x^2 - ay^2 - bz^2 + abw^2 = 0 is equivalent to the existence of a non-trivial
    integer solution to x^2 - ay^2 - bz^2 = 0 -/
theorem diophantine_equation_equivalence (a b : ℤ) 
  (ha : ¬ ∃ (n : ℤ), n^2 = a) (hb : ¬ ∃ (n : ℤ), n^2 = b) :
  (∃ (x y z w : ℤ), x^2 - a*y^2 - b*z^2 + a*b*w^2 = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ w ≠ 0)) ↔
  (∃ (x y z : ℤ), x^2 - a*y^2 - b*z^2 = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry


end diophantine_equation_equivalence_l1800_180028


namespace arithmetic_sequence_formula_l1800_180079

theorem arithmetic_sequence_formula (x : ℝ) (a : ℕ → ℝ) :
  (a 1 = x - 1) →
  (a 2 = x + 1) →
  (a 3 = 2*x + 3) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = a 2 - a 1) →
  (∀ n : ℕ, n ≥ 1 → a n = 2*n - 3) :=
by sorry

end arithmetic_sequence_formula_l1800_180079


namespace jerrys_books_count_l1800_180036

theorem jerrys_books_count :
  let initial_action_figures : ℕ := 3
  let added_action_figures : ℕ := 2
  let total_action_figures := initial_action_figures + added_action_figures
  let books_count := total_action_figures + 2
  books_count = 7 := by sorry

end jerrys_books_count_l1800_180036


namespace valid_n_count_l1800_180054

-- Define the triangle sides as functions of n
def AB (n : ℕ) : ℕ := 3 * n + 6
def BC (n : ℕ) : ℕ := 2 * n + 15
def AC (n : ℕ) : ℕ := 2 * n + 5

-- Define the conditions for a valid triangle
def isValidTriangle (n : ℕ) : Prop :=
  AB n + BC n > AC n ∧
  AB n + AC n > BC n ∧
  BC n + AC n > AB n ∧
  BC n > AB n ∧
  AB n > AC n

-- Theorem stating that there are exactly 7 valid values for n
theorem valid_n_count :
  ∃! (s : Finset ℕ), s.card = 7 ∧ ∀ n, n ∈ s ↔ isValidTriangle n :=
sorry

end valid_n_count_l1800_180054


namespace sock_cost_l1800_180018

theorem sock_cost (total_cost shoes_cost : ℝ) 
  (h1 : total_cost = 111) 
  (h2 : shoes_cost = 92) : 
  (total_cost - shoes_cost) / 2 = 9.5 := by
  sorry

end sock_cost_l1800_180018


namespace next_coincidence_l1800_180091

def factory_interval : ℕ := 18
def fire_station_interval : ℕ := 24
def town_hall_interval : ℕ := 30

theorem next_coincidence (start_time : ℕ) :
  ∃ (t : ℕ), t > start_time ∧ 
  t % factory_interval = 0 ∧
  t % fire_station_interval = 0 ∧
  t % town_hall_interval = 0 ∧
  t - start_time = 360 := by
sorry

end next_coincidence_l1800_180091


namespace max_value_constraint_l1800_180058

theorem max_value_constraint (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) :
  x^2 + y^2 ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), 5 * x₀^2 + 4 * y₀^2 = 10 * x₀ ∧ x₀^2 + y₀^2 = 4 :=
by sorry

end max_value_constraint_l1800_180058


namespace carousel_horses_l1800_180010

theorem carousel_horses (blue purple green gold : ℕ) : 
  purple = 3 * blue →
  green = 2 * purple →
  gold = green / 6 →
  blue + purple + green + gold = 33 →
  blue = 3 := by
sorry

end carousel_horses_l1800_180010


namespace expression_simplification_and_evaluation_l1800_180025

theorem expression_simplification_and_evaluation :
  ∀ a : ℚ, -3 < a → a ≤ 0 → a ≠ -1 → a ≠ 0 → a ≠ 1 →
  let original_expr := (a - (2*a - 1) / a) / (1/a - a)
  let simplified_expr := (1 - a) / (1 + a)
  original_expr = simplified_expr ∧
  (a = -2 → simplified_expr = -3) :=
by sorry

end expression_simplification_and_evaluation_l1800_180025


namespace M_intersect_N_eq_closed_interval_l1800_180080

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x ≥ 3, y = Real.log (x + 1) / Real.log (1/2)}
def N : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

-- State the theorem
theorem M_intersect_N_eq_closed_interval :
  M ∩ N = Set.Icc (-3) (-2) := by sorry

end M_intersect_N_eq_closed_interval_l1800_180080


namespace floor_plus_self_equation_l1800_180027

theorem floor_plus_self_equation (r : ℝ) : ⌊r⌋ + r = 10.3 ↔ r = 5.3 := by sorry

end floor_plus_self_equation_l1800_180027


namespace alphametic_puzzle_solution_l1800_180005

theorem alphametic_puzzle_solution :
  ∃! (A R K : Nat),
    A < 10 ∧ R < 10 ∧ K < 10 ∧
    A ≠ R ∧ A ≠ K ∧ R ≠ K ∧
    1000 * A + 100 * R + 10 * K + A +
    100 * R + 10 * K + A +
    10 * K + A +
    A = 2014 ∧
    A = 1 ∧ R = 4 ∧ K = 7 :=
by sorry

end alphametic_puzzle_solution_l1800_180005


namespace jerome_classmates_l1800_180059

/-- Represents Jerome's contact list --/
structure ContactList where
  classmates : ℕ
  outOfSchoolFriends : ℕ
  familyMembers : ℕ
  total : ℕ

/-- The properties of Jerome's contact list --/
def jeromeContactList : ContactList → Prop
  | cl => cl.outOfSchoolFriends = cl.classmates / 2 ∧
          cl.familyMembers = 3 ∧
          cl.total = 33 ∧
          cl.total = cl.classmates + cl.outOfSchoolFriends + cl.familyMembers

/-- Theorem: Jerome has 20 classmates on his contact list --/
theorem jerome_classmates :
  ∀ cl : ContactList, jeromeContactList cl → cl.classmates = 20 := by
  sorry


end jerome_classmates_l1800_180059


namespace base_3_minus_base_8_digits_of_2048_l1800_180001

theorem base_3_minus_base_8_digits_of_2048 : 
  (Nat.log 3 2048 + 1) - (Nat.log 8 2048 + 1) = 4 := by
  sorry

end base_3_minus_base_8_digits_of_2048_l1800_180001


namespace quadratic_roots_sum_of_squares_l1800_180032

theorem quadratic_roots_sum_of_squares (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2 * x₁^2 + 4 * m * x₁ + m = 0 ∧
    2 * x₂^2 + 4 * m * x₂ + m = 0 ∧
    x₁^2 + x₂^2 = 3/16) →
  m = -1/8 := by
sorry

end quadratic_roots_sum_of_squares_l1800_180032


namespace city_college_juniors_seniors_l1800_180000

theorem city_college_juniors_seniors (total : ℕ) (j s : ℕ) : 
  total = 300 →
  j + s = total →
  (1 : ℚ) / 3 * j = (2 : ℚ) / 3 * s →
  j - s = 100 :=
by sorry

end city_college_juniors_seniors_l1800_180000


namespace smallest_multiple_ending_in_three_l1800_180066

theorem smallest_multiple_ending_in_three : 
  ∀ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 5 = 0 → n ≥ 53 := by
  sorry

end smallest_multiple_ending_in_three_l1800_180066


namespace complex_real_implies_m_equals_five_l1800_180006

theorem complex_real_implies_m_equals_five (m : ℝ) (z : ℂ) :
  z = Complex.I * (m^2 - 2*m - 15) → z.im = 0 → m = 5 := by sorry

end complex_real_implies_m_equals_five_l1800_180006


namespace tennis_tournament_matches_l1800_180034

theorem tennis_tournament_matches (n : Nat) (byes : Nat) :
  n = 100 →
  byes = 28 →
  ∃ m : Nat, m = n - 1 ∧ m % 11 = 0 :=
by sorry

end tennis_tournament_matches_l1800_180034


namespace expression_evaluation_l1800_180040

theorem expression_evaluation :
  let a : ℤ := 1001
  let b : ℤ := 1002
  let c : ℤ := 1000
  b^3 - a*b^2 - a^2*b + a^3 - c^3 = 2009007 := by
  sorry

end expression_evaluation_l1800_180040


namespace problem_statement_l1800_180004

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then (x - 1)^3 else |x - 1|

theorem problem_statement :
  (∃ a : ℝ, ∀ y : ℝ, ∃ x : ℝ, f a x < y) ∧
  (∀ a : ℝ, ∃ x : ℝ, f a x = 0) ∧
  (∀ a : ℝ, a > 1 → a < 2 → ∃ m : ℝ, ∃ x₁ x₂ x₃ : ℝ,
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) :=
by sorry

end problem_statement_l1800_180004


namespace trigonometric_identity_l1800_180063

theorem trigonometric_identity (α : ℝ) 
  (h : (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 2) :
  (1 + Real.sin (4 * α) - Real.cos (4 * α)) / 
  (1 + Real.sin (4 * α) + Real.cos (4 * α)) = 3 / 4 := by
  sorry

end trigonometric_identity_l1800_180063


namespace min_value_2a_plus_b_l1800_180075

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : 3 * a + b = a^2 + a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → 3 * x + y = x^2 + x * y → 2 * x + y ≥ 2 * Real.sqrt 2 + 3 :=
by sorry

end min_value_2a_plus_b_l1800_180075


namespace inequality_system_solution_l1800_180068

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (0 ≤ x ∧ x < 1) ↔ (x + 2*a ≥ 4 ∧ (2*x - b) / 3 < 1)) →
  a + b = 1 := by
sorry

end inequality_system_solution_l1800_180068


namespace regular_triangular_pyramid_volume_l1800_180093

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume
  (a : ℝ) -- base side length
  (γ : ℝ) -- angle between lateral faces
  (h : 0 < a ∧ 0 < γ ∧ γ < π) -- assumptions to ensure validity
  : ∃ V : ℝ, V = (a^3 * Real.sin (γ/2)) / (12 * Real.sqrt (3/4 - Real.sin (γ/2)^2)) :=
sorry

end regular_triangular_pyramid_volume_l1800_180093


namespace prob_red_base_is_half_l1800_180076

-- Define the total number of bases
def total_bases : ℕ := 4

-- Define the number of red educational bases
def red_bases : ℕ := 2

-- Define the probability of choosing a red educational base
def prob_red_base : ℚ := red_bases / total_bases

-- Theorem statement
theorem prob_red_base_is_half : prob_red_base = 1/2 := by
  sorry

end prob_red_base_is_half_l1800_180076


namespace hall_dimension_difference_l1800_180007

/-- Represents the dimensions and volume of a rectangular hall -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- The width is half the length, the height is one-third of the width, 
    and the volume is 600 cubic meters -/
def hall_constraints (hall : RectangularHall) : Prop :=
  hall.width = hall.length / 2 ∧
  hall.height = hall.width / 3 ∧
  hall.volume = 600

/-- The theorem stating the difference between length, width, and height -/
theorem hall_dimension_difference (hall : RectangularHall) 
  (h : hall_constraints hall) : 
  ∃ ε > 0, |hall.length - hall.width - hall.height - 6.43| < ε :=
sorry

end hall_dimension_difference_l1800_180007


namespace exists_digit_sum_div_11_l1800_180009

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem: Among 39 consecutive natural numbers, there is always at least one number
    whose sum of digits is divisible by 11. -/
theorem exists_digit_sum_div_11 (n : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (digit_sum (n + k) % 11 = 0) := by sorry

end exists_digit_sum_div_11_l1800_180009


namespace machine_quality_comparison_l1800_180053

/-- Represents a machine producing products of different quality classes -/
structure Machine where
  first_class : ℕ
  second_class : ℕ

/-- Calculates the frequency of first-class products for a machine -/
def first_class_frequency (m : Machine) : ℚ :=
  m.first_class / (m.first_class + m.second_class)

/-- Calculates the K² statistic for comparing two machines -/
def k_squared (m1 m2 : Machine) : ℚ :=
  let n := m1.first_class + m1.second_class + m2.first_class + m2.second_class
  let a := m1.first_class
  let b := m1.second_class
  let c := m2.first_class
  let d := m2.second_class
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The theorem to be proved -/
theorem machine_quality_comparison (machine_a machine_b : Machine)
  (h_a : machine_a = ⟨150, 50⟩)
  (h_b : machine_b = ⟨120, 80⟩) :
  first_class_frequency machine_a = 3/4 ∧
  first_class_frequency machine_b = 3/5 ∧
  k_squared machine_a machine_b > 6635/1000 := by
  sorry

end machine_quality_comparison_l1800_180053


namespace total_time_for_ten_pictures_l1800_180037

/-- The total time spent on drawing and coloring pictures -/
def total_time (num_pictures : ℕ) (draw_time : ℝ) (color_time_reduction : ℝ) : ℝ :=
  let color_time := draw_time * (1 - color_time_reduction)
  num_pictures * (draw_time + color_time)

/-- Theorem: The total time spent on 10 pictures is 34 hours -/
theorem total_time_for_ten_pictures :
  total_time 10 2 0.3 = 34 := by
  sorry

#eval total_time 10 2 0.3

end total_time_for_ten_pictures_l1800_180037


namespace derivative_at_pi_over_four_l1800_180074

theorem derivative_at_pi_over_four (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x * (Real.cos x + 1)) :
  deriv f (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end derivative_at_pi_over_four_l1800_180074


namespace daniel_has_five_dogs_l1800_180061

/-- The number of legs for a healthy horse -/
def horse_legs : ℕ := 4

/-- The number of legs for a healthy cat -/
def cat_legs : ℕ := 4

/-- The number of legs for a healthy turtle -/
def turtle_legs : ℕ := 4

/-- The number of legs for a healthy goat -/
def goat_legs : ℕ := 4

/-- The number of legs for a healthy dog -/
def dog_legs : ℕ := 4

/-- The number of horses Daniel has -/
def num_horses : ℕ := 2

/-- The number of cats Daniel has -/
def num_cats : ℕ := 7

/-- The number of turtles Daniel has -/
def num_turtles : ℕ := 3

/-- The number of goats Daniel has -/
def num_goats : ℕ := 1

/-- The total number of legs of all animals Daniel has -/
def total_legs : ℕ := 72

theorem daniel_has_five_dogs :
  ∃ (num_dogs : ℕ), 
    num_dogs * dog_legs + 
    num_horses * horse_legs + 
    num_cats * cat_legs + 
    num_turtles * turtle_legs + 
    num_goats * goat_legs = total_legs ∧ 
    num_dogs = 5 := by
  sorry

end daniel_has_five_dogs_l1800_180061


namespace smallest_number_with_given_remainders_l1800_180064

theorem smallest_number_with_given_remainders :
  ∃! x : ℕ,
    x > 0 ∧
    x % 5 = 2 ∧
    x % 4 = 2 ∧
    x % 6 = 3 ∧
    ∀ y : ℕ, y > 0 → y % 5 = 2 → y % 4 = 2 → y % 6 = 3 → x ≤ y :=
by
  -- The proof goes here
  sorry

end smallest_number_with_given_remainders_l1800_180064


namespace tan_alpha_neg_half_implies_expression_eq_neg_third_l1800_180083

theorem tan_alpha_neg_half_implies_expression_eq_neg_third (α : Real) 
  (h : Real.tan α = -1/2) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1/3 := by
  sorry

end tan_alpha_neg_half_implies_expression_eq_neg_third_l1800_180083


namespace area_conversions_l1800_180013

-- Define conversion rates
def sq_dm_to_sq_cm : ℝ := 100
def hectare_to_sq_m : ℝ := 10000
def sq_km_to_hectare : ℝ := 100
def sq_m_to_sq_dm : ℝ := 100

-- Theorem to prove the conversions
theorem area_conversions :
  (7 * sq_dm_to_sq_cm = 700) ∧
  (5 * hectare_to_sq_m = 50000) ∧
  (600 / sq_km_to_hectare = 6) ∧
  (200 / sq_m_to_sq_dm = 2) :=
by sorry

end area_conversions_l1800_180013


namespace reciprocal_problem_l1800_180060

theorem reciprocal_problem (x : ℚ) (h : 7 * x = 3) : 150 * (1 / x) = 350 := by
  sorry

end reciprocal_problem_l1800_180060


namespace bridge_length_l1800_180096

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 →
  train_speed_kmh = 42.3 →
  crossing_time = 40 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 320 := by
  sorry

#check bridge_length

end bridge_length_l1800_180096


namespace equal_distribution_l1800_180024

theorem equal_distribution (total_amount : ℕ) (num_persons : ℕ) (amount_per_person : ℕ) :
  total_amount = 42900 →
  num_persons = 22 →
  amount_per_person = total_amount / num_persons →
  amount_per_person = 1950 :=
by
  sorry

end equal_distribution_l1800_180024


namespace kangaroo_jumps_odd_jumps_zero_four_jumps_two_l1800_180056

/-- Represents a regular octagon with vertices labeled from 0 to 7 -/
def Octagon := Fin 8

/-- Defines whether two vertices are adjacent in the octagon -/
def adjacent (v w : Octagon) : Prop :=
  (v.val + 1) % 8 = w.val ∨ (w.val + 1) % 8 = v.val

/-- Defines the number of ways a kangaroo can reach vertex E from A in n jumps -/
def num_ways (n : ℕ) : ℕ :=
  sorry -- Definition to be implemented

/-- Main theorem: Characterizes the number of ways to reach E from A in n jumps -/
theorem kangaroo_jumps (n : ℕ) :
  num_ways n = if n % 2 = 0
    then let m := n / 2
         (((2 : ℝ) + Real.sqrt 2) ^ (m - 1) - ((2 : ℝ) - Real.sqrt 2) ^ (m - 1)) / Real.sqrt 2
    else 0 :=
  sorry

/-- The number of ways to reach E from A in an odd number of jumps is 0 -/
theorem odd_jumps_zero (n : ℕ) (h : n % 2 = 1) :
  num_ways n = 0 :=
  sorry

/-- The number of ways to reach E from A in 4 jumps is 2 -/
theorem four_jumps_two :
  num_ways 4 = 2 :=
  sorry

end kangaroo_jumps_odd_jumps_zero_four_jumps_two_l1800_180056


namespace circle_center_proof_l1800_180055

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation x² - 8x + y² - 4y = 4, prove that its center is (4, 2) -/
theorem circle_center_proof (eq : CircleEquation) 
    (h1 : eq.a = 1)
    (h2 : eq.b = -8)
    (h3 : eq.c = 1)
    (h4 : eq.d = -4)
    (h5 : eq.e = -4) :
    CircleCenter.mk 4 2 = CircleCenter.mk (-eq.b / (2 * eq.a)) (-eq.d / (2 * eq.c)) :=
  sorry

end circle_center_proof_l1800_180055


namespace circle_radius_from_chord_and_secant_l1800_180041

/-- Given a circle with a chord of length 10 and a secant parallel to the tangent at one end of the chord,
    where the internal segment of the secant is 12 units long, the radius of the circle is 10 units. -/
theorem circle_radius_from_chord_and_secant (C : ℝ → ℝ → Prop) (A B M : ℝ × ℝ) (r : ℝ) :
  (∀ x y, C x y ↔ (x - r)^2 + (y - r)^2 = r^2) →  -- C is a circle with center (r, r) and radius r
  C A.1 A.2 →  -- A is on the circle
  C B.1 B.2 →  -- B is on the circle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 100 →  -- AB is a chord of length 10
  (∃ t : ℝ, C (A.1 + t) (A.2 + t) ∧ (A.1 + t - B.1)^2 + (A.2 + t - B.2)^2 = 36) →  -- Secant parallel to tangent at A
  r = 10 :=
by sorry

end circle_radius_from_chord_and_secant_l1800_180041


namespace unique_solution_trig_equation_l1800_180011

theorem unique_solution_trig_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  (Real.tan ((150 : ℝ) - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) ∧
  x = 120 := by
sorry

end unique_solution_trig_equation_l1800_180011


namespace shopkeeper_profit_calculation_l1800_180016

/-- Represents the profit percentage calculation for a shopkeeper's sale --/
theorem shopkeeper_profit_calculation 
  (cost_price : ℝ) 
  (discount_percent : ℝ) 
  (profit_with_discount : ℝ) 
  (h_positive_cp : cost_price > 0)
  (h_discount : discount_percent = 5)
  (h_profit : profit_with_discount = 20.65) :
  let selling_price_with_discount := cost_price * (1 - discount_percent / 100)
  let selling_price_no_discount := cost_price * (1 + profit_with_discount / 100)
  let profit_no_discount := (selling_price_no_discount - cost_price) / cost_price * 100
  profit_no_discount = profit_with_discount := by
sorry

end shopkeeper_profit_calculation_l1800_180016


namespace total_value_is_five_dollars_l1800_180084

/-- Represents the value of different coin types in dollars -/
def coin_value : Fin 4 → ℚ
  | 0 => 0.25  -- Quarter
  | 1 => 0.10  -- Dime
  | 2 => 0.05  -- Nickel
  | 3 => 0.01  -- Penny

/-- Represents the count of each coin type -/
def coin_count : Fin 4 → ℕ
  | 0 => 10   -- Quarters
  | 1 => 3    -- Dimes
  | 2 => 4    -- Nickels
  | 3 => 200  -- Pennies

/-- Calculates the total value of coins -/
def total_value : ℚ :=
  (Finset.sum Finset.univ (λ i => coin_value i * coin_count i))

/-- Theorem stating that the total value of coins is $5.00 -/
theorem total_value_is_five_dollars : total_value = 5 := by
  sorry

end total_value_is_five_dollars_l1800_180084


namespace oil_tank_explosion_theorem_l1800_180069

/-- The number of bullets available -/
def num_bullets : ℕ := 5

/-- The probability of hitting the target on each shot -/
def hit_probability : ℚ := 2/3

/-- The probability of the oil tank exploding -/
def explosion_probability : ℚ := 232/243

/-- The probability that the number of shots is not less than 4 -/
def shots_ge_4_probability : ℚ := 7/27

/-- Each shot is independent and the probability of hitting each time is 2/3.
    The first hit causes oil to flow out, and the second hit causes an explosion.
    Shooting stops when the oil tank explodes or bullets run out. -/
theorem oil_tank_explosion_theorem :
  (∀ (n : ℕ), n ≤ num_bullets → (hit_probability^n * (1 - hit_probability)^(num_bullets - n) : ℚ) = (2/3)^n * (1/3)^(num_bullets - n)) →
  explosion_probability = 232/243 ∧
  shots_ge_4_probability = 7/27 :=
sorry

end oil_tank_explosion_theorem_l1800_180069


namespace B_k_closed_form_l1800_180038

/-- B_k(n) is the largest possible number of elements in a 2-separable k-configuration of a set with 2n elements -/
def B_k (k n : ℕ) : ℕ := Nat.choose (2*n) k - 2 * Nat.choose n k

/-- Theorem stating the closed-form expression for B_k(n) -/
theorem B_k_closed_form (k n : ℕ) (h1 : 2 ≤ k) (h2 : k ≤ n) :
  B_k k n = Nat.choose (2*n) k - 2 * Nat.choose n k := by
  sorry

end B_k_closed_form_l1800_180038


namespace pascal_triangle_43_numbers_l1800_180087

/-- The number of elements in a row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

/-- The second number in a row of Pascal's triangle -/
def pascal_second_number (n : ℕ) : ℕ := n

theorem pascal_triangle_43_numbers :
  ∃ n : ℕ, pascal_row_length n = 43 ∧ pascal_second_number n = 42 :=
sorry

end pascal_triangle_43_numbers_l1800_180087


namespace mark_age_is_18_l1800_180014

/-- Represents the ages of family members --/
structure FamilyAges where
  mark : ℕ
  john : ℕ
  parents : ℕ

/-- Defines the relationships between family members' ages --/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.john = ages.mark - 10 ∧
  ages.parents = 5 * ages.john ∧
  ages.parents - 22 = ages.mark

/-- Theorem stating that Mark's age is 18 given the family age relationships --/
theorem mark_age_is_18 :
  ∀ (ages : FamilyAges), validFamilyAges ages → ages.mark = 18 := by
  sorry

end mark_age_is_18_l1800_180014


namespace circumscribed_sphere_surface_area_l1800_180046

theorem circumscribed_sphere_surface_area (a : ℝ) (h : a = 2 * Real.sqrt 3 / 3) :
  let R := Real.sqrt 3 * a / 2
  4 * Real.pi * R^2 = 4 * Real.pi :=
by sorry

end circumscribed_sphere_surface_area_l1800_180046


namespace triangle_bottom_number_l1800_180085

/-- Define the triangle structure -/
def Triangle (n : ℕ) : Type :=
  Fin n → Fin n → ℕ

/-- The first row of the triangle contains numbers from 1 to 2000 -/
def first_row_condition (t : Triangle 2000) : Prop :=
  ∀ i : Fin 2000, t 0 i = i.val + 1

/-- Each subsequent number is the sum of the two numbers immediately above it -/
def sum_condition (t : Triangle 2000) : Prop :=
  ∀ i j : Fin 2000, i > 0 → t i j = t (i-1) j + t (i-1) (j+1)

/-- The theorem to be proved -/
theorem triangle_bottom_number (t : Triangle 2000) 
  (h1 : first_row_condition t) (h2 : sum_condition t) : 
  t 1999 0 = 2^1998 * 2001 := by
  sorry

end triangle_bottom_number_l1800_180085


namespace group_collection_l1800_180092

/-- Calculates the total amount collected in rupees given the number of students in a group,
    where each student contributes as many paise as there are members. -/
def totalCollected (numStudents : ℕ) : ℚ :=
  (numStudents * numStudents : ℚ) / 100

/-- Theorem stating that for a group of 96 students, the total amount collected is 92.16 rupees. -/
theorem group_collection :
  totalCollected 96 = 92.16 := by
  sorry

end group_collection_l1800_180092


namespace solve_system_for_y_l1800_180044

theorem solve_system_for_y (x y : ℚ) 
  (eq1 : 2 * x - y = 10) 
  (eq2 : x + 3 * y = 2) : 
  y = -6/7 := by sorry

end solve_system_for_y_l1800_180044


namespace walnut_trees_remaining_l1800_180008

/-- The number of walnut trees remaining after some are cut down -/
def remaining_walnut_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of remaining walnut trees is 29 -/
theorem walnut_trees_remaining :
  remaining_walnut_trees 42 13 = 29 := by
  sorry

end walnut_trees_remaining_l1800_180008


namespace complex_fraction_equality_l1800_180020

theorem complex_fraction_equality : Complex.I / (1 + Complex.I) = (1 / 2 : ℂ) + (1 / 2 : ℂ) * Complex.I := by
  sorry

end complex_fraction_equality_l1800_180020


namespace simplify_radical_sum_l1800_180048

theorem simplify_radical_sum : Real.sqrt 98 + Real.sqrt 32 + (27 : Real).rpow (1/3) = 11 * Real.sqrt 2 + 3 := by
  sorry

end simplify_radical_sum_l1800_180048


namespace exists_function_double_composition_l1800_180030

theorem exists_function_double_composition :
  ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f (f n) = 2 * n := by
  sorry

end exists_function_double_composition_l1800_180030


namespace largest_prime_divisor_test_l1800_180035

theorem largest_prime_divisor_test (m : ℕ) : 
  700 ≤ m → m ≤ 750 → 
  (∀ p : ℕ, p.Prime → p ≤ 23 → m % p ≠ 0) → 
  m.Prime :=
sorry

end largest_prime_divisor_test_l1800_180035


namespace cube_volume_after_removal_l1800_180072

/-- Theorem: Volume of a cube with edge sum 72 cm after removing a 1 cm cube corner -/
theorem cube_volume_after_removal (edge_sum : ℝ) (small_cube_edge : ℝ) : 
  edge_sum = 72 → small_cube_edge = 1 → 
  (edge_sum / 12)^3 - small_cube_edge^3 = 215 := by
  sorry

end cube_volume_after_removal_l1800_180072


namespace diophantine_equation_solutions_l1800_180023

theorem diophantine_equation_solutions : 
  ∀ a b c : ℕ+, 
  (8 * a.val - 5 * b.val)^2 + (3 * b.val - 2 * c.val)^2 + (3 * c.val - 7 * a.val)^2 = 2 ↔ 
  ((a.val = 3 ∧ b.val = 5 ∧ c.val = 7) ∨ (a.val = 12 ∧ b.val = 19 ∧ c.val = 28)) :=
by sorry

#check diophantine_equation_solutions

end diophantine_equation_solutions_l1800_180023


namespace factorization_equality_l1800_180045

theorem factorization_equality (a b : ℝ) : a * b^2 - a = a * (b + 1) * (b - 1) := by
  sorry

end factorization_equality_l1800_180045


namespace special_set_characterization_l1800_180071

/-- The set of integers n ≥ 1 such that 2^n - 1 has exactly n positive integer divisors -/
def special_set : Set ℕ+ :=
  {n | (Nat.card (Nat.divisors ((2:ℕ)^(n:ℕ) - 1))) = n}

/-- Theorem stating that the special set is equal to {1, 2, 4, 6, 8, 16, 32} -/
theorem special_set_characterization :
  special_set = {1, 2, 4, 6, 8, 16, 32} := by sorry

end special_set_characterization_l1800_180071


namespace catastrophic_network_properties_l1800_180089

/-- A catastrophic road network between 6 cities -/
structure CatastrophicNetwork :=
  (cities : Fin 6 → Type)
  (road : cities i → cities j → Prop)
  (no_return : ∀ (i j : Fin 6) (x : cities i) (y : cities j), road x y → ¬ ∃ path : cities j → cities i, True)

theorem catastrophic_network_properties (n : CatastrophicNetwork) :
  (∃ i : Fin 6, ∀ j : Fin 6, ¬ ∃ x : n.cities i, ∃ y : n.cities j, n.road x y) ∧
  (∃ i : Fin 6, ∀ j : Fin 6, j ≠ i → ∃ x : n.cities i, ∃ y : n.cities j, n.road x y) ∧
  (∃ i j : Fin 6, ∀ k l : Fin 6, ∃ path : n.cities k → n.cities l, True) ∧
  (∃ f : Fin 6 → Fin 6, Function.Bijective f ∧ 
    ∀ i j : Fin 6, i ≠ j → (f i < f j ↔ ∃ x : n.cities i, ∃ y : n.cities j, n.road x y)) :=
sorry

#check catastrophic_network_properties

end catastrophic_network_properties_l1800_180089


namespace lilies_per_centerpiece_l1800_180021

/-- Proves that the number of lilies per centerpiece is 6 given the specified conditions -/
theorem lilies_per_centerpiece
  (num_centerpieces : ℕ)
  (roses_per_centerpiece : ℕ)
  (orchids_per_centerpiece : ℕ)
  (total_budget : ℚ)
  (flower_cost : ℚ)
  (h1 : num_centerpieces = 6)
  (h2 : roses_per_centerpiece = 8)
  (h3 : orchids_per_centerpiece = 2 * roses_per_centerpiece)
  (h4 : total_budget = 2700)
  (h5 : flower_cost = 15)
  : (total_budget / flower_cost / num_centerpieces : ℚ) - roses_per_centerpiece - orchids_per_centerpiece = 6 :=
sorry

end lilies_per_centerpiece_l1800_180021


namespace dancing_and_math_intersection_l1800_180094

theorem dancing_and_math_intersection (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (a b : ℕ),
    b ≥ 1 ∧
    (a + b)^2 = (p + 1) * a + b ∧
    b = 1 :=
by sorry

end dancing_and_math_intersection_l1800_180094


namespace intersection_equals_C_l1800_180042

-- Define the set of angles less than 90°
def A : Set ℝ := {α | α < 90}

-- Define the set of angles in the first quadrant
def B : Set ℝ := {α | ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90}

-- Define the set of angles α such that k · 360° < α < k · 360° + 90° for some integer k ≤ 0
def C : Set ℝ := {α | ∃ k : ℤ, k ≤ 0 ∧ k * 360 < α ∧ α < k * 360 + 90}

-- Theorem statement
theorem intersection_equals_C : A ∩ B = C := by sorry

end intersection_equals_C_l1800_180042


namespace greatest_multiple_of_five_cubed_less_than_8000_l1800_180099

theorem greatest_multiple_of_five_cubed_less_than_8000 :
  ∃ (y : ℕ), y > 0 ∧ 5 ∣ y ∧ y^3 < 8000 ∧ ∀ (z : ℕ), z > 0 → 5 ∣ z → z^3 < 8000 → z ≤ y :=
by sorry

end greatest_multiple_of_five_cubed_less_than_8000_l1800_180099


namespace triangle_proof_l1800_180078

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_proof (t : Triangle) (h1 : t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0)
                       (h2 : t.a = 2)
                       (h3 : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry


end triangle_proof_l1800_180078


namespace max_first_term_arithmetic_progression_l1800_180029

def arithmetic_progression (a₁ : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n+1 => arithmetic_progression a₁ d n + d

def sum_arithmetic_progression (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem max_first_term_arithmetic_progression 
  (a₁ : ℚ) (d : ℚ) 
  (h₁ : ∃ (n : ℕ), sum_arithmetic_progression a₁ d 4 = n)
  (h₂ : ∃ (m : ℕ), sum_arithmetic_progression a₁ d 7 = m)
  (h₃ : a₁ ≤ 2/3) :
  a₁ ≤ 9/14 :=
sorry

end max_first_term_arithmetic_progression_l1800_180029


namespace cos_angle_relation_l1800_180095

theorem cos_angle_relation (α : Real) (h : Real.cos (75 * Real.pi / 180 + α) = 1/2) :
  Real.cos (105 * Real.pi / 180 - α) = -1/2 := by
  sorry

end cos_angle_relation_l1800_180095


namespace rodrigos_classroom_chairs_l1800_180097

/-- The number of chairs left in Rodrigo's classroom after Lisa borrows some chairs -/
def chairs_left (red_chairs yellow_chairs blue_chairs borrowed : ℕ) : ℕ :=
  red_chairs + yellow_chairs + blue_chairs - borrowed

/-- Theorem stating the number of chairs left in Rodrigo's classroom -/
theorem rodrigos_classroom_chairs :
  ∀ (red_chairs : ℕ),
  red_chairs = 4 →
  ∀ (yellow_chairs : ℕ),
  yellow_chairs = 2 * red_chairs →
  ∀ (blue_chairs : ℕ),
  blue_chairs = yellow_chairs - 2 →
  chairs_left red_chairs yellow_chairs blue_chairs 3 = 15 :=
by
  sorry

end rodrigos_classroom_chairs_l1800_180097


namespace fraction_value_zero_l1800_180098

theorem fraction_value_zero (B A P E H b K p J C O : ℕ) :
  (B ≠ A ∧ B ≠ P ∧ B ≠ E ∧ B ≠ H ∧ B ≠ b ∧ B ≠ K ∧ B ≠ p ∧ B ≠ J ∧ B ≠ C ∧ B ≠ O) ∧
  (A ≠ P ∧ A ≠ E ∧ A ≠ H ∧ A ≠ b ∧ A ≠ K ∧ A ≠ p ∧ A ≠ J ∧ A ≠ C ∧ A ≠ O) ∧
  (P ≠ E ∧ P ≠ H ∧ P ≠ b ∧ P ≠ K ∧ P ≠ p ∧ P ≠ J ∧ P ≠ C ∧ P ≠ O) ∧
  (E ≠ H ∧ E ≠ b ∧ E ≠ K ∧ E ≠ p ∧ E ≠ J ∧ E ≠ C ∧ E ≠ O) ∧
  (H ≠ b ∧ H ≠ K ∧ H ≠ p ∧ H ≠ J ∧ H ≠ C ∧ H ≠ O) ∧
  (b ≠ K ∧ b ≠ p ∧ b ≠ J ∧ b ≠ C ∧ b ≠ O) ∧
  (K ≠ p ∧ K ≠ J ∧ K ≠ C ∧ K ≠ O) ∧
  (p ≠ J ∧ p ≠ C ∧ p ≠ O) ∧
  (J ≠ C ∧ J ≠ O) ∧
  (C ≠ O) ∧
  (B < 10 ∧ A < 10 ∧ P < 10 ∧ E < 10 ∧ H < 10 ∧ b < 10 ∧ K < 10 ∧ p < 10 ∧ J < 10 ∧ C < 10 ∧ O < 10) →
  (B * A * P * E * H * b * E : ℚ) / (K * A * p * J * C * O * H : ℕ) = 0 :=
by sorry

end fraction_value_zero_l1800_180098


namespace stars_permutations_l1800_180051

def word_length : ℕ := 5
def repeated_letter_count : ℕ := 2
def unique_letters_count : ℕ := 3

theorem stars_permutations :
  (word_length.factorial) / (repeated_letter_count.factorial) = 60 := by
  sorry

end stars_permutations_l1800_180051


namespace greatest_integer_with_gcd_six_l1800_180081

theorem greatest_integer_with_gcd_six : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 → Nat.gcd m 18 = 6 → m ≤ n := by
  sorry

end greatest_integer_with_gcd_six_l1800_180081


namespace book_weight_l1800_180031

theorem book_weight (num_books : ℕ) (total_weight : ℝ) (bag_weight : ℝ) :
  num_books = 14 →
  total_weight = 11.14 →
  bag_weight = 0.5 →
  (total_weight - bag_weight) / num_books = 0.76 := by
  sorry

end book_weight_l1800_180031


namespace metal_bar_weight_l1800_180090

/-- Represents the properties of a metal alloy bar --/
structure MetalBar where
  tin_weight : ℝ
  silver_weight : ℝ
  total_weight_loss : ℝ
  tin_loss_rate : ℝ
  silver_loss_rate : ℝ
  tin_silver_ratio : ℝ

/-- Theorem stating the weight of the metal bar given the conditions --/
theorem metal_bar_weight (bar : MetalBar)
  (h1 : bar.total_weight_loss = 6)
  (h2 : bar.tin_loss_rate = 1.375 / 10)
  (h3 : bar.silver_loss_rate = 0.375 / 5)
  (h4 : bar.tin_silver_ratio = 2 / 3)
  (h5 : bar.tin_weight * bar.tin_loss_rate + bar.silver_weight * bar.silver_loss_rate = bar.total_weight_loss)
  (h6 : bar.tin_weight / bar.silver_weight = bar.tin_silver_ratio) :
  bar.tin_weight + bar.silver_weight = 60 := by
  sorry

end metal_bar_weight_l1800_180090


namespace power_of_eleven_l1800_180050

/-- Given an expression (11)^n * (4)^11 * (7)^5 where the total number of prime factors is 29,
    prove that the value of n (the power of 11) is 2. -/
theorem power_of_eleven (n : ℕ) : 
  (n + 22 + 5 = 29) → n = 2 := by
sorry

end power_of_eleven_l1800_180050


namespace savings_interest_rate_equation_l1800_180019

/-- Represents the annual interest rate calculation for a savings account --/
theorem savings_interest_rate_equation 
  (initial_amount : ℝ) 
  (final_amount : ℝ) 
  (years : ℕ) 
  (interest_rate : ℝ) 
  (h1 : initial_amount = 3000) 
  (h2 : final_amount = 3243) 
  (h3 : years = 3) :
  final_amount = initial_amount + initial_amount * years * (interest_rate / 100) :=
by sorry

end savings_interest_rate_equation_l1800_180019


namespace minimize_sum_with_constraint_l1800_180067

theorem minimize_sum_with_constraint :
  ∀ a b : ℕ+,
  (4 * a.val + b.val = 30) →
  (∀ x y : ℕ+, (4 * x.val + y.val = 30) → (a.val + b.val ≤ x.val + y.val)) →
  (a.val = 7 ∧ b.val = 2) :=
by sorry

end minimize_sum_with_constraint_l1800_180067


namespace product_seven_consecutive_divisible_by_ten_l1800_180086

theorem product_seven_consecutive_divisible_by_ten (n : ℕ) : 
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) :=
by sorry

end product_seven_consecutive_divisible_by_ten_l1800_180086


namespace f_n_formula_l1800_180062

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => f x
  | m + 1 => deriv (f_n m) x

theorem f_n_formula (n : ℕ) (x : ℝ) :
  f_n (n + 1) x = ((-1)^(n + 1) * (x - (n + 1))) / Real.exp x :=
by sorry

end f_n_formula_l1800_180062


namespace workers_per_block_l1800_180052

/-- Proves that given a total budget of $4000, a cost of $4 per gift, and 10 blocks in the company,
    the number of workers in each block is 100. -/
theorem workers_per_block (total_budget : ℕ) (cost_per_gift : ℕ) (num_blocks : ℕ)
  (h1 : total_budget = 4000)
  (h2 : cost_per_gift = 4)
  (h3 : num_blocks = 10) :
  (total_budget / cost_per_gift) / num_blocks = 100 := by
sorry

#eval (4000 / 4) / 10  -- Should output 100

end workers_per_block_l1800_180052


namespace absolute_value_inequality_l1800_180088

theorem absolute_value_inequality (a b c : ℝ) : 
  |a - c| < |b| → |a| < |b| + |c| := by
  sorry

end absolute_value_inequality_l1800_180088


namespace line_symmetry_l1800_180082

-- Define the original line
def original_line (x y : ℝ) : Prop := 2 * x - y = 0

-- Define symmetry with respect to x-axis
def symmetric_wrt_x_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, g x = -f x

-- Define the proposed symmetric line
def proposed_symmetric_line (x y : ℝ) : Prop := 2 * x + y = 0

-- Theorem statement
theorem line_symmetry :
  ∃ (f g : ℝ → ℝ),
    (∀ x y, original_line x y ↔ y = f x) ∧
    (∀ x y, proposed_symmetric_line x y ↔ y = g x) ∧
    symmetric_wrt_x_axis f g :=
sorry

end line_symmetry_l1800_180082


namespace raft_minimum_capacity_l1800_180043

/-- Represents an animal with its weight -/
structure Animal where
  weight : ℕ

/-- Represents the raft with its capacity -/
structure Raft where
  capacity : ℕ

/-- Checks if the raft can carry at least two of the lightest animals -/
def canCarryTwoLightest (r : Raft) (animals : List Animal) : Prop :=
  r.capacity ≥ 2 * (animals.map Animal.weight).minimum

/-- Checks if all animals can be transported using the given raft -/
def canTransportAll (r : Raft) (animals : List Animal) : Prop :=
  canCarryTwoLightest r animals

/-- The theorem to be proved -/
theorem raft_minimum_capacity 
  (mice : List Animal) 
  (moles : List Animal) 
  (hamsters : List Animal) 
  (h_mice : mice.length = 5 ∧ ∀ m ∈ mice, m.weight = 70)
  (h_moles : moles.length = 3 ∧ ∀ m ∈ moles, m.weight = 90)
  (h_hamsters : hamsters.length = 4 ∧ ∀ h ∈ hamsters, h.weight = 120)
  : ∃ (r : Raft), r.capacity = 140 ∧ canTransportAll r (mice ++ moles ++ hamsters) :=
sorry

end raft_minimum_capacity_l1800_180043


namespace floor_of_4_7_l1800_180077

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by
  sorry

end floor_of_4_7_l1800_180077
