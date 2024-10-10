import Mathlib

namespace greatest_integer_inequality_l548_54866

theorem greatest_integer_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 5 * x - 4 < 3 - 2 * x :=
by sorry

end greatest_integer_inequality_l548_54866


namespace no_equal_notebooks_l548_54842

theorem no_equal_notebooks : ¬∃ (x : ℝ), x > 0 ∧ 12 / x = 21 / (x + 1.2) := by
  sorry

end no_equal_notebooks_l548_54842


namespace halloween_bags_cost_l548_54848

/-- Calculates the minimum cost to buy a given number of items, 
    where items can be bought in packs of 5 or individually --/
def minCost (numItems : ℕ) (packPrice packSize : ℕ) (individualPrice : ℕ) : ℕ :=
  let numPacks := numItems / packSize
  let numIndividuals := numItems % packSize
  numPacks * packPrice + numIndividuals * individualPrice

theorem halloween_bags_cost : 
  let totalStudents : ℕ := 25
  let vampireRequests : ℕ := 11
  let pumpkinRequests : ℕ := 14
  let packPrice : ℕ := 3
  let packSize : ℕ := 5
  let individualPrice : ℕ := 1
  
  vampireRequests + pumpkinRequests = totalStudents →
  
  minCost vampireRequests packPrice packSize individualPrice + 
  minCost pumpkinRequests packPrice packSize individualPrice = 17 := by
  sorry

end halloween_bags_cost_l548_54848


namespace existence_of_x_l548_54824

/-- A sequence of nonnegative integers satisfying the given condition -/
def SequenceCondition (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≥ 1 → j ≥ 1 → i + j ≤ 1997 →
    a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1

/-- The main theorem -/
theorem existence_of_x (a : ℕ → ℕ) (h : SequenceCondition a) :
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 → a n = ⌊n * x⌋ := by
  sorry

end existence_of_x_l548_54824


namespace digit_sum_problem_l548_54823

theorem digit_sum_problem (a b c x s z : ℕ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → x ≠ 0 → s ≠ 0 → z ≠ 0 →
  a + b = x →
  x + c = s →
  s + a = z →
  b + c + z = 16 →
  s = 8 := by
sorry

end digit_sum_problem_l548_54823


namespace instantaneous_velocity_at_2_seconds_l548_54807

-- Define the motion equation
def s (t : ℝ) : ℝ := (2 * t + 3) ^ 2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := 4 * (2 * t + 3)

-- Theorem statement
theorem instantaneous_velocity_at_2_seconds : v 2 = 28 := by
  sorry

end instantaneous_velocity_at_2_seconds_l548_54807


namespace sqrt_equation_roots_l548_54890

theorem sqrt_equation_roots (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ Real.sqrt (x - p) = x ∧ Real.sqrt (y - p) = y) ↔ 0 ≤ p ∧ p < (1/4 : ℝ) :=
sorry

end sqrt_equation_roots_l548_54890


namespace circle_radius_l548_54815

/-- The radius of the circle defined by x^2 + y^2 - 8x = 0 is 4 -/
theorem circle_radius (x y : ℝ) : (x^2 + y^2 - 8*x = 0) → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 4^2 := by
  sorry

end circle_radius_l548_54815


namespace percentage_of_apples_after_adding_l548_54825

/-- Given a basket of fruits with the following conditions:
  * x is the initial number of apples
  * y is the initial number of oranges
  * z is the number of oranges added
  * w is the number of apples added
  * The sum of initial apples and oranges is 30
  * The sum of added oranges and apples is 12
  * The ratio of initial apples to initial oranges is 2:1
  * The ratio of added apples to added oranges is 3:1
  Prove that the percentage of apples in the basket after adding extra fruits is (29/42) * 100 -/
theorem percentage_of_apples_after_adding (x y z w : ℕ) : 
  x + y = 30 →
  z + w = 12 →
  x = 2 * y →
  w = 3 * z →
  (x + w : ℚ) / (x + y + z + w) * 100 = 29 / 42 * 100 := by
  sorry

end percentage_of_apples_after_adding_l548_54825


namespace problem_statement_l548_54844

theorem problem_statement (x : ℝ) : 
  (1/5)^35 * (1/4)^18 = 1/(x*(10)^35) → x = 2 := by
  sorry

end problem_statement_l548_54844


namespace area_of_side_face_l548_54834

/-- Represents a rectangular box with length, width, and height -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Theorem: Area of side face of a rectangular box -/
theorem area_of_side_face (b : Box) 
  (h1 : b.width * b.height = 0.5 * (b.length * b.width))
  (h2 : b.length * b.width = 1.5 * (b.length * b.height))
  (h3 : b.length * b.width * b.height = 5184) :
  b.length * b.height = 288 := by
  sorry

end area_of_side_face_l548_54834


namespace charitable_woman_age_l548_54801

theorem charitable_woman_age (x : ℚ) : 
  (x / 2 + 1) + ((x / 2 - 1) / 2 + 2) + ((x / 4 - 3 / 2) / 2 + 3) + 1 = x → x = 38 :=
by sorry

end charitable_woman_age_l548_54801


namespace triangle_is_equilateral_l548_54804

theorem triangle_is_equilateral (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π / 2 →  -- Angle A is acute
  3 * b = 2 * Real.sqrt 3 * a * Real.sin B →  -- Given equation
  Real.cos B = Real.cos C →  -- Given condition
  0 < B ∧ B < π →  -- B is a valid angle
  0 < C ∧ C < π →  -- C is a valid angle
  A + B + C = π →  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3  -- Equilateral triangle
  := by sorry

end triangle_is_equilateral_l548_54804


namespace simplify_expression_l548_54828

theorem simplify_expression :
  let x : ℝ := 3
  let expr := (Real.sqrt (x - 2 * Real.sqrt 2)) / (Real.sqrt (x^2 - 4*x*Real.sqrt 2 + 8)) -
               (Real.sqrt (x + 2 * Real.sqrt 2)) / (Real.sqrt (x^2 + 4*x*Real.sqrt 2 + 8))
  expr = 2 := by sorry

end simplify_expression_l548_54828


namespace parallel_lines_intersection_l548_54805

/-- Two lines are parallel if they have the same slope -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- A point (x, y) lies on a line ax + by = c if the equation is satisfied -/
def point_on_line (a b c x y : ℝ) : Prop := a * x + b * y = c

theorem parallel_lines_intersection (c d : ℝ) : 
  parallel (3 / 4) (-6 / d) ∧ 
  point_on_line 3 (-4) c 2 (-3) ∧
  point_on_line 6 d (2 * c) 2 (-3) →
  c = 18 ∧ d = -8 := by
sorry

end parallel_lines_intersection_l548_54805


namespace statement_a_statement_b_l548_54877

-- Define rationality for real numbers
def IsRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Statement a
theorem statement_a : ∃ (x : ℝ), IsRational (x^7) ∧ IsRational (x^12) ∧ ¬IsRational x := by
  sorry

-- Statement b
theorem statement_b : ∀ (x : ℝ), IsRational (x^9) ∧ IsRational (x^12) → IsRational x := by
  sorry

end statement_a_statement_b_l548_54877


namespace kittens_remaining_l548_54865

theorem kittens_remaining (initial_kittens given_away : ℕ) : 
  initial_kittens = 8 → given_away = 2 → initial_kittens - given_away = 6 := by
  sorry

end kittens_remaining_l548_54865


namespace total_problems_l548_54840

def math_pages : ℕ := 6
def reading_pages : ℕ := 4
def problems_per_page : ℕ := 3

theorem total_problems : math_pages + reading_pages * problems_per_page = 30 := by
  sorry

end total_problems_l548_54840


namespace prob_red_then_blue_is_one_thirteenth_l548_54806

def total_marbles : ℕ := 4 + 3 + 6

def red_marbles : ℕ := 4
def blue_marbles : ℕ := 3
def yellow_marbles : ℕ := 6

def prob_red_then_blue : ℚ := (red_marbles : ℚ) / total_marbles * blue_marbles / (total_marbles - 1)

theorem prob_red_then_blue_is_one_thirteenth :
  prob_red_then_blue = 1 / 13 := by
  sorry

end prob_red_then_blue_is_one_thirteenth_l548_54806


namespace sum_of_squares_and_square_of_sum_l548_54875

theorem sum_of_squares_and_square_of_sum : (5 + 7)^2 + (5^2 + 7^2) = 218 := by
  sorry

end sum_of_squares_and_square_of_sum_l548_54875


namespace quadratic_discriminant_l548_54831

-- Define the discriminant function for a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- State the theorem
theorem quadratic_discriminant :
  discriminant 1 3 1 = 5 := by sorry

end quadratic_discriminant_l548_54831


namespace doll_completion_time_l548_54841

/-- Time in minutes to craft one doll -/
def craft_time : ℕ := 105

/-- Break time in minutes -/
def break_time : ℕ := 30

/-- Number of dolls to be made -/
def num_dolls : ℕ := 10

/-- Number of dolls after which a break is taken -/
def dolls_per_break : ℕ := 3

/-- Start time in minutes after midnight -/
def start_time : ℕ := 10 * 60

theorem doll_completion_time :
  let total_craft_time := num_dolls * craft_time
  let total_breaks := (num_dolls / dolls_per_break) * break_time
  let total_time := total_craft_time + total_breaks
  let completion_time := (start_time + total_time) % (24 * 60)
  completion_time = 5 * 60 :=
by sorry

end doll_completion_time_l548_54841


namespace function_zero_range_l548_54864

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

def has_exactly_one_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0

theorem function_zero_range (a : ℝ) :
  has_exactly_one_zero (f a) 1 (Real.exp 2) →
  a ∈ Set.Iic (-(Real.exp 4) / 2) ∪ {-2 * Real.exp 1} :=
sorry

end function_zero_range_l548_54864


namespace equation_solution_l548_54836

theorem equation_solution : 
  ∃ x : ℚ, (x ≠ -2) ∧ (8 * x / (x + 2) - 5 / (x + 2) = 2 / (x + 2)) ∧ (x = 7 / 8) :=
by sorry

end equation_solution_l548_54836


namespace some_athletes_not_honor_society_l548_54813

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Athlete : U → Prop)
variable (Disciplined : U → Prop)
variable (HonorSocietyMember : U → Prop)

-- Define the conditions
variable (h1 : ∃ x, Athlete x ∧ ¬Disciplined x)
variable (h2 : ∀ x, HonorSocietyMember x → Disciplined x)

-- Theorem to prove
theorem some_athletes_not_honor_society :
  ∃ x, Athlete x ∧ ¬HonorSocietyMember x :=
sorry

end some_athletes_not_honor_society_l548_54813


namespace total_pigeons_l548_54802

def initial_pigeons : ℕ := 1
def joined_pigeons : ℕ := 1

theorem total_pigeons : initial_pigeons + joined_pigeons = 2 := by
  sorry

end total_pigeons_l548_54802


namespace pokemon_game_l548_54838

theorem pokemon_game (n : ℕ) : 
  (∃ (m : ℕ), 
    n * m + 11 * (m + 6) = n^2 + 3*n - 2 ∧ 
    m > 0 ∧ 
    (m + 6) > 0) → 
  n = 9 :=
by sorry

end pokemon_game_l548_54838


namespace right_triangle_set_l548_54872

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def set_A : Fin 3 → ℕ := ![4, 5, 6]
def set_B : Fin 3 → ℕ := ![12, 16, 20]
def set_C : Fin 3 → ℕ := ![5, 10, 13]
def set_D : Fin 3 → ℕ := ![8, 40, 41]

/-- The main theorem --/
theorem right_triangle_set :
  (¬ is_right_triangle (set_A 0) (set_A 1) (set_A 2)) ∧
  (is_right_triangle (set_B 0) (set_B 1) (set_B 2)) ∧
  (¬ is_right_triangle (set_C 0) (set_C 1) (set_C 2)) ∧
  (¬ is_right_triangle (set_D 0) (set_D 1) (set_D 2)) :=
by sorry

end right_triangle_set_l548_54872


namespace largest_number_with_digit_sum_13_l548_54888

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 7 ∨ d = 1

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digit_sum_13 :
  ∀ n : ℕ, is_valid_number n → digit_sum n = 13 → n ≤ 7111111 :=
by sorry

end largest_number_with_digit_sum_13_l548_54888


namespace existence_of_comparable_indices_l548_54837

theorem existence_of_comparable_indices
  (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by
  sorry

end existence_of_comparable_indices_l548_54837


namespace smallest_integer_with_remainder_one_l548_54811

theorem smallest_integer_with_remainder_one : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 6 = 1) ∧ 
  (n % 7 = 1) ∧ 
  (n % 8 = 1) ∧ 
  (∀ m : ℕ, m > 1 → m % 6 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
  (n > 120) ∧ 
  (n < 209) := by
  sorry

end smallest_integer_with_remainder_one_l548_54811


namespace shelby_poster_purchase_l548_54885

/-- Calculates the number of posters Shelby can buy given the problem conditions --/
def calculate_posters (initial_amount coupon_value tax_rate : ℚ)
  (book1_cost book2_cost bookmark_cost pencils_cost notebook_cost poster_cost : ℚ)
  (discount_rate1 discount_rate2 : ℚ)
  (discount_threshold1 discount_threshold2 : ℚ) : ℕ :=
  sorry

/-- Theorem stating that Shelby can buy exactly 4 posters --/
theorem shelby_poster_purchase :
  let initial_amount : ℚ := 60
  let book1_cost : ℚ := 15
  let book2_cost : ℚ := 9
  let bookmark_cost : ℚ := 3.5
  let pencils_cost : ℚ := 4.8
  let notebook_cost : ℚ := 6.2
  let poster_cost : ℚ := 6
  let discount_rate1 : ℚ := 0.15
  let discount_rate2 : ℚ := 0.10
  let discount_threshold1 : ℚ := 40
  let discount_threshold2 : ℚ := 25
  let coupon_value : ℚ := 5
  let tax_rate : ℚ := 0.08
  calculate_posters initial_amount coupon_value tax_rate
    book1_cost book2_cost bookmark_cost pencils_cost notebook_cost poster_cost
    discount_rate1 discount_rate2 discount_threshold1 discount_threshold2 = 4 :=
by sorry

end shelby_poster_purchase_l548_54885


namespace books_second_shop_l548_54859

def books_first_shop : ℕ := 65
def cost_first_shop : ℕ := 1080
def cost_second_shop : ℕ := 840
def average_price : ℕ := 16

theorem books_second_shop :
  (cost_first_shop + cost_second_shop) / average_price - books_first_shop = 55 := by
  sorry

end books_second_shop_l548_54859


namespace surface_area_comparison_l548_54871

/-- Given a cube, cylinder, and sphere with equal volumes, their surface areas satisfy S₃ < S₂ < S₁ -/
theorem surface_area_comparison 
  (V : ℝ) 
  (h_V_pos : V > 0) 
  (S₁ : ℝ) 
  (S₂ : ℝ) 
  (S₃ : ℝ) 
  (h_S₁ : S₁ = Real.rpow (216 * V^2) (1/3))
  (h_S₂ : S₂ = Real.rpow (54 * π * V^2) (1/3))
  (h_S₃ : S₃ = Real.rpow (36 * π * V^2) (1/3)) :
  S₃ < S₂ ∧ S₂ < S₁ :=
by sorry

end surface_area_comparison_l548_54871


namespace article_cost_price_l548_54894

/-- Given an article with a 15% markup, sold at Rs. 456 after a 26.570048309178745% discount,
    prove that the cost price of the article is Rs. 540. -/
theorem article_cost_price (markup_percentage : ℝ) (selling_price : ℝ) (discount_percentage : ℝ)
    (h1 : markup_percentage = 15)
    (h2 : selling_price = 456)
    (h3 : discount_percentage = 26.570048309178745) :
    ∃ (cost_price : ℝ),
      cost_price * (1 + markup_percentage / 100) * (1 - discount_percentage / 100) = selling_price ∧
      cost_price = 540 := by
  sorry

end article_cost_price_l548_54894


namespace scientific_notation_of_70_62_million_l548_54882

/-- Proves that 70.62 million is equal to 7.062 × 10^7 in scientific notation -/
theorem scientific_notation_of_70_62_million :
  (70.62 * 1000000 : ℝ) = 7.062 * (10 ^ 7) := by
  sorry

end scientific_notation_of_70_62_million_l548_54882


namespace discount_clinic_savings_l548_54850

theorem discount_clinic_savings (normal_fee : ℝ) : 
  (normal_fee - 2 * (0.3 * normal_fee) = 80) → normal_fee = 200 := by
  sorry

end discount_clinic_savings_l548_54850


namespace ninth_term_of_arithmetic_sequence_l548_54810

/-- Given an arithmetic sequence where the first term is 4/7 and the seventeenth term is 5/6,
    the ninth term is equal to 59/84. -/
theorem ninth_term_of_arithmetic_sequence (a : ℕ → ℚ) 
  (h1 : a 1 = 4/7)
  (h17 : a 17 = 5/6)
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) :
  a 9 = 59/84 := by
  sorry

end ninth_term_of_arithmetic_sequence_l548_54810


namespace ellipse_k_range_l548_54891

/-- Represents an ellipse equation in the form x^2 + ky^2 = 2 --/
structure EllipseEquation where
  k : ℝ

/-- Predicate to check if the equation represents a valid ellipse with foci on the y-axis --/
def is_valid_ellipse (e : EllipseEquation) : Prop :=
  0 < e.k ∧ e.k < 1

/-- Theorem stating the range of k for a valid ellipse with foci on the y-axis --/
theorem ellipse_k_range (e : EllipseEquation) : 
  (∃ (x y : ℝ), x^2 + e.k * y^2 = 2) ∧ 
  (∃ (c : ℝ), c ≠ 0 ∧ ∀ (x y : ℝ), x^2 + e.k * y^2 = 2 → x^2 + (y - c)^2 = x^2 + (y + c)^2) 
  ↔ is_valid_ellipse e :=
sorry

end ellipse_k_range_l548_54891


namespace power_of_power_l548_54821

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l548_54821


namespace equation_not_quadratic_l548_54853

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def equation (y : ℝ) : ℝ := 3 * y * (y - 1) - y * (3 * y + 1)

theorem equation_not_quadratic : ¬ is_quadratic equation := by
  sorry

end equation_not_quadratic_l548_54853


namespace wholesale_price_calculation_l548_54808

theorem wholesale_price_calculation (retail_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  retail_price = 132 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.2 →
  ∃ wholesale_price : ℝ,
    wholesale_price = 99 ∧
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) :=
by sorry

end wholesale_price_calculation_l548_54808


namespace garden_dimensions_l548_54845

/-- Represents the dimensions of a rectangular garden. -/
structure GardenDimensions where
  length : ℝ
  breadth : ℝ

/-- Checks if the given dimensions satisfy the garden constraints. -/
def satisfiesConstraints (d : GardenDimensions) : Prop :=
  d.length = (3 / 5) * d.breadth ∧
  d.length * d.breadth = 600 ∧
  2 * (d.length + d.breadth) ≤ 120

/-- Theorem stating the correct dimensions of the garden. -/
theorem garden_dimensions :
  ∃ (d : GardenDimensions),
    satisfiesConstraints d ∧
    d.length = 6 * Real.sqrt 10 ∧
    d.breadth = 10 * Real.sqrt 10 :=
by sorry

end garden_dimensions_l548_54845


namespace perpendicular_vectors_l548_54861

/-- Given vectors a and b, prove that a is perpendicular to b iff n = -3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) :
  a = (3, 2) → b.1 = 2 → a.1 * b.1 + a.2 * b.2 = 0 ↔ b.2 = -3 := by
  sorry

end perpendicular_vectors_l548_54861


namespace juice_consumption_l548_54832

theorem juice_consumption (total_juice : ℚ) (sam_fraction : ℚ) (alex_fraction : ℚ) :
  total_juice = 3/4 ∧ sam_fraction = 1/2 ∧ alex_fraction = 1/4 →
  sam_fraction * total_juice + alex_fraction * total_juice = 9/16 := by
  sorry

end juice_consumption_l548_54832


namespace final_position_of_A_l548_54873

-- Define the initial position of point A
def initial_position : ℝ := -3

-- Define the movement in the positive direction
def movement : ℝ := 4.5

-- Theorem to prove the final position of point A
theorem final_position_of_A : initial_position + movement = 1.5 := by
  sorry

end final_position_of_A_l548_54873


namespace equation_represents_three_lines_not_all_intersecting_l548_54857

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^2 * (x + y + 2) = y^2 * (x + y + 2)

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = -x
def line2 (x y : ℝ) : Prop := y = x
def line3 (x y : ℝ) : Prop := y = -x - 2

-- Theorem statement
theorem equation_represents_three_lines_not_all_intersecting :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    (∀ x y, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y)) ∧
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
    (¬ (line1 p3.1 p3.2 ∧ line2 p3.1 p3.2 ∧ line3 p3.1 p3.2)) :=
by
  sorry


end equation_represents_three_lines_not_all_intersecting_l548_54857


namespace triangle_angle_c_l548_54898

theorem triangle_angle_c (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧ 
  3 * Real.sin A + 4 * Real.cos B = 6 ∧
  4 * Real.sin B + 3 * Real.cos A = 1 →
  C = π / 6 := by sorry

end triangle_angle_c_l548_54898


namespace line_slope_angle_l548_54863

theorem line_slope_angle (a : ℝ) : 
  (∃ (x y : ℝ), a * x + y + 2 = 0) → -- line equation
  (Real.tan (45 * Real.pi / 180) = -1 / a) → -- slope angle is 45°
  a = -1 := by
sorry

end line_slope_angle_l548_54863


namespace product_of_recurring_decimal_and_seven_l548_54843

theorem product_of_recurring_decimal_and_seven :
  ∃ (x : ℚ), (∃ (n : ℕ), x = (456 : ℚ) / (10^3 - 1)) ∧ 7 * x = 355 / 111 := by
  sorry

end product_of_recurring_decimal_and_seven_l548_54843


namespace min_value_of_f_l548_54870

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

-- Define the closed interval [-1, 0]
def I : Set ℝ := {x | -1 ≤ x ∧ x ≤ 0}

-- Theorem statement
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ I ∧ f x = -1 ∧ ∀ (y : ℝ), y ∈ I → f y ≥ f x :=
sorry

end min_value_of_f_l548_54870


namespace not_p_sufficient_not_necessary_q_l548_54858

-- Define the propositions p and q
def p (m : ℝ) : Prop := m ≥ (1/4 : ℝ)
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + x + m = 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

-- Theorem statement
theorem not_p_sufficient_not_necessary_q :
  sufficient_not_necessary (¬∀ m, p m) (∀ m, q m) :=
sorry

end not_p_sufficient_not_necessary_q_l548_54858


namespace product_integers_exist_l548_54852

theorem product_integers_exist : ∃ (a b c : ℝ), 
  (¬ ∃ (n : ℤ), a = n) ∧ 
  (¬ ∃ (n : ℤ), b = n) ∧ 
  (¬ ∃ (n : ℤ), c = n) ∧ 
  (∃ (n : ℤ), a * b = n) ∧ 
  (∃ (n : ℤ), b * c = n) ∧ 
  (∃ (n : ℤ), c * a = n) ∧ 
  (∃ (n : ℤ), a * b * c = n) := by
sorry

end product_integers_exist_l548_54852


namespace probability_penny_dime_halfdollar_heads_l548_54895

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (dime : CoinFlip)
  (quarter : CoinFlip)
  (half_dollar : CoinFlip)
  (dollar : CoinFlip)

/-- The total number of possible outcomes when flipping six coins -/
def total_outcomes : ℕ := 64

/-- Predicate for the desired outcome (penny, dime, and half-dollar are heads) -/
def desired_outcome (cs : CoinSet) : Prop :=
  cs.penny = CoinFlip.Heads ∧ cs.dime = CoinFlip.Heads ∧ cs.half_dollar = CoinFlip.Heads

/-- The number of outcomes satisfying the desired condition -/
def favorable_outcomes : ℕ := 8

/-- Theorem stating the probability of the desired outcome -/
theorem probability_penny_dime_halfdollar_heads :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8 := by
  sorry


end probability_penny_dime_halfdollar_heads_l548_54895


namespace line_equation_with_slope_and_area_l548_54867

theorem line_equation_with_slope_and_area (x y : ℝ) :
  ∃ (b : ℝ), (3 * x - 4 * y + 12 * b = 0 ∨ 3 * x - 4 * y - 12 * b = 0) ∧
  (3 / 4 : ℝ) = (y - 0) / (x - 0) ∧
  6 = (1 / 2) * |0 - x| * |0 - y| :=
sorry

end line_equation_with_slope_and_area_l548_54867


namespace infinite_sum_equals_one_fourth_l548_54826

/-- The sum of the series (3^n) / (1 + 3^n + 3^(n+1) + 3^(2n+1)) from n=1 to infinity equals 1/4 -/
theorem infinite_sum_equals_one_fourth :
  (∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))) = 1/4 :=
sorry

end infinite_sum_equals_one_fourth_l548_54826


namespace square_root_23_minus_one_expression_l548_54896

theorem square_root_23_minus_one_expression : 
  let x : ℝ := Real.sqrt 23 - 1
  x^2 + 2*x + 2 = 24 := by sorry

end square_root_23_minus_one_expression_l548_54896


namespace smallest_n_divisible_l548_54816

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 
  24 ∣ n^2 ∧ 
  1024 ∣ n^3 ∧ 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(24 ∣ m^2 ∧ 1024 ∣ m^3)) → 
  n = 48 := by
  sorry

end smallest_n_divisible_l548_54816


namespace harolds_marbles_l548_54899

/-- Given that Harold has 100 marbles, keeps 20 for himself, and shares the rest evenly among 5 friends,
    prove that each friend receives 16 marbles. -/
theorem harolds_marbles (total : ℕ) (kept : ℕ) (friends : ℕ) 
    (h1 : total = 100) 
    (h2 : kept = 20)
    (h3 : friends = 5) :
    (total - kept) / friends = 16 := by
  sorry

end harolds_marbles_l548_54899


namespace firecracker_sales_properties_l548_54892

/-- Electronic firecracker sales model -/
structure FirecrackerSales where
  cost : ℝ
  demand : ℝ → ℝ
  price_range : Set ℝ

/-- Daily profit function -/
def daily_profit (model : FirecrackerSales) (x : ℝ) : ℝ :=
  (x - model.cost) * model.demand x

theorem firecracker_sales_properties (model : FirecrackerSales) 
  (h_cost : model.cost = 80)
  (h_demand : ∀ x, model.demand x = -2 * x + 320)
  (h_range : model.price_range = {x | 80 ≤ x ∧ x ≤ 160}) :
  (∀ x ∈ model.price_range, daily_profit model x = -2 * x^2 + 480 * x - 25600) ∧
  (∃ max_price ∈ model.price_range, 
    (∀ x ∈ model.price_range, daily_profit model x ≤ daily_profit model max_price) ∧
    daily_profit model max_price = 3200 ∧
    max_price = 120) ∧
  (∃ price ∈ model.price_range, daily_profit model price = 2400 ∧ price = 100) := by
  sorry

end firecracker_sales_properties_l548_54892


namespace sofa_payment_difference_l548_54880

/-- Given that Joan and Karl bought sofas with a total cost of $600,
    Joan paid $230, and twice Joan's payment is more than Karl's,
    prove that the difference between twice Joan's payment and Karl's is $90. -/
theorem sofa_payment_difference :
  ∀ (joan_payment karl_payment : ℕ),
  joan_payment + karl_payment = 600 →
  joan_payment = 230 →
  2 * joan_payment > karl_payment →
  2 * joan_payment - karl_payment = 90 := by
sorry

end sofa_payment_difference_l548_54880


namespace complex_equality_l548_54817

theorem complex_equality (z : ℂ) : z = -15/8 + 5/4*I → Complex.abs (z - 2*I) = Complex.abs (z + 4) ∧ Complex.abs (z - 2*I) = Complex.abs (z + I) := by
  sorry

end complex_equality_l548_54817


namespace eesha_travel_time_l548_54860

/-- Eesha's usual time to reach her office -/
def usual_time : ℝ := 60

/-- The additional time taken when driving slower -/
def additional_time : ℝ := 20

/-- The ratio of slower speed to usual speed -/
def speed_ratio : ℝ := 0.75

theorem eesha_travel_time :
  usual_time = 60 ∧
  additional_time = usual_time / speed_ratio - usual_time :=
by sorry

end eesha_travel_time_l548_54860


namespace probability_is_half_l548_54887

/-- An equilateral triangle divided by two medians -/
structure TriangleWithMedians where
  /-- The number of regions formed by drawing two medians in an equilateral triangle -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- The total number of regions is 6 -/
  h_total : total_regions = 6
  /-- The number of shaded regions is 3 -/
  h_shaded : shaded_regions = 3

/-- The probability of a point landing in a shaded region -/
def probability (t : TriangleWithMedians) : ℚ :=
  t.shaded_regions / t.total_regions

theorem probability_is_half (t : TriangleWithMedians) :
  probability t = 1 / 2 := by
  sorry

end probability_is_half_l548_54887


namespace shoe_box_problem_l548_54829

theorem shoe_box_problem (pairs : ℕ) (prob : ℝ) (total : ℕ) : 
  pairs = 100 →
  prob = 0.005025125628140704 →
  (pairs : ℝ) / ((total * (total - 1)) / 2) = prob →
  total = 200 :=
sorry

end shoe_box_problem_l548_54829


namespace fold_sum_value_l548_54849

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a fold on graph paper -/
structure Fold :=
  (p1 : Point)
  (p2 : Point)
  (q1 : Point)
  (q2 : Point)

/-- The sum of the coordinates of the unknown point in a fold -/
def fold_sum (f : Fold) : ℝ := f.q2.x + f.q2.y

/-- The theorem stating the sum of coordinates of the unknown point -/
theorem fold_sum_value (f : Fold) 
  (h1 : f.p1 = ⟨0, 2⟩)
  (h2 : f.p2 = ⟨4, 0⟩)
  (h3 : f.q1 = ⟨7, 3⟩) :
  fold_sum f = 6.8 := by
  sorry

end fold_sum_value_l548_54849


namespace smallest_n_for_terminating_decimal_l548_54883

/-- A function that checks if a fraction is a terminating decimal -/
def isTerminatingDecimal (numerator : ℕ) (denominator : ℕ) : Prop :=
  ∃ (a b : ℕ), denominator = 2^a * 5^b

/-- The smallest positive integer n such that n/(n+150) is a terminating decimal -/
theorem smallest_n_for_terminating_decimal : 
  (∀ n : ℕ, n > 0 → n < 50 → ¬(isTerminatingDecimal n (n + 150))) ∧ 
  (isTerminatingDecimal 50 200) := by
  sorry

#check smallest_n_for_terminating_decimal

end smallest_n_for_terminating_decimal_l548_54883


namespace quadratic_function_properties_range_of_g_l548_54886

-- Define the quadratic function f
def f (x : ℝ) : ℝ := -2 * x^2 - 4 * x

-- State the theorem
theorem quadratic_function_properties :
  -- The vertex of f is (-1, 2)
  (f (-1) = 2 ∧ ∀ x, f x ≤ f (-1)) ∧
  -- f passes through the origin
  f 0 = 0 ∧
  -- The range of f(2x) is (-∞, 0)
  (∀ y, (∃ x, f (2*x) = y) ↔ y < 0) := by
sorry

-- Define g as f(2x)
def g (x : ℝ) : ℝ := f (2*x)

-- Additional theorem for the range of g
theorem range_of_g :
  (∀ y, (∃ x, g x = y) ↔ y < 0) := by
sorry

end quadratic_function_properties_range_of_g_l548_54886


namespace sum_of_A_and_B_l548_54822

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Checks if a row contains 2, 3, and 4 -/
def valid_row (g : Grid) (row : Fin 3) : Prop :=
  ({2, 3, 4} : Finset Nat) = {g row 0, g row 1, g row 2}

/-- Checks if a column contains 2, 3, and 4 -/
def valid_column (g : Grid) (col : Fin 3) : Prop :=
  ({2, 3, 4} : Finset Nat) = {g 0 col, g 1 col, g 2 col}

/-- Checks if the grid satisfies all conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 3, valid_row g row) ∧
  (∀ col : Fin 3, valid_column g col) ∧
  g 0 0 = 2 ∧
  g 1 1 = 3

theorem sum_of_A_and_B (g : Grid) (h : valid_grid g) : g 2 0 + g 0 2 = 6 := by
  sorry

end sum_of_A_and_B_l548_54822


namespace triangle_special_angle_l548_54855

theorem triangle_special_angle (a b c : ℝ) (h : (a + 2*b + c)*(a + b - c - 2) = 4*a*b) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))
  C = π / 3 := by sorry

end triangle_special_angle_l548_54855


namespace triangle_area_l548_54874

/-- The area of the triangle formed by two lines intersecting at (3,3) with slopes 1/3 and 3, 
    and a third line x + y = 12 -/
theorem triangle_area : ℝ := by
  -- Define the lines
  let line1 : ℝ → ℝ := fun x ↦ (1/3) * x + 2
  let line2 : ℝ → ℝ := fun x ↦ 3 * x - 6
  let line3 : ℝ → ℝ := fun x ↦ 12 - x

  -- Define the intersection points
  let A : ℝ × ℝ := (3, 3)
  let B : ℝ × ℝ := (4.5, 7.5)
  let C : ℝ × ℝ := (7.5, 4.5)

  -- Calculate the area of the triangle
  have area_formula : ℝ :=
    (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

  -- Assert that the area is equal to 9
  have area_eq_9 : area_formula = 9 := by sorry

  exact 9

end triangle_area_l548_54874


namespace susan_board_game_movement_l548_54862

theorem susan_board_game_movement (total_spaces : ℕ) (first_move : ℕ) (third_move : ℕ) (sent_back : ℕ) (remaining_spaces : ℕ) : 
  total_spaces = 48 →
  first_move = 8 →
  third_move = 6 →
  sent_back = 5 →
  remaining_spaces = 37 →
  first_move + third_move + remaining_spaces + sent_back = total_spaces →
  ∃ (second_move : ℕ), second_move = 28 := by
sorry

end susan_board_game_movement_l548_54862


namespace problem_statement_l548_54846

theorem problem_statement (a : ℤ) 
  (h1 : 0 ≤ a) (h2 : a < 13) 
  (h3 : (51^2018 + a) % 13 = 0) : 
  a = 12 := by sorry

end problem_statement_l548_54846


namespace total_scoops_l548_54879

def flour_cups : ℚ := 3
def sugar_cups : ℚ := 2
def flour_scoop : ℚ := 1/4
def sugar_scoop : ℚ := 1/3

theorem total_scoops : 
  (flour_cups / flour_scoop + sugar_cups / sugar_scoop : ℚ) = 18 := by
  sorry

end total_scoops_l548_54879


namespace penguin_fish_distribution_penguin_fish_distribution_proof_l548_54835

theorem penguin_fish_distribution (total_penguins : ℕ) 
  (emperor_ratio adelie_ratio : ℕ) 
  (emperor_fish adelie_fish : ℚ) 
  (fish_constraint : ℕ) : Prop :=
  let emperor_count := (total_penguins * emperor_ratio) / (emperor_ratio + adelie_ratio)
  let adelie_count := (total_penguins * adelie_ratio) / (emperor_ratio + adelie_ratio)
  let total_fish_needed := (emperor_count : ℚ) * emperor_fish + (adelie_count : ℚ) * adelie_fish
  total_penguins = 48 ∧ 
  emperor_ratio = 3 ∧ 
  adelie_ratio = 5 ∧ 
  emperor_fish = 3/2 ∧ 
  adelie_fish = 2 ∧ 
  fish_constraint = 115 →
  total_fish_needed ≤ fish_constraint

-- Proof
theorem penguin_fish_distribution_proof : 
  penguin_fish_distribution 48 3 5 (3/2) 2 115 := by
  sorry

end penguin_fish_distribution_penguin_fish_distribution_proof_l548_54835


namespace repeating_decimal_equals_fraction_l548_54847

-- Define the repeating decimal 0.454545...
def repeating_decimal : ℚ := 45 / 99

-- State the theorem
theorem repeating_decimal_equals_fraction : repeating_decimal = 5 / 11 := by
  sorry

end repeating_decimal_equals_fraction_l548_54847


namespace second_printer_theorem_l548_54833

/-- The time (in minutes) it takes for the second printer to print 800 flyers -/
def second_printer_time (first_printer_time second_printer_time combined_time : ℚ) : ℚ :=
  30 / 7

/-- Given the specifications of two printers, proves that the second printer
    takes 30/7 minutes to print 800 flyers -/
theorem second_printer_theorem (first_printer_time combined_time : ℚ) 
  (h1 : first_printer_time = 10)
  (h2 : combined_time = 3) :
  second_printer_time first_printer_time (second_printer_time first_printer_time (30/7) combined_time) combined_time = 30 / 7 := by
  sorry

#check second_printer_theorem

end second_printer_theorem_l548_54833


namespace cos_2alpha_plus_3pi_over_5_l548_54812

theorem cos_2alpha_plus_3pi_over_5 (α : ℝ) 
  (h : Real.sin (π / 5 - α) = 1 / 3) : 
  Real.cos (2 * α + 3 * π / 5) = - 7 / 9 := by
  sorry

end cos_2alpha_plus_3pi_over_5_l548_54812


namespace prob_receive_one_out_of_two_prob_receive_at_least_ten_l548_54809

/-- The probability of receiving a red envelope for each recipient -/
def prob_receive : ℚ := 1 / 3

/-- The probability of not receiving a red envelope for each recipient -/
def prob_not_receive : ℚ := 2 / 3

/-- The number of recipients -/
def num_recipients : ℕ := 3

/-- The number of red envelopes sent in the first scenario -/
def num_envelopes_1 : ℕ := 2

/-- The number of red envelopes sent in the second scenario -/
def num_envelopes_2 : ℕ := 3

/-- The amounts in the red envelopes for the second scenario -/
def envelope_amounts : List ℚ := [5, 5, 10]

/-- Theorem 1: Probability of receiving exactly one envelope out of two -/
theorem prob_receive_one_out_of_two :
  let p := prob_receive
  let q := prob_not_receive
  p * q + q * p = 4 / 9 := by sorry

/-- Theorem 2: Probability of receiving at least 10 yuan out of three envelopes -/
theorem prob_receive_at_least_ten :
  let p := prob_receive
  let q := prob_not_receive
  p^2 * q + 2 * p^2 * q + p^3 = 11 / 27 := by sorry

end prob_receive_one_out_of_two_prob_receive_at_least_ten_l548_54809


namespace san_antonio_bound_passes_two_austin_bound_l548_54839

/-- Represents the direction of travel for a bus -/
inductive Direction
  | AustinToSanAntonio
  | SanAntonioToAustin

/-- Represents a bus schedule -/
structure BusSchedule where
  direction : Direction
  departureInterval : ℕ  -- in hours
  departureOffset : ℕ    -- in hours

/-- Represents the bus system between Austin and San Antonio -/
structure BusSystem where
  travelTime : ℕ
  austinToSanAntonioSchedule : BusSchedule
  sanAntonioToAustinSchedule : BusSchedule

/-- Counts the number of buses passed during a journey -/
def countPassedBuses (system : BusSystem) : ℕ :=
  sorry

/-- The main theorem stating that a San Antonio-bound bus passes exactly 2 Austin-bound buses -/
theorem san_antonio_bound_passes_two_austin_bound :
  ∀ (system : BusSystem),
    system.travelTime = 3 ∧
    system.austinToSanAntonioSchedule.direction = Direction.AustinToSanAntonio ∧
    system.austinToSanAntonioSchedule.departureInterval = 2 ∧
    system.austinToSanAntonioSchedule.departureOffset = 0 ∧
    system.sanAntonioToAustinSchedule.direction = Direction.SanAntonioToAustin ∧
    system.sanAntonioToAustinSchedule.departureInterval = 2 ∧
    system.sanAntonioToAustinSchedule.departureOffset = 1 →
    countPassedBuses system = 2 :=
  sorry

end san_antonio_bound_passes_two_austin_bound_l548_54839


namespace max_n_value_l548_54818

theorem max_n_value (a b c : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c) 
  (h3 : ∀ a b c, a > b → b > c → 1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c)) :
  n ≤ 2 :=
sorry

end max_n_value_l548_54818


namespace boys_ratio_in_class_l548_54827

theorem boys_ratio_in_class (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  (n : ℚ) / (n + m : ℚ) = 2 / 5 ↔ 
  (n : ℚ) / (n + m : ℚ) = 2 / 3 * (m : ℚ) / (n + m : ℚ) :=
by sorry

end boys_ratio_in_class_l548_54827


namespace choose_three_from_eight_l548_54814

theorem choose_three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end choose_three_from_eight_l548_54814


namespace fish_pond_population_l548_54897

/-- Proves that given the conditions of the fish tagging problem, the approximate number of fish in the pond is 1250 -/
theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) 
  (h1 : initial_tagged = 50)
  (h2 : second_catch = 50)
  (h3 : tagged_in_second = 2)
  (h4 : (initial_tagged : ℚ) / total_fish = (tagged_in_second : ℚ) / second_catch) :
  total_fish = 1250 := by
  sorry

#check fish_pond_population

end fish_pond_population_l548_54897


namespace triangle_rds_area_l548_54803

/-- The area of a triangle RDS with given coordinates and perpendicular sides -/
theorem triangle_rds_area (k : ℝ) : 
  let R : ℝ × ℝ := (0, 15)
  let D : ℝ × ℝ := (3, 15)
  let S : ℝ × ℝ := (0, k)
  -- RD is perpendicular to RS (implied by coordinates)
  (45 - 3 * k) / 2 = (1 / 2) * 3 * (15 - k) := by sorry

end triangle_rds_area_l548_54803


namespace kimmie_earnings_l548_54800

theorem kimmie_earnings (kimmie_earnings : ℚ) : 
  (kimmie_earnings / 2 + (2 / 3 * kimmie_earnings) / 2 = 375) → 
  kimmie_earnings = 450 := by
  sorry

end kimmie_earnings_l548_54800


namespace unique_function_satisfying_condition_l548_54889

open Function Real

theorem unique_function_satisfying_condition :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2 ∧ f = id := by
  sorry

end unique_function_satisfying_condition_l548_54889


namespace star_shape_perimeter_star_shape_perimeter_is_4pi_l548_54878

/-- The perimeter of a star-like shape formed by arcs of six unit circles arranged in a regular hexagon configuration --/
theorem star_shape_perimeter : ℝ :=
  let n : ℕ := 6  -- number of coins
  let r : ℝ := 1  -- radius of each coin
  let angle_sum : ℝ := 2 * Real.pi  -- sum of internal angles of a hexagon
  4 * Real.pi

/-- Proof that the perimeter of the star-like shape is 4π --/
theorem star_shape_perimeter_is_4pi : star_shape_perimeter = 4 * Real.pi := by
  sorry

end star_shape_perimeter_star_shape_perimeter_is_4pi_l548_54878


namespace planning_committee_selection_l548_54893

theorem planning_committee_selection (n : ℕ) : 
  (n.choose 2 = 21) → (n.choose 4 = 35) := by
  sorry

end planning_committee_selection_l548_54893


namespace sixth_angle_measure_l548_54820

/-- The sum of internal angles of a hexagon is 720 degrees -/
def hexagon_angle_sum : ℝ := 720

/-- The sum of the five known angles in the hexagon -/
def known_angles_sum : ℝ := 130 + 100 + 105 + 115 + 95

/-- Theorem: In a hexagon where five of the internal angles measure 130°, 100°, 105°, 115°, and 95°,
    the measure of the sixth angle is 175°. -/
theorem sixth_angle_measure :
  hexagon_angle_sum - known_angles_sum = 175 := by sorry

end sixth_angle_measure_l548_54820


namespace geometric_sequence_product_l548_54868

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 4 * a 8 = 4 →
  a 5 * a 6 * a 7 = 8 := by
  sorry

end geometric_sequence_product_l548_54868


namespace power_greater_than_linear_l548_54854

theorem power_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2*n + 1 := by
  sorry

end power_greater_than_linear_l548_54854


namespace cars_without_features_l548_54819

theorem cars_without_features (total : ℕ) (steering : ℕ) (windows : ℕ) (both : ℕ)
  (h1 : total = 65)
  (h2 : steering = 45)
  (h3 : windows = 25)
  (h4 : both = 17) :
  total - (steering + windows - both) = 12 := by
  sorry

end cars_without_features_l548_54819


namespace equation_solution_l548_54851

theorem equation_solution : 
  ∃ (x : ℝ), x ≥ 0 ∧ 2021 * x = 2022 * (x^2021)^(1/2022) - 1 ∧ x = 1 :=
by sorry

end equation_solution_l548_54851


namespace eighth_pentagon_shaded_fraction_l548_54869

/-- Triangular number sequence -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Pentagonal number sequence -/
def pentagonal (n : ℕ) : ℕ := (3 * n^2 - n) / 2

/-- Total sections in the nth pentagon -/
def total_sections (n : ℕ) : ℕ := n^2

/-- Shaded sections in the nth pentagon -/
def shaded_sections (n : ℕ) : ℕ :=
  if n % 2 = 1 then triangular (n / 2 + 1)
  else pentagonal (n / 2)

theorem eighth_pentagon_shaded_fraction :
  (shaded_sections 8 : ℚ) / (total_sections 8 : ℚ) = 11 / 32 := by
  sorry

end eighth_pentagon_shaded_fraction_l548_54869


namespace abc_divisibility_problem_l548_54856

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end abc_divisibility_problem_l548_54856


namespace shaded_to_unshaded_ratio_is_two_to_one_l548_54884

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length 3 -/
structure Square where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Represents the configuration of points and lines in the square -/
structure SquareConfiguration where
  square : Square
  t : Point
  u : Point
  v : Point
  w : Point

/-- The ratio of shaded to unshaded area in the square configuration -/
def shadedToUnshadedRatio (config : SquareConfiguration) : ℚ := 2

/-- Theorem stating that the ratio of shaded to unshaded area is 2:1 -/
theorem shaded_to_unshaded_ratio_is_two_to_one (config : SquareConfiguration) :
  shadedToUnshadedRatio config = 2 := by
  sorry

end shaded_to_unshaded_ratio_is_two_to_one_l548_54884


namespace parallel_iff_no_common_points_l548_54881

-- Define a structure for a 3D space
structure Space3D where
  Point : Type
  Line : Type
  Plane : Type

-- Define the concept of a line being parallel to a plane
def parallel (S : Space3D) (l : S.Line) (p : S.Plane) : Prop := sorry

-- Define the concept of a line having no common points with a plane
def no_common_points (S : Space3D) (l : S.Line) (p : S.Plane) : Prop := sorry

-- Theorem statement
theorem parallel_iff_no_common_points (S : Space3D) (a : S.Line) (M : S.Plane) :
  parallel S a M ↔ no_common_points S a M := by sorry

end parallel_iff_no_common_points_l548_54881


namespace books_on_shelves_l548_54876

/-- The number of ways to place n distinct books onto k shelves with no empty shelf -/
def place_books (n : ℕ) (k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

theorem books_on_shelves :
  place_books 10 3 * Nat.factorial 10 = 55980 * Nat.factorial 10 :=
by sorry

end books_on_shelves_l548_54876


namespace toy_value_proof_l548_54830

theorem toy_value_proof (total_toys : ℕ) (total_worth : ℕ) (special_toy_value : ℕ) :
  total_toys = 9 →
  total_worth = 52 →
  special_toy_value = 12 →
  ∃ (other_toy_value : ℕ),
    other_toy_value * (total_toys - 1) + special_toy_value = total_worth ∧
    other_toy_value = 5 := by
  sorry

end toy_value_proof_l548_54830
