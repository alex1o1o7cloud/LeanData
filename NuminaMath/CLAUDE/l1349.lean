import Mathlib

namespace quadratic_inequality_solution_set_l1349_134976

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (1 + a) * x + a > 0}
  (a > 1 → S = {x : ℝ | x > a ∨ x < 1}) ∧
  (a = 1 → S = {x : ℝ | x ≠ 1}) ∧
  (a < 1 → S = {x : ℝ | x > 1 ∨ x < a}) := by
  sorry

end quadratic_inequality_solution_set_l1349_134976


namespace modulo_equivalence_l1349_134908

theorem modulo_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 54126 ≡ n [ZMOD 23] ∧ n = 13 := by
  sorry

end modulo_equivalence_l1349_134908


namespace jack_reading_pages_l1349_134972

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 13

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 67

/-- The total number of pages Jack needs to read -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem jack_reading_pages : total_pages = 871 := by
  sorry

end jack_reading_pages_l1349_134972


namespace line_translation_upwards_l1349_134991

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically --/
def translateLine (l : Line) (c : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + c }

/-- The equation of a line in slope-intercept form --/
def lineEquation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem line_translation_upwards 
  (original : Line) 
  (c : ℝ) 
  (h : c > 0) : 
  ∀ x y : ℝ, lineEquation original x y ↔ lineEquation (translateLine original c) x (y + c) :=
by sorry

end line_translation_upwards_l1349_134991


namespace combination_minus_permutation_l1349_134958

-- Define combination
def combination (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define permutation
def permutation (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem combination_minus_permutation : combination 7 4 - permutation 5 2 = 15 := by
  sorry

end combination_minus_permutation_l1349_134958


namespace perfect_square_condition_l1349_134956

theorem perfect_square_condition (n : ℕ+) : 
  ∃ (m : ℕ), 2^n.val + 12^n.val + 2011^n.val = m^2 ↔ n = 1 := by
  sorry

end perfect_square_condition_l1349_134956


namespace five_digit_sum_l1349_134973

def sum_of_digits (x : ℕ) : ℕ := 1 + 3 + 4 + 6 + x

def number_of_permutations : ℕ := 120  -- This is A₅⁵

theorem five_digit_sum (x : ℕ) :
  sum_of_digits x * number_of_permutations = 2640 → x = 8 := by
  sorry

end five_digit_sum_l1349_134973


namespace max_value_of_4x_plus_3y_l1349_134941

theorem max_value_of_4x_plus_3y (x y : ℝ) : 
  x^2 + y^2 = 10*x + 8*y + 10 → (4*x + 3*y ≤ 70) ∧ ∃ x y, x^2 + y^2 = 10*x + 8*y + 10 ∧ 4*x + 3*y = 70 := by
  sorry

end max_value_of_4x_plus_3y_l1349_134941


namespace inequality_proof_l1349_134997

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 9*y + 3*z) * (x + 4*y + 2*z) * (2*x + 12*y + 9*z) ≥ 1029 * x * y * z := by
  sorry

end inequality_proof_l1349_134997


namespace negation_equivalence_l1349_134946

-- Define the curve
def is_curve (m : ℕ) (x y : ℝ) : Prop := x^2 / m + y^2 = 1

-- Define what it means for the curve to be an ellipse (this is a placeholder definition)
def is_ellipse (m : ℕ) : Prop := ∃ x y : ℝ, is_curve m x y

-- The theorem to prove
theorem negation_equivalence :
  (¬ ∃ m : ℕ, is_ellipse m) ↔ (∀ m : ℕ, ¬ is_ellipse m) := by sorry

end negation_equivalence_l1349_134946


namespace monotonic_decreasing_interval_f_l1349_134961

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) := x * Real.log x

-- State the theorem
theorem monotonic_decreasing_interval_f :
  ∀ x : ℝ, x > 0 → (StrictMonoOn f (Set.Ioo 0 (Real.exp (-1)))) ∧
  (∀ y : ℝ, y > Real.exp (-1) → ¬ StrictMonoOn f (Set.Ioo 0 y)) :=
by sorry

end monotonic_decreasing_interval_f_l1349_134961


namespace parabola_minimum_point_l1349_134937

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the points A, B, C
def A : ℝ × ℝ := (-1, -3)
def B : ℝ × ℝ := (4, 2)
def C : ℝ × ℝ := (0, 2)

-- Define the theorem
theorem parabola_minimum_point (a b c : ℝ) :
  ∃ (m n : ℝ),
    -- The parabola passes through points A, B, C
    parabola a b c A.1 = A.2 ∧
    parabola a b c B.1 = B.2 ∧
    parabola a b c C.1 = C.2 ∧
    -- P(m, n) is on the axis of symmetry
    m = -b / (2 * a) ∧
    -- P(m, n) minimizes PA + PC
    ∀ (x y : ℝ), x = m → parabola a b c x = y →
      (Real.sqrt ((x - A.1)^2 + (y - A.2)^2) +
       Real.sqrt ((x - C.1)^2 + (y - C.2)^2)) ≥
      (Real.sqrt ((m - A.1)^2 + (n - A.2)^2) +
       Real.sqrt ((m - C.1)^2 + (n - C.2)^2)) →
    -- The y-coordinate of P is 0
    n = 0 :=
by
  sorry

end parabola_minimum_point_l1349_134937


namespace rectangle_area_error_percentage_l1349_134984

theorem rectangle_area_error_percentage (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  let actual_area := L * W
  let measured_area := 1.10 * L * 0.95 * W
  let error_percentage := (measured_area - actual_area) / actual_area * 100
  error_percentage = 4.5 := by sorry

end rectangle_area_error_percentage_l1349_134984


namespace skating_time_calculation_l1349_134971

/-- The number of days Gage skated for each duration -/
def days_per_duration : ℕ := 4

/-- The duration of skating in minutes for the first set of days -/
def duration1 : ℕ := 80

/-- The duration of skating in minutes for the second set of days -/
def duration2 : ℕ := 105

/-- The desired average skating time in minutes per day -/
def desired_average : ℕ := 100

/-- The total number of days, including the day to be calculated -/
def total_days : ℕ := 2 * days_per_duration + 1

/-- The required skating time on the last day to achieve the desired average -/
def required_time : ℕ := 160

theorem skating_time_calculation :
  (days_per_duration * duration1 + days_per_duration * duration2 + required_time) / total_days = desired_average := by
  sorry

end skating_time_calculation_l1349_134971


namespace geometric_sequence_general_term_l1349_134910

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_general_term 
  (a : ℕ → ℚ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 3/2) 
  (h_S3 : (a 1) + (a 2) + (a 3) = 9/2) :
  (∃ n : ℕ, a n = 3/2 * (-2)^(n-1)) ∨ (∀ n : ℕ, a n = 3/2) :=
sorry

end geometric_sequence_general_term_l1349_134910


namespace special_function_properties_l1349_134988

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (4 - x) = 0) ∧ (∀ x, f (x + 2) - f (x - 2) = 0)

/-- Theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (h : SpecialFunction f) :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x) := by
  sorry


end special_function_properties_l1349_134988


namespace point_on_graph_l1349_134947

theorem point_on_graph (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a * x - 1
  f 1 = 1 := by sorry

end point_on_graph_l1349_134947


namespace cos_angle_F₁PF₂_l1349_134996

-- Define the ellipse and hyperbola
def is_on_ellipse (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1
def is_on_hyperbola (x y : ℝ) : Prop := x^2/3 - y^2 = 1

-- Define the common foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the common point P
structure CommonPoint where
  x : ℝ
  y : ℝ
  on_ellipse : is_on_ellipse x y
  on_hyperbola : is_on_hyperbola x y

-- Theorem statement
theorem cos_angle_F₁PF₂ (P : CommonPoint) : 
  let PF₁ := (F₁.1 - P.x, F₁.2 - P.y)
  let PF₂ := (F₂.1 - P.x, F₂.2 - P.y)
  let dot_product := PF₁.1 * PF₂.1 + PF₁.2 * PF₂.2
  let magnitude_PF₁ := Real.sqrt (PF₁.1^2 + PF₁.2^2)
  let magnitude_PF₂ := Real.sqrt (PF₂.1^2 + PF₂.2^2)
  dot_product / (magnitude_PF₁ * magnitude_PF₂) = 1/3 :=
by sorry

end cos_angle_F₁PF₂_l1349_134996


namespace sqrt_sum_equals_two_sqrt_31_l1349_134919

theorem sqrt_sum_equals_two_sqrt_31 :
  Real.sqrt (24 - 10 * Real.sqrt 5) + Real.sqrt (24 + 10 * Real.sqrt 5) = 2 * Real.sqrt 31 := by
  sorry

end sqrt_sum_equals_two_sqrt_31_l1349_134919


namespace puzzle_missing_pieces_l1349_134938

theorem puzzle_missing_pieces 
  (total_pieces : ℕ) 
  (border_pieces : ℕ) 
  (trevor_pieces : ℕ) 
  (joe_multiplier : ℕ) : 
  total_pieces = 500 →
  border_pieces = 75 →
  trevor_pieces = 105 →
  joe_multiplier = 3 →
  total_pieces - border_pieces - (trevor_pieces + joe_multiplier * trevor_pieces) = 5 :=
by
  sorry

#check puzzle_missing_pieces

end puzzle_missing_pieces_l1349_134938


namespace exists_m_divisible_by_1988_l1349_134975

def f (x : ℕ) : ℕ := 3 * x + 2

def iterate (n : ℕ) (f : ℕ → ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate n f x)

theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ, (1988 : ℕ) ∣ (iterate 100 f m) := by
  sorry

end exists_m_divisible_by_1988_l1349_134975


namespace symmetric_about_origin_l1349_134959

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function g: ℝ → ℝ is even if g(-x) = g(x) for all x ∈ ℝ -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

/-- A function v: ℝ → ℝ is symmetric about the origin if v(-x) = -v(x) for all x ∈ ℝ -/
def SymmetricAboutOrigin (v : ℝ → ℝ) : Prop := ∀ x, v (-x) = -v x

/-- Main theorem: If f is odd and g is even, then v(x) = f(x)|g(x)| is symmetric about the origin -/
theorem symmetric_about_origin (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  SymmetricAboutOrigin (fun x ↦ f x * |g x|) := by
  sorry

end symmetric_about_origin_l1349_134959


namespace target_hit_probability_l1349_134921

theorem target_hit_probability (hit_rate_A hit_rate_B : ℚ) 
  (h1 : hit_rate_A = 4/5)
  (h2 : hit_rate_B = 5/6) :
  1 - (1 - hit_rate_A) * (1 - hit_rate_B) = 29/30 := by
  sorry

end target_hit_probability_l1349_134921


namespace smallest_three_digit_non_divisor_l1349_134902

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def factorial (n : ℕ) : ℕ := Nat.factorial n

def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_three_digit_non_divisor :
  ∀ k : ℕ, 100 ≤ k → k < 101 →
    is_divisor (sum_of_squares k) (factorial k) →
    ¬ is_divisor (sum_of_squares 101) (factorial 101) :=
by sorry

end smallest_three_digit_non_divisor_l1349_134902


namespace tangent_line_problem_l1349_134933

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_line_problem (a : ℝ) : 
  (f_derivative a 1 * (2 - 1) + f a 1 = 7) → a = 1 := by
  sorry

end tangent_line_problem_l1349_134933


namespace total_oranges_l1349_134974

/-- Given 3.0 children and 1.333333333 oranges per child, prove that the total number of oranges is 4. -/
theorem total_oranges (num_children : ℝ) (oranges_per_child : ℝ) 
  (h1 : num_children = 3.0) 
  (h2 : oranges_per_child = 1.333333333) : 
  num_children * oranges_per_child = 4 := by
  sorry

end total_oranges_l1349_134974


namespace cube_root_8000_simplification_l1349_134950

theorem cube_root_8000_simplification :
  ∃ (a b : ℕ+), (a : ℝ) * (b : ℝ)^(1/3) = 8000^(1/3) ∧
                a = 20 ∧ b = 1 ∧
                ∀ (c d : ℕ+), (c : ℝ) * (d : ℝ)^(1/3) = 8000^(1/3) → d ≥ b :=
by sorry

end cube_root_8000_simplification_l1349_134950


namespace polygon_sides_from_angle_sum_l1349_134904

theorem polygon_sides_from_angle_sum (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 900 → (n - 2) * 180 = sum_angles → n = 7 := by
  sorry

end polygon_sides_from_angle_sum_l1349_134904


namespace min_additional_coins_l1349_134934

/-- The number of friends Alex has -/
def num_friends : ℕ := 12

/-- The initial number of coins Alex has -/
def initial_coins : ℕ := 63

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The minimum number of additional coins needed -/
def additional_coins_needed : ℕ := sum_first_n num_friends - initial_coins

theorem min_additional_coins :
  additional_coins_needed = 15 :=
sorry

end min_additional_coins_l1349_134934


namespace car_travel_distance_l1349_134990

/-- Proves that a car can travel 500 miles before refilling given specific journey conditions. -/
theorem car_travel_distance (fuel_cost : ℝ) (journey_distance : ℝ) (food_ratio : ℝ) (total_spent : ℝ)
  (h1 : fuel_cost = 45)
  (h2 : journey_distance = 2000)
  (h3 : food_ratio = 3/5)
  (h4 : total_spent = 288) :
  journey_distance / (total_spent / ((1 + food_ratio) * fuel_cost)) = 500 := by
  sorry

end car_travel_distance_l1349_134990


namespace quadratic_roots_difference_l1349_134925

theorem quadratic_roots_difference (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 - m*x₁ + 8 = 0 ∧ 
   x₂^2 - m*x₂ + 8 = 0 ∧ 
   |x₁ - x₂| = Real.sqrt 84) →
  m ≤ 2 * Real.sqrt 29 :=
by sorry

end quadratic_roots_difference_l1349_134925


namespace ellipse_m_range_l1349_134980

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the square perimeter condition
def square_perimeter_condition (a b : ℝ) : Prop := 
  ∃ (c : ℝ), a^2 = b^2 + c^2 ∧ 4 * a = 4 * Real.sqrt 2 ∧ b = c

-- Define the line l
def line_l (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the symmetric point D
def symmetric_point (m : ℝ) (x y : ℝ) : Prop := x = 0 ∧ y = -m

-- Define the condition for D being inside the circle with EF as diameter
def inside_circle_condition (m : ℝ) : Prop :=
  ∀ k : ℝ, (m * Real.sqrt (4 * k^2 + 1))^2 < 2 * (1 + k^2) * (2 * k^2 + 1 - m^2)

-- Main theorem
theorem ellipse_m_range :
  ∀ a b m : ℝ,
  a > b ∧ b > 0 ∧ m > 0 ∧
  square_perimeter_condition a b ∧
  inside_circle_condition m →
  0 < m ∧ m < Real.sqrt 3 / 3 :=
sorry

end ellipse_m_range_l1349_134980


namespace rectangle_area_l1349_134917

/-- A rectangle divided into four identical squares with a given perimeter has a specific area -/
theorem rectangle_area (perimeter : ℝ) (h_perimeter : perimeter = 160) :
  let side_length := perimeter / 10
  let length := 4 * side_length
  let width := side_length
  let area := length * width
  area = 1024 := by sorry

end rectangle_area_l1349_134917


namespace arthur_walked_four_point_five_miles_l1349_134945

/-- The distance Arthur walked in miles -/
def arthur_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 4.5 miles -/
theorem arthur_walked_four_point_five_miles :
  arthur_distance 8 10 (1/4) = 4.5 := by
sorry

end arthur_walked_four_point_five_miles_l1349_134945


namespace nancy_carrot_nv_l1349_134930

/-- Calculates the total nutritional value of carrots based on given conditions -/
def total_nutritional_value (initial_carrots : ℕ) (kept_carrots : ℕ) (new_seeds : ℕ) 
  (growth_factor : ℕ) (base_nv : ℝ) (nv_per_cm : ℝ) (growth_cm : ℝ) : ℝ :=
  let new_carrots := new_seeds * growth_factor
  let total_carrots := initial_carrots - kept_carrots + new_carrots
  let good_carrots := total_carrots - (total_carrots / 3)
  let new_carrot_nv := new_carrots * (base_nv + nv_per_cm * growth_cm)
  let kept_carrot_nv := kept_carrots * base_nv
  new_carrot_nv + kept_carrot_nv

/-- Theorem stating that the total nutritional value of Nancy's carrots is 92 -/
theorem nancy_carrot_nv : 
  total_nutritional_value 12 2 5 3 1 0.5 12 = 92 := by
  sorry

end nancy_carrot_nv_l1349_134930


namespace thirtieth_term_of_sequence_l1349_134983

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 30th term of the arithmetic sequence with first term 3 and common difference 6 is 177 -/
theorem thirtieth_term_of_sequence : arithmetic_sequence 3 6 30 = 177 := by
  sorry

end thirtieth_term_of_sequence_l1349_134983


namespace arithmetic_calculation_l1349_134952

theorem arithmetic_calculation : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := by
  sorry

end arithmetic_calculation_l1349_134952


namespace group_size_proof_l1349_134970

theorem group_size_proof (n : ℕ) (f m : ℕ) : 
  f = 8 → 
  m + f = n → 
  (n - f : ℚ) / n - (n - m : ℚ) / n = 36 / 100 → 
  n = 25 := by
sorry

end group_size_proof_l1349_134970


namespace andy_candy_canes_l1349_134994

/-- The number of candy canes Andy got from his parents -/
def parents_candy : ℕ := sorry

/-- The number of candy canes Andy got from teachers -/
def teachers_candy : ℕ := 3 * 4

/-- The ratio of candy canes to cavities -/
def candy_to_cavity_ratio : ℕ := 4

/-- The number of cavities Andy got -/
def cavities : ℕ := 16

/-- The fraction of additional candy canes Andy buys -/
def bought_candy_fraction : ℚ := 1 / 7

theorem andy_candy_canes :
  parents_candy = 44 ∧
  parents_candy + teachers_candy + (parents_candy + teachers_candy : ℚ) * bought_candy_fraction = cavities * candy_to_cavity_ratio := by sorry

end andy_candy_canes_l1349_134994


namespace largest_three_digit_product_l1349_134927

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def has_no_repeated_prime_factors (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → (n % (p * p) ≠ 0)

def is_valid_triple (x y : ℕ) : Prop :=
  x < 10 ∧ y < 10 ∧ x ≠ y ∧ is_prime (10 * x + y)

theorem largest_three_digit_product :
  ∃ m x y : ℕ,
    m = x * y * (10 * x + y) ∧
    is_valid_triple x y ∧
    has_no_repeated_prime_factors m ∧
    m < 1000 ∧
    (∀ m' x' y' : ℕ,
      m' = x' * y' * (10 * x' + y') →
      is_valid_triple x' y' →
      has_no_repeated_prime_factors m' →
      m' < 1000 →
      m' ≤ m) ∧
    m = 777 :=
sorry

end largest_three_digit_product_l1349_134927


namespace binomial_coefficient_20_19_l1349_134913

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_coefficient_20_19_l1349_134913


namespace parabola_vertex_l1349_134995

/-- The parabola defined by y = -x^2 + 3 has its vertex at (0, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -x^2 + 3 → (0, 3) = (x, y) ∨ ∃ z, y < -z^2 + 3 := by
  sorry

end parabola_vertex_l1349_134995


namespace circle_and_line_problem_l1349_134916

-- Define the circles and line
def C₁ (x y : ℝ) := (x + 3)^2 + y^2 = 4
def C₂ (x y : ℝ) := (x + 1)^2 + (y + 2)^2 = 4
def symmetry_line (x y : ℝ) := x - y + 1 = 0

-- Define points
def A : ℝ × ℝ := (0, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the theorem
theorem circle_and_line_problem :
  -- Given conditions
  (∀ x y : ℝ, C₁ x y ↔ C₂ (y - 1) (-x - 1)) →  -- Symmetry condition
  (∃ k : ℝ, ∀ x : ℝ, C₁ x (k * x + 3)) →      -- Line l intersects C₁
  -- Conclusion
  ((∀ x y : ℝ, C₁ x y ↔ (x + 3)^2 + y^2 = 4) ∧
   (∃ M N : ℝ × ℝ, 
     (C₁ M.1 M.2 ∧ C₁ N.1 N.2) ∧
     (M.2 = 2 * M.1 + 3 ∨ M.2 = 3 * M.1 + 3) ∧
     (N.2 = 2 * N.1 + 3 ∨ N.2 = 3 * N.1 + 3) ∧
     (M.1 * N.1 + M.2 * N.2 = 7/5))) :=
by sorry


end circle_and_line_problem_l1349_134916


namespace badminton_players_count_l1349_134924

/-- A sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  tennis : ℕ
  both : ℕ
  neither : ℕ
  tennis_le_total : tennis ≤ total
  both_le_tennis : both ≤ tennis
  neither_le_total : neither ≤ total

/-- The number of members who play badminton in the sports club -/
def badminton_players (club : SportsClub) : ℕ :=
  club.total - club.tennis + club.both - club.neither

/-- Theorem stating the number of badminton players in the specific club scenario -/
theorem badminton_players_count (club : SportsClub) 
  (h_total : club.total = 30)
  (h_tennis : club.tennis = 19)
  (h_both : club.both = 8)
  (h_neither : club.neither = 2) :
  badminton_players club = 17 := by
  sorry

end badminton_players_count_l1349_134924


namespace integral_sin4_cos4_3x_l1349_134935

theorem integral_sin4_cos4_3x (x : ℝ) : 
  ∫ x in (0 : ℝ)..(2 * Real.pi), (Real.sin (3 * x))^4 * (Real.cos (3 * x))^4 = (3 * Real.pi) / 64 := by
  sorry

end integral_sin4_cos4_3x_l1349_134935


namespace expression_value_l1349_134901

theorem expression_value : 4^3 - 2 * 4^2 + 2 * 4 - 1 = 39 := by
  sorry

end expression_value_l1349_134901


namespace inequality_squared_not_always_true_l1349_134905

theorem inequality_squared_not_always_true : ¬ ∀ x y : ℝ, x < y → x^2 < y^2 := by
  sorry

end inequality_squared_not_always_true_l1349_134905


namespace sum_of_digits_0_to_99_l1349_134926

/-- The sum of all digits of integers from 0 to 99 inclusive -/
def sum_of_digits : ℕ := 900

/-- Theorem stating that the sum of all digits of integers from 0 to 99 inclusive is 900 -/
theorem sum_of_digits_0_to_99 : sum_of_digits = 900 := by
  sorry

end sum_of_digits_0_to_99_l1349_134926


namespace kolya_tolya_ages_l1349_134951

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (is_valid : tens < 10 ∧ ones < 10)

/-- Calculates the numeric value of an Age -/
def Age.value (a : Age) : Nat :=
  10 * a.tens + a.ones

/-- Reverses the digits of an Age -/
def Age.reverse (a : Age) : Age :=
  ⟨a.ones, a.tens, a.is_valid.symm⟩

theorem kolya_tolya_ages :
  ∃ (kolya_age tolya_age : Age),
    -- Kolya is older than Tolya
    kolya_age.value > tolya_age.value ∧
    -- Both ages are less than 100
    kolya_age.value < 100 ∧ tolya_age.value < 100 ∧
    -- Reversing Kolya's age gives Tolya's age
    kolya_age.reverse = tolya_age ∧
    -- The difference of squares is a perfect square
    ∃ (k : Nat), (kolya_age.value ^ 2 - tolya_age.value ^ 2 = k ^ 2) ∧
    -- Kolya is 65 and Tolya is 56
    kolya_age.value = 65 ∧ tolya_age.value = 56 := by
  sorry

end kolya_tolya_ages_l1349_134951


namespace f_has_root_in_interval_l1349_134992

def f (x : ℝ) := x^3 - 3*x - 3

theorem f_has_root_in_interval :
  ∃ (x : ℝ), x ∈ Set.Ioo 2 3 ∧ f x = 0 := by
  sorry

end f_has_root_in_interval_l1349_134992


namespace initial_number_proof_l1349_134998

theorem initial_number_proof (x : ℝ) : x - 70 = 70 + 40 → x = 180 := by
  sorry

end initial_number_proof_l1349_134998


namespace inheritance_distribution_correct_l1349_134932

/-- Represents the distribution of an inheritance among three sons and a hospital. -/
structure InheritanceDistribution where
  eldest : ℕ
  middle : ℕ
  youngest : ℕ
  hospital : ℕ

/-- Checks if the given distribution satisfies the inheritance conditions. -/
def satisfies_conditions (d : InheritanceDistribution) : Prop :=
  -- Total inheritance is $1320
  d.eldest + d.middle + d.youngest + d.hospital = 1320 ∧
  -- If hospital's portion went to eldest son
  d.eldest + d.hospital = d.middle + d.youngest ∧
  -- If hospital's portion went to middle son
  d.middle + d.hospital = 2 * (d.eldest + d.youngest) ∧
  -- If hospital's portion went to youngest son
  d.youngest + d.hospital = 3 * (d.eldest + d.middle)

/-- The theorem stating that the given distribution satisfies all conditions. -/
theorem inheritance_distribution_correct : 
  satisfies_conditions ⟨55, 275, 385, 605⟩ := by
  sorry

end inheritance_distribution_correct_l1349_134932


namespace linear_equation_properties_l1349_134986

-- Define the linear equation
def linear_equation (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem linear_equation_properties :
  ∃ (k b : ℝ),
    (linear_equation k b (-3) = -9) ∧
    (linear_equation k b 0 = -3) ∧
    (k = 2 ∧ b = -3) ∧
    (∀ x, linear_equation k b x ≥ 0 → x ≥ 1.5) ∧
    (∀ x, -1 ≤ x ∧ x < 2 → -5 ≤ linear_equation k b x ∧ linear_equation k b x < 1) :=
by
  sorry

end linear_equation_properties_l1349_134986


namespace y_axis_intersection_l1349_134955

/-- The quadratic function f(x) = 3x^2 - 4x + 5 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 5

/-- The y-axis intersection point of f(x) is (0, 5) -/
theorem y_axis_intersection :
  f 0 = 5 :=
by sorry

end y_axis_intersection_l1349_134955


namespace probability_exactly_one_instrument_l1349_134936

theorem probability_exactly_one_instrument (total : ℕ) (at_least_one_fraction : ℚ) (two_or_more : ℕ) :
  total = 800 →
  at_least_one_fraction = 2 / 5 →
  two_or_more = 96 →
  (((at_least_one_fraction * total) - two_or_more) / total : ℚ) = 28 / 100 := by
  sorry

end probability_exactly_one_instrument_l1349_134936


namespace additional_water_needed_l1349_134920

/-- Represents the capacity of a tank in liters -/
def TankCapacity : ℝ := 1000

/-- Represents the volume of water in the first tank in liters -/
def FirstTankVolume : ℝ := 300

/-- Represents the volume of water in the second tank in liters -/
def SecondTankVolume : ℝ := 450

/-- Represents the percentage of the second tank that is filled -/
def SecondTankPercentage : ℝ := 0.45

theorem additional_water_needed : 
  let remaining_first := TankCapacity - FirstTankVolume
  let remaining_second := TankCapacity - SecondTankVolume
  remaining_first + remaining_second = 1250 := by sorry

end additional_water_needed_l1349_134920


namespace hash_difference_l1349_134981

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem hash_difference : hash 4 2 - hash 2 4 = -8 := by
  sorry

end hash_difference_l1349_134981


namespace tiger_tree_trunk_length_l1349_134912

/-- The length of a fallen tree trunk over which a tiger runs --/
theorem tiger_tree_trunk_length (tiger_length : ℝ) (grass_time : ℝ) (trunk_time : ℝ)
  (h_length : tiger_length = 5)
  (h_grass : grass_time = 1)
  (h_trunk : trunk_time = 5) :
  tiger_length * trunk_time = 25 := by
  sorry

end tiger_tree_trunk_length_l1349_134912


namespace simplify_fraction_l1349_134979

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l1349_134979


namespace equation_solution_l1349_134978

theorem equation_solution : ∃ r : ℝ, (24 - 5 = 3 * r + 7) ∧ (r = 4) := by sorry

end equation_solution_l1349_134978


namespace min_value_fraction_l1349_134939

theorem min_value_fraction (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ m : ℝ, m = -1 - Real.sqrt 2 ∧ ∀ z, z = (2*x*y)/(x+y+1) → m ≤ z :=
sorry

end min_value_fraction_l1349_134939


namespace function_difference_theorem_l1349_134914

theorem function_difference_theorem (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 3 * x^2 - 1/x + 5) →
  (∀ x, g x = 2 * x^2 - k) →
  f 3 - g 3 = 6 →
  k = -23/3 := by sorry

end function_difference_theorem_l1349_134914


namespace ad_arrangement_count_l1349_134962

/-- The number of ways to arrange n items, taking r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways to arrange 6 advertisements (4 commercial and 2 public service) 
    where the 2 public service ads cannot be consecutive -/
def ad_arrangements : ℕ :=
  permutations 4 4 * permutations 5 2

theorem ad_arrangement_count : 
  ad_arrangements = permutations 4 4 * permutations 5 2 := by
  sorry

end ad_arrangement_count_l1349_134962


namespace min_area_enclosed_l1349_134940

/-- The function f(x) = 3 - x^2 --/
def f (x : ℝ) : ℝ := 3 - x^2

/-- A point on the graph of f --/
structure PointOnGraph where
  x : ℝ
  y : ℝ
  on_graph : y = f x

/-- The area enclosed by tangents and x-axis --/
def enclosed_area (A B : PointOnGraph) : ℝ :=
  sorry -- Definition of the area calculation

/-- Theorem: Minimum area enclosed by tangents and x-axis --/
theorem min_area_enclosed (A B : PointOnGraph) 
    (h_opposite : A.x * B.x < 0) : -- A and B are on opposite sides of y-axis
  ∃ (min_area : ℝ), min_area = 8 ∧ ∀ (P Q : PointOnGraph), 
    P.x * Q.x < 0 → enclosed_area P Q ≥ min_area := by
  sorry

end min_area_enclosed_l1349_134940


namespace a_range_l1349_134900

theorem a_range (a : ℝ) (h1 : a < 9 * a^3 - 11 * a) (h2 : 9 * a^3 - 11 * a < |a|) (h3 : a < 0) :
  -2 * Real.sqrt 3 / 3 < a ∧ a < -Real.sqrt 10 / 3 := by
  sorry

end a_range_l1349_134900


namespace tan_X_equals_four_l1349_134948

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ) where
  -- Angle Y is 90°
  right_angle : Y = 90
  -- Length of side YZ
  yz_length : Z - Y = 4
  -- Length of side XZ
  xz_length : Z - X = Real.sqrt 17

-- Theorem statement
theorem tan_X_equals_four {X Y Z : ℝ} (t : Triangle X Y Z) : Real.tan X = 4 := by
  sorry

end tan_X_equals_four_l1349_134948


namespace tangent_and_below_and_two_zeros_l1349_134922

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x + 1

def tangent_line (x y : ℝ) : Prop := (1 - a) * x - y = 0

def g (x : ℝ) : ℝ := 1/2 * a * x^2 - (f a x + a * x)

theorem tangent_and_below_and_two_zeros :
  (∀ y, tangent_line a 1 y ↔ y = f a 1) ∧
  (∀ x > 0, x ≠ 1 → f a x < (1 - a) * x) ∧
  (∃ x₁ x₂, x₁ < x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ ∀ x, x ≠ x₁ → x ≠ x₂ → g a x ≠ 0) ↔
  0 < a ∧ a < Real.exp 1 :=
sorry

end tangent_and_below_and_two_zeros_l1349_134922


namespace average_decrease_l1349_134957

theorem average_decrease (n : ℕ) (initial_avg : ℚ) (new_obs : ℚ) : 
  n = 6 → 
  initial_avg = 11 → 
  new_obs = 4 → 
  (n * initial_avg + new_obs) / (n + 1) = initial_avg - 1 := by
sorry

end average_decrease_l1349_134957


namespace sixth_grade_homework_forgetfulness_l1349_134903

theorem sixth_grade_homework_forgetfulness
  (group_a_size : ℕ)
  (group_b_size : ℕ)
  (group_a_forget_rate : ℚ)
  (group_b_forget_rate : ℚ)
  (h1 : group_a_size = 20)
  (h2 : group_b_size = 80)
  (h3 : group_a_forget_rate = 1/5)
  (h4 : group_b_forget_rate = 3/20)
  : (((group_a_size * group_a_forget_rate + group_b_size * group_b_forget_rate) /
     (group_a_size + group_b_size)) : ℚ) = 4/25 :=
by sorry

end sixth_grade_homework_forgetfulness_l1349_134903


namespace dogsled_race_speed_l1349_134977

/-- Given two teams racing on a 300-mile course, where one team finishes 3 hours faster
    and has an average speed 5 mph greater than the other, prove that the slower team's
    average speed is 20 mph. -/
theorem dogsled_race_speed (course_length : ℝ) (time_difference : ℝ) (speed_difference : ℝ)
  (h1 : course_length = 300)
  (h2 : time_difference = 3)
  (h3 : speed_difference = 5) :
  let speed_B := (course_length / (course_length / (20 + speed_difference) + time_difference))
  speed_B = 20 := by sorry

end dogsled_race_speed_l1349_134977


namespace f_2_eq_0_f_positive_solution_set_l1349_134964

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - (3 - a)*x + 2*(1 - a)

-- Theorem for f(2) = 0
theorem f_2_eq_0 (a : ℝ) : f a 2 = 0 := by sorry

-- Define the solution set for f(x) > 0
def solution_set (a : ℝ) : Set ℝ :=
  if a < -1 then {x | x < 2 ∨ x > 1 - a}
  else if a = -1 then {x | x < 2 ∨ x > 2}
  else {x | x < 1 - a ∨ x > 2}

-- Theorem for the solution set of f(x) > 0
theorem f_positive_solution_set (a : ℝ) :
  {x : ℝ | f a x > 0} = solution_set a := by sorry

end f_2_eq_0_f_positive_solution_set_l1349_134964


namespace f_positive_m_range_l1349_134923

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for the solution set of f(x) > 0
theorem f_positive (x : ℝ) : f x > 0 ↔ x < -1/3 ∨ x > 3 := by sorry

-- Theorem for the range of m
theorem m_range (m : ℝ) : 
  (∃ x₀ : ℝ, f x₀ + 2*m^2 < 4*m) ↔ -1/2 < m ∧ m < 5/2 := by sorry

end f_positive_m_range_l1349_134923


namespace quadratic_real_root_l1349_134989

theorem quadratic_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end quadratic_real_root_l1349_134989


namespace problem_solution_l1349_134906

theorem problem_solution (y : ℝ) (h : y + Real.sqrt (y^2 - 4) + 1 / (y - Real.sqrt (y^2 - 4)) = 12) :
  y^2 + Real.sqrt (y^4 - 4) + 1 / (y^2 - Real.sqrt (y^4 - 4)) = 200 / 9 := by
  sorry

end problem_solution_l1349_134906


namespace equation_solutions_l1349_134982

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 6 ∧ 
    ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = 3/4 ∧ 
    ∀ x : ℝ, 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = y₁ ∨ x = y₂) :=
by sorry

end equation_solutions_l1349_134982


namespace smallest_zack_students_correct_l1349_134931

/-- Represents the number of students in a group for each tutor -/
structure TutorGroup where
  zack : Nat
  karen : Nat
  julie : Nat

/-- Represents the ratio of students for each tutor -/
structure TutorRatio where
  zack : Nat
  karen : Nat
  julie : Nat

/-- The smallest number of students Zack can have given the conditions -/
def smallestZackStudents (g : TutorGroup) (r : TutorRatio) : Nat :=
  630

theorem smallest_zack_students_correct (g : TutorGroup) (r : TutorRatio) :
  g.zack = 14 →
  g.karen = 10 →
  g.julie = 15 →
  r.zack = 3 →
  r.karen = 2 →
  r.julie = 5 →
  smallestZackStudents g r = 630 ∧
  smallestZackStudents g r % g.zack = 0 ∧
  (smallestZackStudents g r / r.zack * r.karen) % g.karen = 0 ∧
  (smallestZackStudents g r / r.zack * r.julie) % g.julie = 0 ∧
  ∀ n : Nat, n < smallestZackStudents g r →
    (n % g.zack = 0 ∧ (n / r.zack * r.karen) % g.karen = 0 ∧ (n / r.zack * r.julie) % g.julie = 0) →
    False :=
by
  sorry

#check smallest_zack_students_correct

end smallest_zack_students_correct_l1349_134931


namespace sum_equation_proof_l1349_134960

theorem sum_equation_proof (N : ℕ) : 
  985 + 987 + 989 + 991 + 993 + 995 + 997 + 999 = 8000 - N → N = 64 := by
  sorry

end sum_equation_proof_l1349_134960


namespace melissa_family_theorem_l1349_134967

/-- The number of Melissa's daughters and granddaughters who have no daughters -/
def num_without_daughters (total_descendants : ℕ) (num_daughters : ℕ) (daughters_with_children : ℕ) (granddaughters_per_daughter : ℕ) : ℕ :=
  (num_daughters - daughters_with_children) + (daughters_with_children * granddaughters_per_daughter)

theorem melissa_family_theorem :
  let total_descendants := 50
  let num_daughters := 10
  let daughters_with_children := num_daughters / 2
  let granddaughters_per_daughter := 4
  num_without_daughters total_descendants num_daughters daughters_with_children granddaughters_per_daughter = 45 := by
sorry

end melissa_family_theorem_l1349_134967


namespace rectangle_horizontal_length_l1349_134985

/-- The horizontal length of a rectangle with perimeter 54 cm and horizontal length 3 cm longer than vertical length is 15 cm. -/
theorem rectangle_horizontal_length :
  ∀ (h v : ℝ), 
    (2 * h + 2 * v = 54) →  -- Perimeter is 54 cm
    (h = v + 3) →           -- Horizontal length is 3 cm longer than vertical length
    h = 15 := by            -- Horizontal length is 15 cm
  sorry

end rectangle_horizontal_length_l1349_134985


namespace inequality_proof_l1349_134915

theorem inequality_proof (x y z t : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (ht : 0 < t ∧ t < 1) : 
  Real.sqrt (x^2 + (1-t)^2) + Real.sqrt (y^2 + (1-x)^2) + 
  Real.sqrt (z^2 + (1-y)^2) + Real.sqrt (t^2 + (1-z)^2) < 4 :=
by sorry

end inequality_proof_l1349_134915


namespace candidate_a_support_l1349_134929

/-- Represents the percentage of registered voters in each category -/
structure VoterDistribution :=
  (democrats : ℝ)
  (republicans : ℝ)
  (independents : ℝ)
  (undecided : ℝ)

/-- Represents the percentage of voters in each category supporting candidate A -/
structure SupportDistribution :=
  (democrats : ℝ)
  (republicans : ℝ)
  (independents : ℝ)
  (undecided : ℝ)

/-- Calculates the total percentage of registered voters supporting candidate A -/
def calculateTotalSupport (vd : VoterDistribution) (sd : SupportDistribution) : ℝ :=
  vd.democrats * sd.democrats +
  vd.republicans * sd.republicans +
  vd.independents * sd.independents +
  vd.undecided * sd.undecided

theorem candidate_a_support :
  let vd : VoterDistribution := {
    democrats := 0.45,
    republicans := 0.30,
    independents := 0.20,
    undecided := 0.05
  }
  let sd : SupportDistribution := {
    democrats := 0.75,
    republicans := 0.25,
    independents := 0.50,
    undecided := 0.50
  }
  calculateTotalSupport vd sd = 0.5375 := by
  sorry

end candidate_a_support_l1349_134929


namespace erased_numbers_sum_l1349_134954

/-- Represents a sequence of consecutive odd numbers -/
def OddSequence : ℕ → ℕ := λ n => 2 * n - 1

/-- Sum of the first n odd numbers -/
def SumOfOddNumbers (n : ℕ) : ℕ := n * n

theorem erased_numbers_sum (first_segment_sum second_segment_sum : ℕ) 
  (h1 : first_segment_sum = 961) 
  (h2 : second_segment_sum = 1001) : 
  ∃ (k1 k2 : ℕ), 
    k1 < k2 ∧ 
    SumOfOddNumbers (k1 - 1) = first_segment_sum ∧
    SumOfOddNumbers (k2 - 1) - SumOfOddNumbers k1 = second_segment_sum ∧
    OddSequence k1 + OddSequence k2 = 154 := by
  sorry

end erased_numbers_sum_l1349_134954


namespace race_course_length_race_course_length_proof_l1349_134944

/-- Given two runners A and B, where A runs 4 times as fast as B and gives B a 63-meter head start,
    the length of the race course that allows both runners to finish at the same time is 84 meters. -/
theorem race_course_length : ℝ → ℝ → Prop :=
  fun (speed_B : ℝ) (course_length : ℝ) =>
    speed_B > 0 →
    course_length > 63 →
    course_length / (4 * speed_B) = (course_length - 63) / speed_B →
    course_length = 84

/-- Proof of the race_course_length theorem -/
theorem race_course_length_proof : ∃ (speed_B : ℝ) (course_length : ℝ),
  race_course_length speed_B course_length :=
by
  sorry

end race_course_length_race_course_length_proof_l1349_134944


namespace polygon_construction_possible_l1349_134942

/-- Represents a line segment with a fixed length -/
structure LineSegment where
  length : ℝ

/-- Represents a polygon constructed from line segments -/
structure Polygon where
  segments : List LineSegment
  isValid : Bool  -- Indicates if the polygon is valid (closed and non-self-intersecting)

/-- Calculates the area of a polygon -/
def calculateArea (p : Polygon) : ℝ := sorry

/-- Checks if it's possible to construct a polygon with given area using given line segments -/
def canConstructPolygon (segments : List LineSegment) (targetArea : ℝ) : Prop :=
  ∃ (p : Polygon), p.segments = segments ∧ p.isValid ∧ calculateArea p = targetArea

theorem polygon_construction_possible :
  let segments := List.replicate 12 { length := 2 }
  canConstructPolygon segments 16 := by
  sorry

end polygon_construction_possible_l1349_134942


namespace polynomial_remainder_theorem_l1349_134911

theorem polynomial_remainder_theorem (a b : ℚ) : 
  let f : ℚ → ℚ := λ x ↦ a * x^4 + 3 * x^3 - 5 * x^2 + b * x - 7
  (f 2 = 9 ∧ f (-1) = -4) → (a = 7/9 ∧ b = -2/9) := by
  sorry

end polynomial_remainder_theorem_l1349_134911


namespace diane_has_27_cents_l1349_134907

/-- The amount of money Diane has, given the cost of cookies and the additional amount she needs. -/
def dianes_money (cookie_cost additional_needed : ℕ) : ℕ :=
  cookie_cost - additional_needed

/-- Theorem stating that Diane has 27 cents given the problem conditions. -/
theorem diane_has_27_cents :
  dianes_money 65 38 = 27 := by
  sorry

end diane_has_27_cents_l1349_134907


namespace merry_go_round_revolutions_l1349_134928

theorem merry_go_round_revolutions 
  (distance_A : ℝ) 
  (distance_B : ℝ) 
  (revolutions_A : ℝ) 
  (h1 : distance_A = 36) 
  (h2 : distance_B = 12) 
  (h3 : revolutions_A = 40) 
  (h4 : distance_A * revolutions_A = distance_B * revolutions_B) : 
  revolutions_B = 120 := by
  sorry

end merry_go_round_revolutions_l1349_134928


namespace vehicle_travel_time_l1349_134909

/-- 
Given two vehicles A and B traveling towards each other, prove that B's total travel time is 7.2 hours
under the following conditions:
1. They meet after 3 hours.
2. A turns back to its starting point, taking 3 hours.
3. A then turns around again and meets B after 0.5 hours.
-/
theorem vehicle_travel_time (v_A v_B : ℝ) (d : ℝ) : 
  v_A > 0 ∧ v_B > 0 ∧ d > 0 → 
  d = 3 * (v_A + v_B) →
  3 * v_A = d / 2 →
  d / 2 + 0.5 * v_A = 3.5 * v_B →
  d / v_B = 7.2 := by
sorry

end vehicle_travel_time_l1349_134909


namespace park_length_l1349_134968

/-- The length of a rectangular park given its perimeter and breadth -/
theorem park_length (perimeter breadth : ℝ) (h1 : perimeter = 1000) (h2 : breadth = 200) :
  2 * (perimeter / 2 - breadth) = 300 := by
  sorry

end park_length_l1349_134968


namespace division_to_ratio_l1349_134943

theorem division_to_ratio (a b : ℝ) (h : a / b = 0.4) : a / b = 2 / 5 := by
  sorry

end division_to_ratio_l1349_134943


namespace combined_price_is_3105_l1349_134963

/-- Calculate the selling price of an item given its cost and profit percentage -/
def selling_price (cost : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost + cost * profit_percentage / 100

/-- Combined selling price of three items -/
def combined_selling_price (cost_A cost_B cost_C : ℕ) (profit_A profit_B profit_C : ℕ) : ℕ :=
  selling_price cost_A profit_A + selling_price cost_B profit_B + selling_price cost_C profit_C

/-- Theorem stating the combined selling price of the three items -/
theorem combined_price_is_3105 :
  combined_selling_price 500 800 1200 25 30 20 = 3105 := by
  sorry


end combined_price_is_3105_l1349_134963


namespace f_f_eq_x_solutions_l1349_134987

def f (x : ℝ) : ℝ := x^2 - 4*x - 5

def solution_set : Set ℝ := {(5 + 3*Real.sqrt 5)/2, (5 - 3*Real.sqrt 5)/2, (3 + Real.sqrt 41)/2, (3 - Real.sqrt 41)/2}

theorem f_f_eq_x_solutions :
  ∀ x : ℝ, f (f x) = x ↔ x ∈ solution_set :=
sorry

end f_f_eq_x_solutions_l1349_134987


namespace water_one_eighth_after_three_pourings_l1349_134965

def water_remaining (n : ℕ) : ℚ :=
  (1 : ℚ) / 2^n

theorem water_one_eighth_after_three_pourings :
  water_remaining 3 = (1 : ℚ) / 8 := by
  sorry

#check water_one_eighth_after_three_pourings

end water_one_eighth_after_three_pourings_l1349_134965


namespace largest_divisor_of_n_squared_div_360_l1349_134993

theorem largest_divisor_of_n_squared_div_360 (n : ℕ+) (h : 360 ∣ n^2) :
  ∃ (t : ℕ), t = 60 ∧ t ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ t :=
sorry

end largest_divisor_of_n_squared_div_360_l1349_134993


namespace line_ellipse_no_intersection_l1349_134918

/-- Given a line y = 2x + b and an ellipse x^2/4 + y^2 = 1,
    if the line has no point in common with the ellipse,
    then b < -2√2 or b > 2√2 -/
theorem line_ellipse_no_intersection (b : ℝ) : 
  (∀ x y : ℝ, y = 2*x + b → x^2/4 + y^2 ≠ 1) → 
  (b < -2 * Real.sqrt 2 ∨ b > 2 * Real.sqrt 2) := by
  sorry

end line_ellipse_no_intersection_l1349_134918


namespace logarithmic_identity_l1349_134969

theorem logarithmic_identity (a b : ℝ) (h1 : a^2 + b^2 = 7*a*b) (h2 : a*b ≠ 0) :
  Real.log (|a + b| / 3) = (1/2) * (Real.log |a| + Real.log |b|) := by
  sorry

end logarithmic_identity_l1349_134969


namespace max_q_minus_r_for_1027_l1349_134966

theorem max_q_minus_r_for_1027 :
  ∃ (q r : ℕ+), 1027 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1027 = 23 * q' + r' → q' - r' ≤ q - r ∧ q - r = 29 := by
sorry

end max_q_minus_r_for_1027_l1349_134966


namespace total_highlighters_l1349_134949

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h_pink : pink = 4)
  (h_yellow : yellow = 2)
  (h_blue : blue = 5) :
  pink + yellow + blue = 11 := by
  sorry

end total_highlighters_l1349_134949


namespace complex_fraction_simplification_l1349_134953

theorem complex_fraction_simplification :
  (7 + 16 * Complex.I) / (3 - 4 * Complex.I) = 6 - (38 / 7) * Complex.I :=
by sorry

end complex_fraction_simplification_l1349_134953


namespace min_quadrilateral_area_l1349_134999

-- Define the curve E
def curve_E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the tangent circle
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the tangent line
def tangent_line (x : ℝ) : Prop := x = -2

-- Define the point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the quadrilateral area function
def quadrilateral_area (a b c d : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem min_quadrilateral_area :
  ∀ (a b c d : ℝ × ℝ),
    (∃ (m : ℝ), m ≠ 0 ∧
      curve_E a.1 a.2 ∧ curve_E b.1 b.2 ∧ curve_E c.1 c.2 ∧ curve_E d.1 d.2 ∧
      (a.1 - point_F.1) * (c.1 - point_F.1) + (a.2 - point_F.2) * (c.2 - point_F.2) = 0 ∧
      (b.1 - point_F.1) * (d.1 - point_F.1) + (b.2 - point_F.2) * (d.2 - point_F.2) = 0) →
    quadrilateral_area a b c d ≥ 32 :=
sorry

end min_quadrilateral_area_l1349_134999
