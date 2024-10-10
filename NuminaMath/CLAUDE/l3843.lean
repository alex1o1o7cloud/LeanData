import Mathlib

namespace quadratic_completion_l3843_384324

theorem quadratic_completion (x : ℝ) : ∃ (a b c : ℤ), 
  a > 0 ∧ 
  (a * x + b : ℝ)^2 = 64 * x^2 + 96 * x + c ∧
  a + b + c = 131 := by
  sorry

end quadratic_completion_l3843_384324


namespace election_result_l3843_384349

theorem election_result (total_votes : ℕ) (second_candidate_votes : ℕ) :
  total_votes = 1200 →
  second_candidate_votes = 480 →
  (total_votes - second_candidate_votes) / total_votes = 3 / 5 := by
sorry

end election_result_l3843_384349


namespace square_area_ratio_l3843_384300

theorem square_area_ratio (y : ℝ) (y_pos : y > 0) : 
  (3 * y)^2 / (12 * y)^2 = 1 / 16 :=
by sorry

end square_area_ratio_l3843_384300


namespace special_function_properties_l3843_384398

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ 
  (f (1/3) = 1) ∧
  (∀ x : ℝ, x > 0 → f x > 0)

theorem special_function_properties (f : ℝ → ℝ) (h : special_function f) :
  (f 0 = 0) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, x < -2/3 → f x + f (2 + x) < 2) :=
by sorry

end special_function_properties_l3843_384398


namespace hawks_score_l3843_384345

theorem hawks_score (total_points margin : ℕ) (h1 : total_points = 48) (h2 : margin = 16) :
  ∃ (eagles_score hawks_score : ℕ),
    eagles_score + hawks_score = total_points ∧
    eagles_score - hawks_score = margin ∧
    hawks_score = 16 := by
  sorry

end hawks_score_l3843_384345


namespace max_min_difference_c_l3843_384344

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 5)
  (sum_squares_eq : a^2 + b^2 + c^2 = 27) :
  ∃ (c_max c_min : ℝ),
    (∀ c' : ℝ, (∃ a' b' : ℝ, a' + b' + c' = 5 ∧ a'^2 + b'^2 + c'^2 = 27) → c' ≤ c_max) ∧
    (∀ c' : ℝ, (∃ a' b' : ℝ, a' + b' + c' = 5 ∧ a'^2 + b'^2 + c'^2 = 27) → c_min ≤ c') ∧
    c_max - c_min = 22/3 :=
sorry

end max_min_difference_c_l3843_384344


namespace complex_equation_solution_l3843_384325

theorem complex_equation_solution (Z : ℂ) :
  (1 + 2*Complex.I)^3 * Z = 1 + 2*Complex.I →
  Z = -3/25 + 24/125*Complex.I :=
by sorry

end complex_equation_solution_l3843_384325


namespace book_selection_count_l3843_384374

/-- Represents the number of books in each genre -/
def genre_books : Fin 4 → ℕ
  | 0 => 4  -- Mystery novels
  | 1 => 3  -- Fantasy novels
  | 2 => 3  -- Biographies
  | 3 => 3  -- Science fiction novels

/-- The number of ways to choose three books from three different genres -/
def book_combinations : ℕ := 4 * 3 * 3 * 3

theorem book_selection_count :
  book_combinations = 108 :=
sorry

end book_selection_count_l3843_384374


namespace square_roots_sum_zero_l3843_384336

theorem square_roots_sum_zero (x : ℝ) (n : ℝ) (h1 : n > 0) 
  (h2 : (x - 4) + 3 = 0) : x = 1 := by
  sorry

end square_roots_sum_zero_l3843_384336


namespace no_bounded_function_satisfying_conditions_l3843_384322

theorem no_bounded_function_satisfying_conditions :
  ¬ ∃ (f : ℝ → ℝ), 
    (∃ (M : ℝ), ∀ x, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y, (f (x + y))^2 ≥ (f x)^2 + 2 * f (x * y) + (f y)^2) :=
by sorry

end no_bounded_function_satisfying_conditions_l3843_384322


namespace rectangle_ratio_l3843_384314

theorem rectangle_ratio (w l a : ℝ) : 
  w = 4 → 
  a = 48 → 
  a = l * w → 
  l / w = 3 := by
sorry

end rectangle_ratio_l3843_384314


namespace last_digit_77_base_4_l3843_384348

def last_digit_base_4 (n : ℕ) : ℕ :=
  n % 4

theorem last_digit_77_base_4 :
  last_digit_base_4 77 = 1 := by
  sorry

end last_digit_77_base_4_l3843_384348


namespace shaniqua_styles_l3843_384339

def haircut_price : ℕ := 12
def style_price : ℕ := 25
def total_earned : ℕ := 221
def haircuts_given : ℕ := 8

theorem shaniqua_styles (styles : ℕ) : styles = 5 := by
  sorry

end shaniqua_styles_l3843_384339


namespace inscribed_circle_distance_l3843_384370

theorem inscribed_circle_distance (a b : ℝ) (ha : a = 36) (hb : b = 48) :
  let c := Real.sqrt (a^2 + b^2)
  let r := (a + b - c) / 2
  let h := a * b / c
  let d := Real.sqrt ((r * Real.sqrt 2)^2 - ((h - r) * (h - r)))
  d = 12 / 5 := by sorry

end inscribed_circle_distance_l3843_384370


namespace incorrect_multiplication_result_l3843_384360

theorem incorrect_multiplication_result 
  (x : ℝ) 
  (h1 : ∃ a b : ℕ, 987 * x = 500000 + 10000 * a + 700 + b / 100 + 0.0989999999)
  (h2 : 987 * x ≠ 555707.2899999999)
  (h3 : 555707.2899999999 = 987 * x) : 
  987 * x = 598707.2989999999 := by
sorry

end incorrect_multiplication_result_l3843_384360


namespace range_of_abc_l3843_384365

theorem range_of_abc (a b c : ℝ) (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := by
  sorry

end range_of_abc_l3843_384365


namespace pens_left_after_sale_l3843_384327

def initial_pens : ℕ := 42
def sold_pens : ℕ := 23

theorem pens_left_after_sale : initial_pens - sold_pens = 19 := by
  sorry

end pens_left_after_sale_l3843_384327


namespace quadratic_function_ratio_bound_l3843_384308

/-- A quadratic function f(x) = ax² + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  derivative_at_zero_positive : b > 0
  nonnegative : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The ratio of f(1) to f'(0) for a QuadraticFunction is always at least 2 -/
theorem quadratic_function_ratio_bound (f : QuadraticFunction) :
  (f.a + f.b + f.c) / f.b ≥ 2 := by sorry

end quadratic_function_ratio_bound_l3843_384308


namespace point_in_second_quadrant_l3843_384347

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point (-1,2) -/
def point : Point :=
  { x := -1, y := 2 }

theorem point_in_second_quadrant : second_quadrant point := by
  sorry

end point_in_second_quadrant_l3843_384347


namespace m_range_l3843_384358

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

def B (m : ℝ) : Set ℝ := {x : ℝ | -1 < x ∧ x < m + 1}

theorem m_range (m : ℝ) : B m ⊆ A ∧ B m ≠ A → -2 < m ∧ m < 2 := by
  sorry

end m_range_l3843_384358


namespace uruguay_goals_conceded_l3843_384309

theorem uruguay_goals_conceded : 
  ∀ (x : ℕ), 
  (5 + 5 + 4 + 0 = 2 + 4 + x + 3) → 
  x = 5 := by
sorry

end uruguay_goals_conceded_l3843_384309


namespace complement_intersection_A_B_l3843_384369

def A : Set ℝ := {x | |x - 2| ≤ 3}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x : ℝ | x ≤ 1 ∨ x > 5} := by sorry

end complement_intersection_A_B_l3843_384369


namespace carrot_price_l3843_384394

/-- Calculates the price of a carrot given the number of tomatoes and carrots,
    the price of a tomato, and the total revenue from selling all produce. -/
theorem carrot_price
  (num_tomatoes : ℕ)
  (num_carrots : ℕ)
  (tomato_price : ℚ)
  (total_revenue : ℚ)
  (h1 : num_tomatoes = 200)
  (h2 : num_carrots = 350)
  (h3 : tomato_price = 1)
  (h4 : total_revenue = 725) :
  (total_revenue - num_tomatoes * tomato_price) / num_carrots = 3/2 := by
  sorry

#eval (725 : ℚ) - 200 * 1
#eval ((725 : ℚ) - 200 * 1) / 350

end carrot_price_l3843_384394


namespace plane_equation_proof_l3843_384388

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane given by its coefficients -/
def pointLiesOnPlane (p : Point3D) (coeff : PlaneCoefficients) : Prop :=
  coeff.A * p.x + coeff.B * p.y + coeff.C * p.z + coeff.D = 0

/-- The greatest common divisor of the absolute values of four integers is 1 -/
def gcdIsOne (a b c d : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

theorem plane_equation_proof (p1 p2 p3 : Point3D) (coeff : PlaneCoefficients) : 
  p1 = ⟨2, -3, 1⟩ →
  p2 = ⟨6, -3, 3⟩ →
  p3 = ⟨4, -5, 2⟩ →
  coeff = ⟨2, 3, -4, 9⟩ →
  pointLiesOnPlane p1 coeff ∧
  pointLiesOnPlane p2 coeff ∧
  pointLiesOnPlane p3 coeff ∧
  coeff.A > 0 ∧
  gcdIsOne coeff.A coeff.B coeff.C coeff.D := by
  sorry

end plane_equation_proof_l3843_384388


namespace common_external_tangent_y_intercept_l3843_384311

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  (l.m * x₀ - y₀ + l.b)^2 = (c.radius^2 * (l.m^2 + 1))

theorem common_external_tangent_y_intercept :
  let c₁ : Circle := ⟨(1, 3), 3⟩
  let c₂ : Circle := ⟨(15, 10), 8⟩
  ∃ (l : Line), l.m > 0 ∧ isTangent l c₁ ∧ isTangent l c₂ ∧ l.b = 5/3 :=
sorry

end common_external_tangent_y_intercept_l3843_384311


namespace penguin_giraffe_ratio_l3843_384379

/-- Represents the zoo with its animal composition -/
structure Zoo where
  total_animals : ℕ
  giraffes : ℕ
  penguins : ℕ
  elephants : ℕ

/-- The conditions of the zoo -/
def zoo_conditions (z : Zoo) : Prop :=
  z.giraffes = 5 ∧
  z.penguins = (20 : ℕ) * z.total_animals / 100 ∧
  z.elephants = 2 ∧
  z.elephants = (4 : ℕ) * z.total_animals / 100

/-- The theorem stating the ratio of penguins to giraffes -/
theorem penguin_giraffe_ratio (z : Zoo) (h : zoo_conditions z) : 
  z.penguins / z.giraffes = 2 := by
  sorry

#check penguin_giraffe_ratio

end penguin_giraffe_ratio_l3843_384379


namespace quadratic_completion_square_l3843_384306

theorem quadratic_completion_square (x : ℝ) : 
  (∃ (d e : ℤ), (x + d : ℝ)^2 = e ∧ x^2 - 6*x - 15 = 0) → 
  (∃ (d e : ℤ), (x + d : ℝ)^2 = e ∧ x^2 - 6*x - 15 = 0 ∧ d + e = 21) := by
sorry

end quadratic_completion_square_l3843_384306


namespace absolute_value_and_square_root_l3843_384367

theorem absolute_value_and_square_root (x : ℝ) (h : 1 < x ∧ x ≤ 2) :
  |x - 3| + Real.sqrt ((x - 2)^2) = 5 - 2*x := by
  sorry

end absolute_value_and_square_root_l3843_384367


namespace expressions_equality_l3843_384354

theorem expressions_equality :
  -- Expression 1
  (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = -1 + Real.sqrt 3 ∧
  -- Expression 2
  2 * (Real.sqrt (9/2) - Real.sqrt 8 / 3) * (2 * Real.sqrt 2) = 10/3 ∧
  -- Expression 3
  Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1/8) = 5 * Real.sqrt 2 / 4 ∧
  -- Expression 4
  (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1/2) = -6 * Real.sqrt 5 :=
by sorry

end expressions_equality_l3843_384354


namespace pizza_order_l3843_384393

theorem pizza_order (adults children adult_slices child_slices slices_per_pizza : ℕ) 
  (h1 : adults = 2)
  (h2 : children = 6)
  (h3 : adult_slices = 3)
  (h4 : child_slices = 1)
  (h5 : slices_per_pizza = 4) :
  (adults * adult_slices + children * child_slices) / slices_per_pizza = 3 := by
  sorry


end pizza_order_l3843_384393


namespace arithmetic_sequence_m_value_l3843_384373

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The theorem stating that if S_{m-1} = -3, S_m = 0, and S_{m+1} = 5, then m = 4 -/
theorem arithmetic_sequence_m_value
  (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ)
  (h_arithmetic : arithmetic_sequence a S)
  (h_m_minus_1 : S (m - 1) = -3)
  (h_m : S m = 0)
  (h_m_plus_1 : S (m + 1) = 5) :
  m = 4 := by
  sorry


end arithmetic_sequence_m_value_l3843_384373


namespace average_hamburgers_is_nine_l3843_384313

/-- The number of hamburgers sold in a week -/
def hamburgers_sold : ℕ := 63

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of hamburgers sold per day -/
def average_hamburgers_per_day : ℚ := hamburgers_sold / days_in_week

/-- Theorem stating that the average number of hamburgers sold per day is 9 -/
theorem average_hamburgers_is_nine :
  average_hamburgers_per_day = 9 := by
  sorry

end average_hamburgers_is_nine_l3843_384313


namespace alberts_brother_age_difference_l3843_384326

/-- Proves that Albert's brother is 2 years younger than Albert given the problem conditions -/
theorem alberts_brother_age_difference : ℕ → Prop :=
  fun albert_age : ℕ =>
    ∀ (father_age mother_age brother_age : ℕ),
      father_age = albert_age + 48 →
      mother_age = brother_age + 46 →
      father_age = mother_age + 4 →
      brother_age < albert_age →
      albert_age - brother_age = 2

/-- Proof of the theorem -/
lemma prove_alberts_brother_age_difference :
  ∀ albert_age : ℕ, alberts_brother_age_difference albert_age :=
by
  sorry

#check prove_alberts_brother_age_difference

end alberts_brother_age_difference_l3843_384326


namespace remainders_of_1493827_l3843_384390

theorem remainders_of_1493827 : 
  (1493827 % 4 = 3) ∧ (1493827 % 3 = 1) := by
  sorry

end remainders_of_1493827_l3843_384390


namespace mistaken_calculation_l3843_384317

theorem mistaken_calculation (x : ℕ) : 423 - x = 421 → (423 * x) + (423 - x) = 1267 := by
  sorry

end mistaken_calculation_l3843_384317


namespace green_tea_cost_july_l3843_384303

/-- The cost of green tea in July given initial prices and price changes -/
theorem green_tea_cost_july (initial_cost : ℝ) 
  (h1 : initial_cost > 0) 
  (h2 : 3 * (0.1 * initial_cost + 2 * initial_cost) / 2 = 3.15) : 
  0.1 * initial_cost = 0.1 := by sorry

end green_tea_cost_july_l3843_384303


namespace total_money_is_140_l3843_384385

/-- Calculates the total money collected from football game tickets -/
def total_money_collected (adult_price child_price : ℚ) (total_attendees adult_attendees : ℕ) : ℚ :=
  adult_price * adult_attendees + child_price * (total_attendees - adult_attendees)

/-- Theorem stating that the total money collected is $140 -/
theorem total_money_is_140 :
  let adult_price : ℚ := 60 / 100
  let child_price : ℚ := 25 / 100
  let total_attendees : ℕ := 280
  let adult_attendees : ℕ := 200
  total_money_collected adult_price child_price total_attendees adult_attendees = 140 / 1 := by
  sorry

end total_money_is_140_l3843_384385


namespace total_sandcastles_and_towers_l3843_384340

/-- The number of sandcastles on Mark's beach -/
def marks_castles : ℕ := 20

/-- The number of towers per sandcastle on Mark's beach -/
def marks_towers_per_castle : ℕ := 10

/-- The ratio of Jeff's sandcastles to Mark's sandcastles -/
def jeff_castle_ratio : ℕ := 3

/-- The number of towers per sandcastle on Jeff's beach -/
def jeffs_towers_per_castle : ℕ := 5

/-- Theorem stating the combined total number of sandcastles and towers on both beaches -/
theorem total_sandcastles_and_towers :
  marks_castles * marks_towers_per_castle +
  marks_castles +
  (jeff_castle_ratio * marks_castles) * jeffs_towers_per_castle +
  (jeff_castle_ratio * marks_castles) = 580 := by
  sorry

end total_sandcastles_and_towers_l3843_384340


namespace min_abs_diff_solution_product_l3843_384399

theorem min_abs_diff_solution_product (x y : ℤ) : 
  (20 * x + 19 * y = 2019) →
  (∀ a b : ℤ, 20 * a + 19 * b = 2019 → |x - y| ≤ |a - b|) →
  x * y = 2623 := by
  sorry

end min_abs_diff_solution_product_l3843_384399


namespace smallest_sum_of_sequence_l3843_384302

theorem smallest_sum_of_sequence (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 → 
  (C - B = B - A) →  -- arithmetic sequence condition
  (C * C = B * D) →  -- geometric sequence condition
  (C : ℚ) / B = 7 / 4 →
  A + B + C + D ≥ 97 :=
by sorry

end smallest_sum_of_sequence_l3843_384302


namespace largest_prime_factor_of_1729_l3843_384368

theorem largest_prime_factor_of_1729 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p :=
by
  -- The proof would go here
  sorry

end largest_prime_factor_of_1729_l3843_384368


namespace least_three_digit_with_digit_product_18_l3843_384333

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem least_three_digit_with_digit_product_18 :
  ∃ (n : ℕ), is_three_digit n ∧ digit_product n = 18 ∧
  ∀ (m : ℕ), is_three_digit m → digit_product m = 18 → n ≤ m :=
by
  sorry

end least_three_digit_with_digit_product_18_l3843_384333


namespace cube_root_3a_5b_square_root_4x_y_l3843_384391

-- Part 1
theorem cube_root_3a_5b (a b : ℝ) (h : b = 4 * Real.sqrt (3 * a - 2) + 2 * Real.sqrt (2 - 3 * a) + 5) :
  (3 * a + 5 * b) ^ (1/3 : ℝ) = 3 := by sorry

-- Part 2
theorem square_root_4x_y (x y : ℝ) (h : (x - 3)^2 + Real.sqrt (y - 4) = 0) :
  (4 * x + y) ^ (1/2 : ℝ) = 4 ∨ (4 * x + y) ^ (1/2 : ℝ) = -4 := by sorry

end cube_root_3a_5b_square_root_4x_y_l3843_384391


namespace hanna_money_spent_l3843_384343

theorem hanna_money_spent (rose_price : ℚ) (jenna_fraction : ℚ) (imma_fraction : ℚ) (total_given : ℕ) : 
  rose_price = 2 →
  jenna_fraction = 1/3 →
  imma_fraction = 1/2 →
  total_given = 125 →
  (jenna_fraction + imma_fraction) * (total_given / (jenna_fraction + imma_fraction)) * rose_price = 300 := by
sorry

end hanna_money_spent_l3843_384343


namespace pen_probabilities_l3843_384366

/-- Represents the total number of pens in the box -/
def total_pens : ℕ := 6

/-- Represents the number of first-class quality pens -/
def first_class_pens : ℕ := 4

/-- Represents the number of second-class quality pens -/
def second_class_pens : ℕ := 2

/-- Represents the number of pens drawn -/
def pens_drawn : ℕ := 2

/-- Calculates the probability of drawing exactly one first-class quality pen -/
def prob_one_first_class : ℚ :=
  (Nat.choose first_class_pens 1 * Nat.choose second_class_pens 1) / Nat.choose total_pens pens_drawn

/-- Calculates the probability of drawing at least one second-class quality pen -/
def prob_at_least_one_second_class : ℚ :=
  1 - (Nat.choose first_class_pens pens_drawn) / Nat.choose total_pens pens_drawn

theorem pen_probabilities :
  prob_one_first_class = 8/15 ∧
  prob_at_least_one_second_class = 3/5 :=
sorry

end pen_probabilities_l3843_384366


namespace line_perpendicular_plane_implies_planes_perpendicular_l3843_384387

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the contained relation between a line and a plane
variable (contained : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem line_perpendicular_plane_implies_planes_perpendicular
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : contained m β) :
  perpendicularPlanes α β :=
sorry

end line_perpendicular_plane_implies_planes_perpendicular_l3843_384387


namespace lunch_cost_with_tip_l3843_384386

theorem lunch_cost_with_tip (total_cost : ℝ) (tip_percentage : ℝ) (cost_before_tip : ℝ) : 
  total_cost = 60.24 →
  tip_percentage = 0.20 →
  total_cost = cost_before_tip * (1 + tip_percentage) →
  cost_before_tip = 50.20 := by
sorry

end lunch_cost_with_tip_l3843_384386


namespace jason_has_more_blue_marbles_l3843_384337

/-- The number of blue marbles Jason has -/
def jason_blue : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue : ℕ := 24

/-- The difference in blue marbles between Jason and Tom -/
def blue_marble_difference : ℕ := jason_blue - tom_blue

/-- Theorem stating that Jason has 20 more blue marbles than Tom -/
theorem jason_has_more_blue_marbles : blue_marble_difference = 20 := by
  sorry

end jason_has_more_blue_marbles_l3843_384337


namespace sufficient_not_necessary_l3843_384335

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → a ≥ 1) ∧ (∃ a, a ≥ 1 ∧ a ≤ 2) := by
  sorry

end sufficient_not_necessary_l3843_384335


namespace line_perpendicular_to_plane_and_line_in_plane_l3843_384301

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- Theorem statement
theorem line_perpendicular_to_plane_and_line_in_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contains α n) :
  perpendicularLines m n :=
sorry

end line_perpendicular_to_plane_and_line_in_plane_l3843_384301


namespace x_range_l3843_384361

def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0

def q (x : ℝ) : Prop := 1 / (3 - x) > 1

theorem x_range (x : ℝ) (h1 : p x) (h2 : ¬(q x)) : 
  x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3 := by
  sorry

end x_range_l3843_384361


namespace job_completion_solution_l3843_384372

/-- Represents the time taken by machines to complete a job -/
def job_completion_time (x : ℝ) : Prop :=
  let p_alone := x + 5
  let q_alone := x + 2
  let r_alone := 2 * x
  let pq_together := x + 3
  (1 / p_alone + 1 / q_alone + 1 / r_alone = 1 / x) ∧
  (1 / p_alone + 1 / q_alone = 1 / pq_together)

/-- Theorem stating that x = 2 satisfies the job completion time conditions -/
theorem job_completion_solution : job_completion_time 2 := by
  sorry

end job_completion_solution_l3843_384372


namespace solution_set_is_open_unit_interval_l3843_384316

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f x < f y

-- Define the set of x values satisfying f(2x-1) < f(1)
def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f (2*x - 1) < f 1}

-- State the theorem
theorem solution_set_is_open_unit_interval (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) : 
  solution_set f = Set.Ioo 0 1 := by sorry

end solution_set_is_open_unit_interval_l3843_384316


namespace abc_value_l3843_384346

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 165) (h2 : b * (c + a) = 195) (h3 : c * (a + b) = 180) :
  a * b * c = 15 * Real.sqrt 210 := by
sorry

end abc_value_l3843_384346


namespace polar_to_rectangular_l3843_384329

/-- Given a point with polar coordinates (3, π/4), prove that its rectangular coordinates are (3√2/2, 3√2/2) -/
theorem polar_to_rectangular :
  let r : ℝ := 3
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 3 * Real.sqrt 2 / 2 ∧ y = 3 * Real.sqrt 2 / 2 := by
  sorry

end polar_to_rectangular_l3843_384329


namespace similar_triangle_sum_l3843_384341

theorem similar_triangle_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a / 3 = b / 5) (h5 : b / 5 = c / 7) (h6 : c = 21) : a + b = 24 := by
  sorry

end similar_triangle_sum_l3843_384341


namespace smallest_equal_prob_sum_l3843_384318

/-- The number of faces on a standard die -/
def faces : ℕ := 6

/-- The target sum we're comparing to -/
def target_sum : ℕ := 2001

/-- The smallest number of dice needed to potentially reach the target sum -/
def min_dice : ℕ := (target_sum + faces - 1) / faces

/-- The function that transforms a die roll -/
def transform (x : ℕ) : ℕ := faces + 1 - x

/-- The smallest value S with equal probability to the target sum -/
def smallest_S : ℕ := (faces + 1) * min_dice - target_sum

theorem smallest_equal_prob_sum :
  smallest_S = 337 :=
sorry

end smallest_equal_prob_sum_l3843_384318


namespace trioball_playing_time_l3843_384375

theorem trioball_playing_time (total_children : ℕ) (playing_children : ℕ) (total_time : ℕ) 
  (h1 : total_children = 6)
  (h2 : playing_children = 3)
  (h3 : total_time = 180) :
  (total_time * playing_children) / total_children = 90 := by
  sorry

end trioball_playing_time_l3843_384375


namespace tylers_remaining_money_l3843_384380

/-- Calculates the remaining money after Tyler's purchase of scissors and erasers. -/
theorem tylers_remaining_money 
  (initial_money : ℕ) 
  (scissor_cost : ℕ) 
  (eraser_cost : ℕ) 
  (scissor_count : ℕ) 
  (eraser_count : ℕ) 
  (h1 : initial_money = 100)
  (h2 : scissor_cost = 5)
  (h3 : eraser_cost = 4)
  (h4 : scissor_count = 8)
  (h5 : eraser_count = 10) :
  initial_money - (scissor_cost * scissor_count + eraser_cost * eraser_count) = 20 := by
  sorry

#check tylers_remaining_money

end tylers_remaining_money_l3843_384380


namespace coefficient_is_negative_seven_l3843_384383

-- Define the expression
def expression (x : ℝ) : ℝ := 5 * (x - 6) + 6 * (9 - 3 * x^2 + 3 * x) - 10 * (3 * x - 2)

-- Define the coefficient of x
def coefficient_of_x (f : ℝ → ℝ) : ℝ :=
  (f 1 - f 0)

-- Theorem statement
theorem coefficient_is_negative_seven :
  coefficient_of_x expression = -7 := by sorry

end coefficient_is_negative_seven_l3843_384383


namespace trays_needed_to_refill_l3843_384397

/-- The number of ice cubes Dylan used in his glass -/
def dylan_glass_cubes : ℕ := 8

/-- The number of ice cubes used per glass for lemonade -/
def lemonade_glass_cubes : ℕ := 2 * dylan_glass_cubes

/-- The total number of glasses served (including Dylan's) -/
def total_glasses : ℕ := 5 + 1

/-- The number of spaces in each ice cube tray -/
def tray_spaces : ℕ := 14

/-- The fraction of total ice cubes used -/
def fraction_used : ℚ := 4/5

/-- The total number of ice cubes used -/
def total_used : ℕ := dylan_glass_cubes + lemonade_glass_cubes * total_glasses

/-- The initial total number of ice cubes -/
def initial_total : ℚ := (total_used : ℚ) / fraction_used

theorem trays_needed_to_refill : 
  ⌈initial_total / tray_spaces⌉ = 10 := by sorry

end trays_needed_to_refill_l3843_384397


namespace share_difference_l3843_384332

/-- Represents the share of money for each person -/
structure Share :=
  (amount : ℕ)

/-- Represents the distribution of money among three people -/
structure Distribution :=
  (faruk : Share)
  (vasim : Share)
  (ranjith : Share)

/-- Defines the ratio of distribution -/
def distribution_ratio : Distribution → (ℕ × ℕ × ℕ)
  | ⟨f, v, r⟩ => (3, 5, 8)

theorem share_difference (d : Distribution) :
  distribution_ratio d = (3, 5, 8) →
  d.vasim.amount = 1500 →
  d.ranjith.amount - d.faruk.amount = 1500 :=
by sorry

end share_difference_l3843_384332


namespace sine_phase_shift_l3843_384378

theorem sine_phase_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 6)) := by sorry

end sine_phase_shift_l3843_384378


namespace polynomial_equality_l3843_384304

theorem polynomial_equality (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a - a₁ + a₂ - a₃ + a₄ - a₅ = -243 := by
sorry

end polynomial_equality_l3843_384304


namespace pool_volume_l3843_384323

/-- Represents a pool with given parameters -/
structure Pool where
  diameter : ℝ
  fill_time : ℝ
  hose_rates : List ℝ

/-- Calculates the volume of water delivered by hoses over a given time -/
def water_volume (p : Pool) : ℝ :=
  (p.hose_rates.sum * p.fill_time * 60)

/-- The theorem states that a pool with given parameters has a volume of 15000 gallons -/
theorem pool_volume (p : Pool) 
  (h1 : p.diameter = 24)
  (h2 : p.fill_time = 25)
  (h3 : p.hose_rates = [2, 2, 3, 3]) :
  water_volume p = 15000 := by
  sorry

#check pool_volume

end pool_volume_l3843_384323


namespace total_cookies_l3843_384331

theorem total_cookies (num_people : ℕ) (cookies_per_person : ℕ) 
  (h1 : num_people = 6)
  (h2 : cookies_per_person = 4) :
  num_people * cookies_per_person = 24 := by
  sorry

end total_cookies_l3843_384331


namespace circle_equation_radius_l3843_384392

theorem circle_equation_radius (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 4*y - k = 0 ↔ (x + 4)^2 + (y + 2)^2 = 49) ↔ 
  k = 29 := by
sorry

end circle_equation_radius_l3843_384392


namespace pizza_theorem_l3843_384363

/-- Calculates the total number of pizza slices brought by friends -/
def totalPizzaSlices (numFriends : ℕ) (slicesPerFriend : ℕ) : ℕ :=
  numFriends * slicesPerFriend

/-- Theorem: Given 4 friends, each bringing 4 slices of pizza, the total number of pizza slices is 16 -/
theorem pizza_theorem : totalPizzaSlices 4 4 = 16 := by
  sorry

end pizza_theorem_l3843_384363


namespace negation_of_existence_is_universal_l3843_384357

theorem negation_of_existence_is_universal : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end negation_of_existence_is_universal_l3843_384357


namespace trigonometric_equation_solution_l3843_384350

theorem trigonometric_equation_solution (x : ℝ) : 
  (4 * (Real.tan (8 * x))^4 + 4 * Real.sin (2 * x) * Real.sin (6 * x) - 
   Real.cos (4 * x) - Real.cos (12 * x) + 2) / Real.sqrt (Real.cos x - Real.sin x) = 0 ∧
  Real.cos x - Real.sin x > 0 ↔
  (∃ n : ℤ, x = -π/2 + 2*n*π ∨ x = -π/4 + 2*n*π ∨ x = 2*n*π) :=
by sorry

end trigonometric_equation_solution_l3843_384350


namespace chess_tournament_games_l3843_384307

/-- The number of games in a chess tournament --/
def tournament_games (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * k

/-- Theorem: In a chess tournament with 50 players, where each player plays
    four times with each opponent, the total number of games is 4900 --/
theorem chess_tournament_games :
  tournament_games 50 4 = 4900 := by
  sorry

end chess_tournament_games_l3843_384307


namespace fraction_division_addition_l3843_384389

theorem fraction_division_addition : (3 / 7 : ℚ) / 4 + 2 = 59 / 28 := by
  sorry

end fraction_division_addition_l3843_384389


namespace play_area_calculation_l3843_384382

/-- Calculates the area of a rectangular play area given specific fencing conditions. -/
theorem play_area_calculation (total_posts : ℕ) (post_spacing : ℕ) (extra_posts_long_side : ℕ) : 
  total_posts = 24 → 
  post_spacing = 5 → 
  extra_posts_long_side = 6 → 
  ∃ (short_side_posts long_side_posts : ℕ),
    short_side_posts + extra_posts_long_side = long_side_posts ∧
    2 * short_side_posts + 2 * long_side_posts - 4 = total_posts ∧
    (short_side_posts - 1) * post_spacing * (long_side_posts - 1) * post_spacing = 675 :=
by sorry

end play_area_calculation_l3843_384382


namespace total_squares_5x6_grid_l3843_384377

/-- The number of squares of a given size in a grid --/
def count_squares (grid_width : ℕ) (grid_height : ℕ) (square_size : ℕ) : ℕ :=
  (grid_width - square_size + 1) * (grid_height - square_size + 1)

/-- The total number of squares in a 5x6 grid --/
theorem total_squares_5x6_grid :
  let grid_width := 5
  let grid_height := 6
  (count_squares grid_width grid_height 1) +
  (count_squares grid_width grid_height 2) +
  (count_squares grid_width grid_height 3) +
  (count_squares grid_width grid_height 4) = 68 := by
  sorry

end total_squares_5x6_grid_l3843_384377


namespace bags_used_by_kid4_l3843_384353

def hours : ℕ := 5
def ears_per_row : ℕ := 85
def seeds_per_bag : ℕ := 48
def seeds_per_ear : ℕ := 2
def rows_per_hour_kid4 : ℕ := 5

def bags_used_kid4 : ℕ :=
  let rows := hours * rows_per_hour_kid4
  let ears := rows * ears_per_row
  let seeds := ears * seeds_per_ear
  (seeds + seeds_per_bag - 1) / seeds_per_bag

theorem bags_used_by_kid4 : bags_used_kid4 = 89 := by sorry

end bags_used_by_kid4_l3843_384353


namespace quadratic_form_equivalence_l3843_384396

theorem quadratic_form_equivalence :
  ∀ (x : ℝ), 3 * x^2 + 9 * x + 20 = 3 * (x + 3/2)^2 + 53/4 ∧
  ∃ (h : ℝ), h = -3/2 ∧ ∀ (x : ℝ), 3 * x^2 + 9 * x + 20 = 3 * (x - h)^2 + 53/4 :=
by sorry

end quadratic_form_equivalence_l3843_384396


namespace launderette_machine_count_l3843_384342

/-- Represents a laundry machine with quarters and dimes -/
structure LaundryMachine where
  quarters : ℕ
  dimes : ℕ

/-- Calculates the value of a laundry machine in cents -/
def machine_value (m : LaundryMachine) : ℕ :=
  m.quarters * 25 + m.dimes * 10

/-- Represents the launderette -/
structure Launderette where
  machine : LaundryMachine
  total_value : ℕ
  machine_count : ℕ

/-- Theorem: The number of machines in the launderette is 3 -/
theorem launderette_machine_count (l : Launderette) 
  (h1 : l.machine.quarters = 80)
  (h2 : l.machine.dimes = 100)
  (h3 : l.total_value = 9000) -- $90 in cents
  (h4 : l.machine_count * machine_value l.machine = l.total_value) :
  l.machine_count = 3 := by
  sorry

end launderette_machine_count_l3843_384342


namespace monotonicity_and_minimum_l3843_384305

/-- The function f(x) = kx^3 - 3x^2 + 1 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 - 3 * x^2 + 1

/-- The derivative of f(x) -/
noncomputable def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * k * x^2 - 6 * x

theorem monotonicity_and_minimum (k : ℝ) (h : k ≥ 0) :
  (∀ x y, x ≤ 0 → y ∈ Set.Ioo 0 (2/k) → f k x ≤ f k y) ∧ 
  (∀ x y, x ∈ Set.Icc 0 (2/k) → y ≥ 2/k → f k x ≤ f k y) ∧
  (k > 2 ↔ f k (2/k) > 0) :=
sorry

end monotonicity_and_minimum_l3843_384305


namespace polygon_sides_l3843_384371

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 540 → ∃ n : ℕ, n = 5 ∧ (n - 2) * 180 = sum_interior_angles :=
by
  sorry

end polygon_sides_l3843_384371


namespace fraction_nonzero_digits_l3843_384310

/-- The number of non-zero digits to the right of the decimal point in the decimal representation of a rational number -/
def nonZeroDigitsAfterDecimal (q : ℚ) : ℕ :=
  sorry

/-- The fraction we're considering -/
def fraction : ℚ := 120 / (2^4 * 5^9)

theorem fraction_nonzero_digits :
  nonZeroDigitsAfterDecimal fraction = 3 :=
sorry

end fraction_nonzero_digits_l3843_384310


namespace train_boarding_probability_l3843_384351

theorem train_boarding_probability 
  (cycle_time : ℝ) 
  (favorable_window : ℝ) 
  (h1 : cycle_time = 5) 
  (h2 : favorable_window = 0.5) 
  (h3 : 0 < favorable_window) 
  (h4 : favorable_window < cycle_time) :
  (favorable_window / cycle_time) = (1 / 10) := by
sorry

end train_boarding_probability_l3843_384351


namespace cos_215_minus_1_l3843_384395

theorem cos_215_minus_1 : Real.cos (215 * π / 180) - 1 = Real.sqrt 3 / 2 := by
  sorry

end cos_215_minus_1_l3843_384395


namespace f_extrema_l3843_384312

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

theorem f_extrema :
  let a : ℝ := -3
  let b : ℝ := 3
  (∀ x ∈ Set.Icc a b, f x ≤ 48) ∧
  (∃ x ∈ Set.Icc a b, f x = 48) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ -4) ∧
  (∃ x ∈ Set.Icc a b, f x = -4) :=
by sorry

end f_extrema_l3843_384312


namespace pyramid_surface_area_l3843_384352

/-- Given a pyramid with its base coinciding with a face of a cube and its apex at the center
    of the opposite face, the surface area of the pyramid can be expressed in terms of the
    cube's edge length. -/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) :
  ∃ (S : ℝ), S = (a * (3 * Real.sqrt (4 * a^2 - a^2) + a * Real.sqrt 3)) / 36 :=
sorry

end pyramid_surface_area_l3843_384352


namespace sum_first_six_terms_eq_54_l3843_384384

/-- An arithmetic sequence with given 3rd, 4th, and 5th terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  a 3 = 7 ∧ a 4 = 11 ∧ a 5 = 15 ∧ ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The sum of the first six terms of the sequence -/
def SumFirstSixTerms (a : ℕ → ℤ) : ℤ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

/-- Theorem stating that the sum of the first six terms is 54 -/
theorem sum_first_six_terms_eq_54 (a : ℕ → ℤ) (h : ArithmeticSequence a) :
  SumFirstSixTerms a = 54 := by
  sorry

end sum_first_six_terms_eq_54_l3843_384384


namespace equation_is_quadratic_l3843_384328

/-- Represents a quadratic equation in one variable x -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- Checks if an equation is quadratic in one variable x -/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (q : QuadraticEquation), ∀ x, f x = q.a * x^2 + q.b * x + q.c

/-- The equation x² = 1 -/
def equation (x : ℝ) : ℝ := x^2 - 1

theorem equation_is_quadratic : is_quadratic_in_x equation := by sorry

end equation_is_quadratic_l3843_384328


namespace smallest_number_with_remainders_l3843_384356

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 4 ∧
  ∀ (m : ℕ), m > 0 → m % 3 = 1 → m % 5 = 3 → m % 6 = 4 → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_number_with_remainders_l3843_384356


namespace classroom_students_l3843_384315

theorem classroom_students (boys girls : ℕ) : 
  boys * 5 = girls * 3 →  -- ratio of boys to girls is 3:5
  girls = boys + 4 →      -- there are 4 more girls than boys
  boys + girls = 16       -- total number of students is 16
:= by sorry

end classroom_students_l3843_384315


namespace cube_root_simplification_l3843_384338

theorem cube_root_simplification : (2^6 * 3^3 * 7^3 * 13^3 : ℝ)^(1/3) = 1092 := by
  sorry

end cube_root_simplification_l3843_384338


namespace max_y_coordinate_difference_l3843_384319

-- Define the two functions
def f (x : ℝ) : ℝ := 3 - x^2 + x^3
def g (x : ℝ) : ℝ := 1 + x^2 + x^3

-- Define the set of intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem max_y_coordinate_difference :
  ∃ (a b : ℝ), a ∈ intersection_points ∧ b ∈ intersection_points ∧
  ∀ (x y : ℝ), x ∈ intersection_points → y ∈ intersection_points →
  |f x - f y| ≤ |f a - f b| ∧ |f a - f b| = 2 :=
sorry

end max_y_coordinate_difference_l3843_384319


namespace tan_half_product_l3843_384321

theorem tan_half_product (a b : Real) :
  3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = Real.sqrt 2) ∨
  (Real.tan (a / 2) * Real.tan (b / 2) = -Real.sqrt 2) := by
sorry

end tan_half_product_l3843_384321


namespace circle_diameter_ratio_l3843_384355

theorem circle_diameter_ratio (R S : Real) (harea : R^2 = 0.64 * S^2) : 
  R = 0.8 * S := by
sorry

end circle_diameter_ratio_l3843_384355


namespace simplify_expression_l3843_384362

theorem simplify_expression : 
  2 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := by
  sorry

end simplify_expression_l3843_384362


namespace same_speed_problem_l3843_384320

theorem same_speed_problem (x : ℝ) :
  let jack_speed := x^2 - 11*x - 22
  let jill_distance := x^2 - 5*x - 36
  let jill_time := x + 4
  jack_speed > 0 ∧ 
  jill_distance > 0 ∧ 
  jill_time > 0 ∧
  jack_speed = jill_distance / jill_time →
  jack_speed = 4 := by
sorry

end same_speed_problem_l3843_384320


namespace quilt_gray_percentage_l3843_384376

/-- Represents a square quilt with white and gray parts -/
structure Quilt where
  size : ℕ
  gray_half_squares : ℕ
  gray_quarter_squares : ℕ
  gray_full_squares : ℕ

/-- Calculates the percentage of gray area in the quilt -/
def gray_percentage (q : Quilt) : ℚ :=
  let total_squares := q.size * q.size
  let gray_squares := q.gray_half_squares / 2 + q.gray_quarter_squares / 4 + q.gray_full_squares
  (gray_squares * 100) / total_squares

/-- Theorem stating that the specific quilt configuration has 40% gray area -/
theorem quilt_gray_percentage :
  let q := Quilt.mk 5 8 8 4
  gray_percentage q = 40 := by
  sorry

end quilt_gray_percentage_l3843_384376


namespace sum_of_complex_numbers_l3843_384359

-- Define the complex numbers
def z1 (a b : ℂ) : ℂ := 2 * a + b * Complex.I
def z2 (c d : ℂ) : ℂ := c + 3 * d * Complex.I
def z3 (e f : ℂ) : ℂ := e + f * Complex.I

-- State the theorem
theorem sum_of_complex_numbers (a b c d e f : ℂ) :
  b = 4 →
  e = -2 * a - c →
  z1 a b + z2 c d + z3 e f = 6 * Complex.I →
  d + f = 2 := by
  sorry


end sum_of_complex_numbers_l3843_384359


namespace hospital_opening_date_l3843_384330

theorem hospital_opening_date :
  ∃! (x y h : ℕ+),
    (x.val : ℤ) - (y.val : ℤ) = h.val ∨ (y.val : ℤ) - (x.val : ℤ) = h.val ∧
    x * (y * h - 1) = 1539 ∧
    h = 2 :=
by sorry

end hospital_opening_date_l3843_384330


namespace distance_between_vertices_l3843_384364

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 2) = 4

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = 3 - (1/12) * x^2
def parabola2 (x y : ℝ) : Prop := y = (1/4) * x^2 - 1

-- Define the vertices
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices :
  ∀ x y : ℝ, equation x y →
  (∃ x1 y1 x2 y2 : ℝ, 
    parabola1 x1 y1 ∧ parabola2 x2 y2 ∧
    (x1, y1) = vertex1 ∧ (x2, y2) = vertex2 ∧
    abs (y1 - y2) = 4) :=
by sorry

end distance_between_vertices_l3843_384364


namespace tony_purchase_cost_l3843_384381

/-- Calculates the total cost of Tony's purchases given the specified conditions --/
def total_cost (lego_price : ℝ) (sword_price_eur : ℝ) (dough_price_gbp : ℝ)
                (day1_discount : ℝ) (day2_discount : ℝ) (sales_tax : ℝ)
                (eur_to_usd_day1 : ℝ) (gbp_to_usd_day1 : ℝ)
                (eur_to_usd_day2 : ℝ) (gbp_to_usd_day2 : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the total cost is $1560.83 given the problem conditions --/
theorem tony_purchase_cost :
  let lego_price := 250
  let sword_price_eur := 100
  let dough_price_gbp := 30
  let day1_discount := 0.2
  let day2_discount := 0.1
  let sales_tax := 0.05
  let eur_to_usd_day1 := 1 / 0.85
  let gbp_to_usd_day1 := 1 / 0.75
  let eur_to_usd_day2 := 1 / 0.84
  let gbp_to_usd_day2 := 1 / 0.74
  total_cost lego_price sword_price_eur dough_price_gbp
             day1_discount day2_discount sales_tax
             eur_to_usd_day1 gbp_to_usd_day1
             eur_to_usd_day2 gbp_to_usd_day2 = 1560.83 :=
by
  sorry

end tony_purchase_cost_l3843_384381


namespace solve_equation_l3843_384334

-- Define y as a constant real number
variable (y : ℝ)

-- Define the theorem
theorem solve_equation (x : ℝ) (h : Real.sqrt (x + y - 3) = 10) : x = 103 - y := by
  sorry

end solve_equation_l3843_384334
