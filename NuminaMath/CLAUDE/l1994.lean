import Mathlib

namespace mountain_hike_l1994_199436

theorem mountain_hike (rate_up : ℝ) (time : ℝ) (rate_down_factor : ℝ) :
  rate_up = 8 →
  time = 2 →
  rate_down_factor = 1.5 →
  (rate_up * time) * rate_down_factor = 24 := by
sorry

end mountain_hike_l1994_199436


namespace henry_twice_jills_age_l1994_199458

/-- Given that Henry and Jill's present ages sum to 40, with Henry being 23 and Jill being 17,
    this theorem proves that 11 years ago, Henry was twice the age of Jill. -/
theorem henry_twice_jills_age (henry_age : ℕ) (jill_age : ℕ) :
  henry_age + jill_age = 40 →
  henry_age = 23 →
  jill_age = 17 →
  ∃ (years_ago : ℕ), henry_age - years_ago = 2 * (jill_age - years_ago) ∧ years_ago = 11 := by
  sorry

end henry_twice_jills_age_l1994_199458


namespace arithmetic_mean_of_unknowns_l1994_199469

theorem arithmetic_mean_of_unknowns (x y z : ℝ) 
  (h : (1 : ℝ) / (x * y) = y / (z - x + 1) ∧ y / (z - x + 1) = 2 / (z + 1)) : 
  x = (z + y) / 2 := by
  sorry

end arithmetic_mean_of_unknowns_l1994_199469


namespace fraction_sum_product_equality_l1994_199463

theorem fraction_sum_product_equality (a b c : ℝ) 
  (h1 : 1 + b * c ≠ 0) (h2 : 1 + c * a ≠ 0) (h3 : 1 + a * b ≠ 0) : 
  (b - c) / (1 + b * c) + (c - a) / (1 + c * a) + (a - b) / (1 + a * b) = 
  (b - c) / (1 + b * c) * (c - a) / (1 + c * a) * (a - b) / (1 + a * b) := by
  sorry

end fraction_sum_product_equality_l1994_199463


namespace square_perimeter_ratio_l1994_199408

theorem square_perimeter_ratio (a₁ a₂ p₁ p₂ : ℝ) (h_positive : a₁ > 0 ∧ a₂ > 0) 
  (h_area_ratio : a₁ / a₂ = 49 / 64) (h_perimeter₁ : p₁ = 4 * Real.sqrt a₁) 
  (h_perimeter₂ : p₂ = 4 * Real.sqrt a₂) : p₁ / p₂ = 7 / 8 := by
sorry

end square_perimeter_ratio_l1994_199408


namespace num_non_officers_calculation_l1994_199486

-- Define the problem parameters
def avg_salary_all : ℝ := 120
def avg_salary_officers : ℝ := 420
def avg_salary_non_officers : ℝ := 110
def num_officers : ℕ := 15

-- Define the theorem
theorem num_non_officers_calculation :
  ∃ (num_non_officers : ℕ),
    (num_officers : ℝ) * avg_salary_officers + (num_non_officers : ℝ) * avg_salary_non_officers =
    ((num_officers : ℝ) + (num_non_officers : ℝ)) * avg_salary_all ∧
    num_non_officers = 450 := by
  sorry

end num_non_officers_calculation_l1994_199486


namespace no_x_squared_term_l1994_199454

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 2*a*x + a^2) = x^3 + (a^2 - 2*a)*x + a^2) → a = 1/2 := by
  sorry

end no_x_squared_term_l1994_199454


namespace clarissa_photos_count_l1994_199480

/-- The number of photos brought by Cristina -/
def cristina_photos : ℕ := 7

/-- The number of photos brought by John -/
def john_photos : ℕ := 10

/-- The number of photos brought by Sarah -/
def sarah_photos : ℕ := 9

/-- The total number of slots in the photo album -/
def album_slots : ℕ := 40

/-- The number of photos Clarissa needs to bring -/
def clarissa_photos : ℕ := album_slots - (cristina_photos + john_photos + sarah_photos)

theorem clarissa_photos_count : clarissa_photos = 14 := by
  sorry

end clarissa_photos_count_l1994_199480


namespace circle_combined_value_l1994_199496

/-- The combined value of circumference and area for a circle with radius 13 cm -/
theorem circle_combined_value :
  let r : ℝ := 13
  let π : ℝ := Real.pi
  let circumference : ℝ := 2 * π * r
  let area : ℝ := π * r^2
  abs ((circumference + area) - 612.6105) < 0.0001 := by
sorry

end circle_combined_value_l1994_199496


namespace figure_36_to_square_cut_and_rearrange_to_square_l1994_199487

/-- Represents a figure made up of small squares --/
structure Figure where
  squares : ℕ

/-- Represents a square --/
structure Square where
  side_length : ℕ

/-- Function to check if a figure can be rearranged into a square --/
def can_form_square (f : Figure) : Prop :=
  ∃ (s : Square), s.side_length * s.side_length = f.squares

/-- Theorem stating that a figure with 36 squares can form a square --/
theorem figure_36_to_square :
  ∀ (f : Figure), f.squares = 36 → can_form_square f :=
by
  sorry

/-- Theorem stating that a figure with 36 squares can be cut into two pieces
    and rearranged to form a square --/
theorem cut_and_rearrange_to_square :
  ∀ (f : Figure), f.squares = 36 →
  ∃ (piece1 piece2 : Figure),
    piece1.squares + piece2.squares = f.squares ∧
    can_form_square (Figure.mk (piece1.squares + piece2.squares)) :=
by
  sorry

end figure_36_to_square_cut_and_rearrange_to_square_l1994_199487


namespace perpendicular_tangents_intersection_l1994_199438

open Real

theorem perpendicular_tangents_intersection (a : ℝ) :
  ∃ x ∈ Set.Ioo 0 (π/2),
    (2 * sin x = a * cos x) ∧
    (2 * cos x) * (-a * sin x) = -1 →
  a = 2 * sqrt 3 / 3 := by
sorry

end perpendicular_tangents_intersection_l1994_199438


namespace binary_to_base4_example_l1994_199417

/-- Converts a binary (base 2) number to its base 4 representation -/
def binary_to_base4 (binary : ℕ) : ℕ := sorry

/-- The binary number 1011010010₂ -/
def binary_num : ℕ := 722  -- 1011010010₂ in decimal

/-- Theorem stating that the base 4 representation of 1011010010₂ is 3122₄ -/
theorem binary_to_base4_example : binary_to_base4 binary_num = 3122 := by sorry

end binary_to_base4_example_l1994_199417


namespace max_value_theorem_l1994_199447

/-- The sum of the first m positive even numbers -/
def sumEvenNumbers (m : ℕ) : ℕ := m * (m + 1)

/-- The sum of the first n positive odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- The constraint that the sum of m even numbers and n odd numbers is 1987 -/
def constraint (m n : ℕ) : Prop := sumEvenNumbers m + sumOddNumbers n = 1987

/-- The objective function to be maximized -/
def objective (m n : ℕ) : ℕ := 3 * m + 4 * n

theorem max_value_theorem :
  ∃ m n : ℕ, constraint m n ∧ 
    ∀ m' n' : ℕ, constraint m' n' → objective m' n' ≤ objective m n ∧
    objective m n = 219 :=
sorry

end max_value_theorem_l1994_199447


namespace exponential_increasing_condition_l1994_199426

theorem exponential_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → a^x < a^y) ↔ a > 1 := by sorry

end exponential_increasing_condition_l1994_199426


namespace ninth_term_is_seven_l1994_199483

/-- A sequence where each term is 1/2 more than the previous term -/
def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = a n + 1/2

/-- The 9th term of the arithmetic sequence is 7 -/
theorem ninth_term_is_seven (a : ℕ → ℚ) (h : arithmeticSequence a) : a 9 = 7 := by
  sorry

end ninth_term_is_seven_l1994_199483


namespace smallest_y_cube_sum_l1994_199456

theorem smallest_y_cube_sum (v w x y : ℕ+) : 
  v.val + 1 = w.val → w.val + 1 = x.val → x.val + 1 = y.val →
  v^3 + w^3 + x^3 = y^3 →
  ∀ (z : ℕ+), z < y → ¬(∃ (a b c : ℕ+), a.val + 1 = b.val ∧ b.val + 1 = c.val ∧ c.val + 1 = z.val ∧ a^3 + b^3 + c^3 = z^3) →
  y = 6 := by
sorry

end smallest_y_cube_sum_l1994_199456


namespace waiter_income_fraction_l1994_199403

theorem waiter_income_fraction (salary : ℚ) (tips : ℚ) (income : ℚ) : 
  tips = (3 : ℚ) / 4 * salary →
  income = salary + tips →
  tips / income = (3 : ℚ) / 7 := by
  sorry

end waiter_income_fraction_l1994_199403


namespace rhombus_height_l1994_199405

/-- A rhombus with diagonals of length 6 and 8 has a height of 24/5 -/
theorem rhombus_height (d₁ d₂ h : ℝ) (hd₁ : d₁ = 6) (hd₂ : d₂ = 8) :
  d₁ * d₂ = 4 * h * (d₁^2 / 4 + d₂^2 / 4).sqrt → h = 24 / 5 := by
  sorry

end rhombus_height_l1994_199405


namespace points_form_hyperbola_l1994_199453

/-- The set of points (x,y) defined by x = 2sinh(t) and y = 4cosh(t) for real t forms a hyperbola -/
theorem points_form_hyperbola :
  ∀ (x y t : ℝ), x = 2 * Real.sinh t ∧ y = 4 * Real.cosh t →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1 :=
sorry

end points_form_hyperbola_l1994_199453


namespace factorial_fraction_equals_fifteen_l1994_199470

theorem factorial_fraction_equals_fifteen :
  (Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 8) = 15 := by
  sorry

end factorial_fraction_equals_fifteen_l1994_199470


namespace angle_b_measure_l1994_199449

theorem angle_b_measure (A B C : ℝ) (a b c : ℝ) : 
  0 < B ∧ B < π →
  0 < A ∧ A < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.cos C + c * Real.sin B →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  B = π / 4 := by
sorry

end angle_b_measure_l1994_199449


namespace vaccine_cost_reduction_formula_correct_l1994_199466

/-- Given an initial cost and an annual decrease rate, calculates the cost reduction of producing vaccines after two years. -/
def vaccine_cost_reduction (initial_cost : ℝ) (annual_decrease_rate : ℝ) : ℝ :=
  let cost_last_year := initial_cost * (1 - annual_decrease_rate)
  let cost_this_year := initial_cost * (1 - annual_decrease_rate)^2
  cost_last_year - cost_this_year

/-- Theorem stating that the vaccine cost reduction formula is correct for the given initial cost. -/
theorem vaccine_cost_reduction_formula_correct :
  ∀ (x : ℝ), vaccine_cost_reduction 5000 x = 5000 * x - 5000 * x^2 :=
by
  sorry

#eval vaccine_cost_reduction 5000 0.1

end vaccine_cost_reduction_formula_correct_l1994_199466


namespace female_students_count_l1994_199488

/-- Given a school with stratified sampling, prove the number of female students -/
theorem female_students_count (total_students sample_size : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : ∃ (girls boys : ℕ), girls + boys = sample_size ∧ boys = girls + 10) :
  (760 : ℝ) = (total_students : ℝ) * (95 : ℝ) / (sample_size : ℝ) :=
sorry

end female_students_count_l1994_199488


namespace cookie_cutter_sides_l1994_199492

/-- The number of sides on a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides on a square -/
def square_sides : ℕ := 4

/-- The number of sides on a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of triangle-shaped cookie cutters -/
def num_triangles : ℕ := 6

/-- The number of square-shaped cookie cutters -/
def num_squares : ℕ := 4

/-- The number of hexagon-shaped cookie cutters -/
def num_hexagons : ℕ := 2

/-- The total number of sides on all cookie cutters -/
def total_sides : ℕ := num_triangles * triangle_sides + num_squares * square_sides + num_hexagons * hexagon_sides

theorem cookie_cutter_sides : total_sides = 46 := by
  sorry

end cookie_cutter_sides_l1994_199492


namespace balloon_distribution_l1994_199464

theorem balloon_distribution (red white green chartreuse : ℕ) (friends : ℕ) : 
  red = 25 → white = 40 → green = 55 → chartreuse = 80 → friends = 10 →
  (red + white + green + chartreuse) % friends = 0 :=
by
  sorry

end balloon_distribution_l1994_199464


namespace fraction_equality_l1994_199427

theorem fraction_equality (a b c : ℝ) (h1 : a = b) (h2 : c ≠ 0) : a / c = b / c := by
  sorry

end fraction_equality_l1994_199427


namespace weight_difference_l1994_199468

theorem weight_difference (steve jim stan : ℕ) : 
  stan = steve + 5 →
  jim = 110 →
  steve + stan + jim = 319 →
  jim - steve = 8 := by
sorry

end weight_difference_l1994_199468


namespace nested_bracket_equals_two_l1994_199460

/-- Defines the operation [a,b,c] as (a+b)/c for c ≠ 0 -/
def bracket (a b c : ℚ) : ℚ := (a + b) / c

/-- The main theorem to prove -/
theorem nested_bracket_equals_two :
  bracket (bracket 120 60 180) (bracket 4 2 6) (bracket 20 10 30) = 2 := by
  sorry


end nested_bracket_equals_two_l1994_199460


namespace intersection_line_of_planes_l1994_199424

/-- Represents a plane with its first trace and angle of inclination -/
structure Plane where
  firstTrace : Line2D
  inclinationAngle : ℝ

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  point1 : Point2D
  point2 : Point2D

/-- Finds the intersection point of two 2D lines -/
def intersectionPoint (l1 l2 : Line2D) : Point2D :=
  sorry

/-- Constructs a point using the angles of inclination -/
def constructPoint (p1 p2 : Plane) : Point2D :=
  sorry

/-- Theorem stating that the intersection line of two planes can be determined
    by connecting two specific points -/
theorem intersection_line_of_planes (p1 p2 : Plane) :
  ∃ (l : Line2D),
    l.point1 = intersectionPoint p1.firstTrace p2.firstTrace ∧
    l.point2 = constructPoint p1 p2 :=
  sorry

end intersection_line_of_planes_l1994_199424


namespace remaining_work_days_for_z_l1994_199490

-- Define work rates for each person
def work_rate_x : ℚ := 1 / 5
def work_rate_y : ℚ := 1 / 20
def work_rate_z : ℚ := 1 / 30

-- Define the total work as 1 (100%)
def total_work : ℚ := 1

-- Define the number of days all three work together
def days_together : ℚ := 2

-- Theorem statement
theorem remaining_work_days_for_z :
  let combined_rate := work_rate_x + work_rate_y + work_rate_z
  let work_done_together := combined_rate * days_together
  let remaining_work := total_work - work_done_together
  (remaining_work / work_rate_z : ℚ) = 13 := by
  sorry

end remaining_work_days_for_z_l1994_199490


namespace complex_number_location_l1994_199493

theorem complex_number_location :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_location_l1994_199493


namespace probability_nine_correct_l1994_199481

/-- The number of English-Russian expression pairs to be matched -/
def total_pairs : ℕ := 10

/-- The number of correctly matched pairs we're interested in -/
def correct_matches : ℕ := 9

/-- Represents the probability of getting exactly 9 out of 10 matches correct when choosing randomly -/
def prob_nine_correct : ℝ := 0

/-- Theorem stating that the probability of getting exactly 9 out of 10 matches correct when choosing randomly is 0 -/
theorem probability_nine_correct :
  prob_nine_correct = 0 := by sorry

end probability_nine_correct_l1994_199481


namespace quadratic_intersects_twice_iff_k_condition_l1994_199482

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The discriminant of a quadratic function ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

/-- Predicate for a quadratic function intersecting x-axis at two points -/
def intersects_twice (a b c : ℝ) : Prop :=
  discriminant a b c > 0 ∧ a ≠ 0

theorem quadratic_intersects_twice_iff_k_condition (k : ℝ) :
  intersects_twice (k - 2) (-(2 * k - 1)) k ↔ k > -1/4 ∧ k ≠ 2 :=
sorry

end quadratic_intersects_twice_iff_k_condition_l1994_199482


namespace john_final_height_l1994_199433

/-- Calculates the final height in feet given initial height, growth rate, and duration -/
def final_height_in_feet (initial_height : ℕ) (growth_rate : ℕ) (duration : ℕ) : ℚ :=
  (initial_height + growth_rate * duration) / 12

/-- Theorem stating that given the specific conditions, the final height is 6 feet -/
theorem john_final_height :
  final_height_in_feet 66 2 3 = 6 := by sorry

end john_final_height_l1994_199433


namespace article_pricing_l1994_199477

theorem article_pricing (P : ℝ) (P_pos : P > 0) : 
  (2/3 * P = 0.9 * ((2/3 * P) / 0.9)) → 
  ((P - ((2/3 * P) / 0.9)) / ((2/3 * P) / 0.9)) * 100 = 35 := by
  sorry

end article_pricing_l1994_199477


namespace combination_permutation_equality_permutation_equation_solution_l1994_199448

-- Define the combination function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the permutation function
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Theorem 1: Prove that C₁₀⁴ - C₇³ × A₃³ = 0
theorem combination_permutation_equality : C 10 4 - C 7 3 * A 3 3 = 0 := by
  sorry

-- Theorem 2: Prove that the solution to 3A₈ˣ = 4A₉ˣ⁻¹ is x = 6
theorem permutation_equation_solution :
  ∃ x : ℕ, (3 * A 8 x = 4 * A 9 (x - 1)) ∧ x = 6 := by
  sorry

end combination_permutation_equality_permutation_equation_solution_l1994_199448


namespace notebooks_promotion_result_l1994_199430

/-- Calculates the maximum number of notebooks obtainable given an initial amount of money,
    the cost per notebook, and a promotion where stickers can be exchanged for free notebooks. -/
def max_notebooks (initial_money : ℕ) (cost_per_notebook : ℕ) (stickers_per_free_notebook : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given 150 rubles, with notebooks costing 4 rubles each,
    and a promotion where 5 stickers can be exchanged for an additional notebook
    (each notebook comes with a sticker), the maximum number of notebooks obtainable is 46. -/
theorem notebooks_promotion_result :
  max_notebooks 150 4 5 = 46 := by
  sorry

end notebooks_promotion_result_l1994_199430


namespace repeating_decimal_sum_l1994_199425

-- Define repeating decimals
def repeating_decimal_6 : ℚ := 2/3
def repeating_decimal_4 : ℚ := 4/9
def repeating_decimal_8 : ℚ := 8/9

-- Theorem statement
theorem repeating_decimal_sum :
  repeating_decimal_6 - repeating_decimal_4 + repeating_decimal_8 = 10/9 := by
  sorry

end repeating_decimal_sum_l1994_199425


namespace rectangle_diagonal_triangle_area_l1994_199402

/-- The area of a right triangle formed by the diagonal of a rectangle. -/
theorem rectangle_diagonal_triangle_area
  (length width : ℝ)
  (h_length : length = 35)
  (h_width : width = 48) :
  (1 / 2) * length * width = 840 := by
  sorry

end rectangle_diagonal_triangle_area_l1994_199402


namespace sugar_consumption_reduction_l1994_199414

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 6)
  (h2 : new_price = 7.5) : 
  (1 - initial_price / new_price) * 100 = 20 := by
  sorry

end sugar_consumption_reduction_l1994_199414


namespace cookies_eaten_by_adults_l1994_199435

/-- Proves that the number of cookies eaten by adults is 40 --/
theorem cookies_eaten_by_adults (total_cookies : ℕ) (num_children : ℕ) (child_cookies : ℕ) : 
  total_cookies = 120 →
  num_children = 4 →
  child_cookies = 20 →
  (total_cookies - num_children * child_cookies : ℚ) = (1/3 : ℚ) * total_cookies :=
by
  sorry

#check cookies_eaten_by_adults

end cookies_eaten_by_adults_l1994_199435


namespace system_solution_conditions_l1994_199429

theorem system_solution_conditions (m : ℝ) :
  let x := (1 + 2*m) / 3
  let y := (1 - m) / 3
  (x + 2*y = 1 ∧ x - y = m) ∧ (x > 1 ∧ y ≥ -1) ↔ 1 < m ∧ m ≤ 4 := by
  sorry

end system_solution_conditions_l1994_199429


namespace reflect_center_l1994_199439

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflect_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center : ℝ × ℝ := reflect_about_y_eq_neg_x original_center
  reflected_center = (3, -8) := by sorry

end reflect_center_l1994_199439


namespace square_diagonals_properties_l1994_199491

structure Square where
  diagonals_perpendicular : Prop
  diagonals_equal : Prop

theorem square_diagonals_properties (s : Square) :
  (s.diagonals_perpendicular ∨ s.diagonals_equal) ∧
  (s.diagonals_perpendicular ∧ s.diagonals_equal) ∧
  ¬(¬s.diagonals_perpendicular) := by
  sorry

end square_diagonals_properties_l1994_199491


namespace initial_dozens_of_doughnuts_l1994_199401

theorem initial_dozens_of_doughnuts (eaten : ℕ) (left : ℕ) (dozen : ℕ) : 
  eaten = 8 → left = 16 → dozen = 12 → (eaten + left) / dozen = 2 := by
  sorry

end initial_dozens_of_doughnuts_l1994_199401


namespace odd_prime_sum_divisors_count_l1994_199415

/-- Sum of positive integer divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Predicate for odd prime numbers -/
def is_odd_prime (n : ℕ) : Prop := sorry

/-- Count of numbers with odd prime sum of divisors -/
def count_odd_prime_sum_divisors : ℕ := sorry

theorem odd_prime_sum_divisors_count :
  count_odd_prime_sum_divisors = 5 := by sorry

end odd_prime_sum_divisors_count_l1994_199415


namespace cube_edge_length_l1994_199444

/-- Given a cube with volume V, surface area S, and edge length a, 
    where V = S + 1, prove that a satisfies a³ - 6a² - 1 = 0 
    and the solution is closest to 6 -/
theorem cube_edge_length (V S a : ℝ) (hV : V = a^3) (hS : S = 6*a^2) (hVS : V = S + 1) :
  a^3 - 6*a^2 - 1 = 0 ∧ ∃ ε > 0, ∀ x : ℝ, x ≠ a → |x - 6| > |a - 6| - ε :=
sorry

end cube_edge_length_l1994_199444


namespace red_light_time_proof_l1994_199434

/-- Represents the time added by each red light -/
def time_per_red_light : ℕ := sorry

/-- Time for the first route with all green lights -/
def green_route_time : ℕ := 10

/-- Time for the second route -/
def second_route_time : ℕ := 14

/-- Number of stoplights on the first route -/
def num_stoplights : ℕ := 3

theorem red_light_time_proof :
  (green_route_time + num_stoplights * time_per_red_light = second_route_time + 5) ∧
  (time_per_red_light = 3) := by sorry

end red_light_time_proof_l1994_199434


namespace quadratic_inequality_properties_l1994_199420

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ :=
  {x | x ≤ -2 ∨ x ≥ 6}

-- Define the quadratic inequality
def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c ≤ 0

-- Theorem statement
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, x ∈ solution_set a b c ↔ quadratic_inequality a b c x) :
  a < 0 ∧
  (∀ x, -1/6 < x ∧ x < 1/2 ↔ c * x^2 - b * x + a < 0) ∧
  a + b + c > 0 :=
by sorry

end quadratic_inequality_properties_l1994_199420


namespace smallest_reciprocal_sum_l1994_199498

/-- Given a quadratic equation x^2 - s*x + p with roots r₁ and r₂ -/
def quadratic_equation (s p : ℝ) (x : ℝ) : ℝ := x^2 - s*x + p

/-- The sum of powers of roots is constant for powers 1 to 1004 -/
def sum_powers_constant (r₁ r₂ : ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1004 → r₁^n + r₂^n = r₁ + r₂

/-- The theorem stating the smallest possible value of 1/r₁^1005 + 1/r₂^1005 -/
theorem smallest_reciprocal_sum (s p r₁ r₂ : ℝ) :
  (∀ x : ℝ, quadratic_equation s p x = 0 ↔ x = r₁ ∨ x = r₂) →
  sum_powers_constant r₁ r₂ →
  (∃ v : ℝ, v = 1/r₁^1005 + 1/r₂^1005 ∧ ∀ w : ℝ, w = 1/r₁^1005 + 1/r₂^1005 → v ≤ w) →
  ∃ v : ℝ, v = 2 ∧ v = 1/r₁^1005 + 1/r₂^1005 := by
  sorry

end smallest_reciprocal_sum_l1994_199498


namespace subway_bike_speed_ratio_l1994_199499

/-- The speed of the mountain bike in km/h -/
def bike_speed : ℝ := sorry

/-- The speed of the subway in km/h -/
def subway_speed : ℝ := sorry

/-- The time taken to ride the bike initially in minutes -/
def initial_bike_time : ℝ := 10

/-- The time taken by subway in minutes -/
def subway_time : ℝ := 40

/-- The total time taken when riding the bike for the entire journey in hours -/
def total_bike_time : ℝ := 3.5

theorem subway_bike_speed_ratio : 
  subway_speed = 5 * bike_speed :=
sorry

end subway_bike_speed_ratio_l1994_199499


namespace janets_height_l1994_199411

/-- Given the heights of various people, prove Janet's height --/
theorem janets_height :
  ∀ (ruby pablo charlene janet : ℝ),
  ruby = pablo - 2 →
  pablo = charlene + 70 →
  charlene = 2 * janet →
  ruby = 192 →
  janet = 62 := by
sorry

end janets_height_l1994_199411


namespace alyssa_fruit_spending_l1994_199423

theorem alyssa_fruit_spending (total_spent cherries_cost : ℚ)
  (h1 : total_spent = 21.93)
  (h2 : cherries_cost = 9.85) :
  total_spent - cherries_cost = 12.08 := by
  sorry

end alyssa_fruit_spending_l1994_199423


namespace three_not_in_range_iff_c_gt_four_l1994_199497

/-- The function g(x) = x^2 + 2x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c

/-- 3 is not in the range of g(x) if and only if c > 4 -/
theorem three_not_in_range_iff_c_gt_four (c : ℝ) :
  (∀ x : ℝ, g c x ≠ 3) ↔ c > 4 := by
  sorry

end three_not_in_range_iff_c_gt_four_l1994_199497


namespace lucky_lacy_correct_percentage_l1994_199473

theorem lucky_lacy_correct_percentage (x : ℕ) : 
  let total_problems : ℕ := 4 * x
  let missed_problems : ℕ := 2 * x
  let correct_problems : ℕ := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) * 100 = 50 := by
  sorry

end lucky_lacy_correct_percentage_l1994_199473


namespace plywood_perimeter_difference_l1994_199495

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its cutting --/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Generates all possible ways to cut the plywood into congruent rectangles --/
def possible_cuts (p : Plywood) : List Rectangle :=
  sorry -- Implementation details omitted

/-- Finds the maximum perimeter among the possible cuts --/
def max_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

/-- Finds the minimum perimeter among the possible cuts --/
def min_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

theorem plywood_perimeter_difference :
  let p : Plywood := { length := 6, width := 9, num_pieces := 6 }
  let cuts := possible_cuts p
  max_perimeter cuts - min_perimeter cuts = 11 := by
  sorry

end plywood_perimeter_difference_l1994_199495


namespace increasing_condition_l1994_199428

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x + 5

theorem increasing_condition (m : ℝ) :
  (∀ x ∈ Set.Ici (-2 : ℝ), Monotone (fun x => f m x)) ↔ m ∈ Set.Icc 0 (1/4) :=
sorry

end increasing_condition_l1994_199428


namespace derivative_f_at_zero_dne_l1994_199484

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan x * Real.sin (7 / x) else 0

theorem derivative_f_at_zero_dne :
  ¬ DifferentiableAt ℝ f 0 := by sorry

end derivative_f_at_zero_dne_l1994_199484


namespace trigonometric_identity_l1994_199421

theorem trigonometric_identity (α β γ : Real) :
  Real.sin α + Real.sin β + Real.sin γ - 
  Real.sin (α + β) * Real.cos γ - Real.cos (α + β) * Real.sin γ = 
  4 * Real.sin ((α + β) / 2) * Real.sin ((β + γ) / 2) * Real.sin ((γ + α) / 2) := by
sorry

end trigonometric_identity_l1994_199421


namespace refrigerator_production_days_l1994_199459

/-- The number of additional days needed to complete refrigerator production -/
def additional_days (total_required : ℕ) (days_worked : ℕ) (initial_rate : ℕ) (increased_rate : ℕ) : ℕ :=
  let produced := days_worked * initial_rate
  let remaining := total_required - produced
  remaining / increased_rate

theorem refrigerator_production_days : 
  additional_days 1590 12 80 90 = 7 := by
  sorry

end refrigerator_production_days_l1994_199459


namespace three_diamonds_balance_six_dots_l1994_199494

-- Define the symbols
variable (triangle diamond dot : ℕ)

-- Define the balance relation
def balances (left right : ℕ) : Prop := left = right

-- State the given conditions
axiom balance1 : balances (4 * triangle + 2 * diamond) (12 * dot)
axiom balance2 : balances (2 * triangle) (diamond + 2 * dot)

-- State the theorem to be proved
theorem three_diamonds_balance_six_dots : balances (3 * diamond) (6 * dot) := by
  sorry

end three_diamonds_balance_six_dots_l1994_199494


namespace sammy_has_eight_caps_l1994_199452

/-- The number of bottle caps Billie has -/
def billies_caps : ℕ := 2

/-- The number of bottle caps Janine has -/
def janines_caps : ℕ := 3 * billies_caps

/-- The number of bottle caps Sammy has -/
def sammys_caps : ℕ := janines_caps + 2

/-- Theorem stating that Sammy has 8 bottle caps -/
theorem sammy_has_eight_caps : sammys_caps = 8 := by
  sorry

end sammy_has_eight_caps_l1994_199452


namespace seventh_term_ratio_l1994_199446

/-- Two arithmetic sequences and their properties -/
structure ArithmeticSequencePair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum of first n terms of sequence a
  T : ℕ → ℚ  -- Sum of first n terms of sequence b
  h : ∀ n, S n / T n = (3 * n - 2) / (2 * n + 1)  -- Given condition

/-- Theorem stating the relation between the 7th terms of the sequences -/
theorem seventh_term_ratio (seq : ArithmeticSequencePair) : seq.a 7 / seq.b 7 = 37 / 27 := by
  sorry

end seventh_term_ratio_l1994_199446


namespace inscribed_circles_area_ratio_l1994_199400

theorem inscribed_circles_area_ratio (s : ℝ) (hs : s > 0) :
  let square_area := s^2
  let semicircle_area := (π * s^2) / 8
  let quarter_circle_area := (π * s^2) / 16
  let combined_area := semicircle_area + quarter_circle_area
  combined_area / square_area = 3 * π / 16 := by
sorry

end inscribed_circles_area_ratio_l1994_199400


namespace sum_divisible_by_five_l1994_199407

theorem sum_divisible_by_five (m : ℤ) : 5 ∣ ((10 - m) + (m + 5)) := by
  sorry

end sum_divisible_by_five_l1994_199407


namespace six_digit_same_digits_prime_divisor_sum_l1994_199474

def is_six_digit_same_digits (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ ∃ d : ℕ, n = d * 111111

def sum_distinct_prime_divisors (n : ℕ) : ℕ := sorry

theorem six_digit_same_digits_prime_divisor_sum 
  (n : ℕ) (h : is_six_digit_same_digits n) : 
  sum_distinct_prime_divisors n ≠ 70 ∧ sum_distinct_prime_divisors n ≠ 80 := by
  sorry

end six_digit_same_digits_prime_divisor_sum_l1994_199474


namespace even_number_divisibility_property_l1994_199440

theorem even_number_divisibility_property (n : ℕ) :
  n % 2 = 0 →
  (∀ p : ℕ, Prime p → p ∣ n → (p - 1) ∣ (n - 1)) →
  ∃ k : ℕ, n = 2^k :=
sorry

end even_number_divisibility_property_l1994_199440


namespace complex_magnitude_problem_l1994_199461

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) / z = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l1994_199461


namespace cone_surface_area_ratio_l1994_199450

theorem cone_surface_area_ratio (r l : ℝ) (h : l = 4 * r) :
  let side_area := (1 / 2) * Real.pi * l ^ 2
  let base_area := Real.pi * r ^ 2
  let total_area := side_area + base_area
  (total_area / side_area) = 5 / 4 := by
sorry

end cone_surface_area_ratio_l1994_199450


namespace max_sum_constrained_squares_l1994_199422

theorem max_sum_constrained_squares (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m^2 + n^2 = 100) :
  m + n ≤ 10 * Real.sqrt 2 := by
sorry

end max_sum_constrained_squares_l1994_199422


namespace fraction_evaluation_l1994_199416

theorem fraction_evaluation (a b c : ℝ) (h : a^3 - b^3 + c^3 ≠ 0) :
  (a^6 - b^6 + c^6) / (a^3 - b^3 + c^3) = a^3 + b^3 + c^3 := by
  sorry

end fraction_evaluation_l1994_199416


namespace cannot_eat_all_except_central_l1994_199465

/-- Represents a 3D coordinate within the cheese cube -/
structure Coordinate where
  x : Fin 3
  y : Fin 3
  z : Fin 3

/-- Represents the color of a unit cube -/
inductive Color
  | White
  | Black

/-- The cheese cube -/
def CheeseCube := Fin 3 → Fin 3 → Fin 3 → Color

/-- Determines if two coordinates are adjacent (share a face) -/
def isAdjacent (c1 c2 : Coordinate) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z + 1 = c2.z)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y + 1 = c2.y)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x + 1 = c2.x))

/-- Assigns a color to each coordinate based on the sum of its components -/
def colorCube : CheeseCube :=
  fun x y z => if (x.val + y.val + z.val) % 2 = 0 then Color.White else Color.Black

/-- The central cube coordinate -/
def centralCube : Coordinate := ⟨1, 1, 1⟩

/-- Theorem stating that it's impossible to eat all cubes except the central one -/
theorem cannot_eat_all_except_central :
  ¬∃ (path : List Coordinate),
    path.Nodup ∧
    path.length = 26 ∧
    (∀ i, i ∈ path → i ≠ centralCube) ∧
    (∀ i j, i ∈ path → j ∈ path → i ≠ j → isAdjacent i j) :=
  sorry

end cannot_eat_all_except_central_l1994_199465


namespace train_journey_distance_l1994_199442

/-- Represents the train journey problem -/
def TrainJourney (x v : ℝ) : Prop :=
  -- Train stops after 1 hour and remains halted for 0.5 hours
  let initial_stop_time : ℝ := 1.5
  -- Train continues at 3/4 of original speed
  let reduced_speed : ℝ := 3/4 * v
  -- Total delay equation
  let delay_equation : Prop := (x/v + initial_stop_time + (x-v)/reduced_speed - x/v = 3.5)
  -- Equation for incident 90 miles further
  let further_incident_equation : Prop := 
    ((x-90)/v + initial_stop_time + (x-90)/reduced_speed - x/v + 90/v = 3)
  
  -- All conditions must be satisfied
  delay_equation ∧ further_incident_equation

/-- The theorem to be proved -/
theorem train_journey_distance : 
  ∃ (v : ℝ), TrainJourney 600 v := by sorry

end train_journey_distance_l1994_199442


namespace solve_cafeteria_problem_l1994_199478

/-- Represents the amount paid by each friend in kopecks -/
structure Payment where
  misha : ℕ
  sasha : ℕ
  grisha : ℕ

/-- Represents the number of dishes each friend paid for -/
structure Dishes where
  misha : ℕ
  sasha : ℕ
  total : ℕ

def cafeteria_problem (p : Payment) (d : Dishes) : Prop :=
  -- All dishes cost the same
  ∃ (dish_cost : ℕ),
  -- Misha paid for 3 dishes
  p.misha = d.misha * dish_cost ∧
  -- Sasha paid for 2 dishes
  p.sasha = d.sasha * dish_cost ∧
  -- Together they ate 5 dishes
  d.total = d.misha + d.sasha ∧
  -- Grisha should pay his friends a total of 50 kopecks
  p.grisha = 50 ∧
  -- Each friend should receive an equal payment
  p.misha + p.sasha + p.grisha = d.total * dish_cost ∧
  -- Prove that Grisha should pay 40 kopecks to Misha and 10 kopecks to Sasha
  p.misha - (p.misha + p.sasha + p.grisha) / 3 = 40 ∧
  p.sasha - (p.misha + p.sasha + p.grisha) / 3 = 10

theorem solve_cafeteria_problem :
  ∃ (p : Payment) (d : Dishes), cafeteria_problem p d :=
sorry

end solve_cafeteria_problem_l1994_199478


namespace not_perfect_square_l1994_199437

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), 4 * n^2 + 4 * n + 4 = m^2 := by
  sorry

end not_perfect_square_l1994_199437


namespace peak_speed_scientific_notation_l1994_199467

/-- The peak computing speed of a certain server in operations per second. -/
def peak_speed : ℕ := 403200000000

/-- The scientific notation representation of the peak speed. -/
def scientific_notation : ℝ := 4.032 * (10 ^ 11)

/-- Theorem stating that the peak speed is equal to its scientific notation representation. -/
theorem peak_speed_scientific_notation : (peak_speed : ℝ) = scientific_notation := by
  sorry

end peak_speed_scientific_notation_l1994_199467


namespace great_pyramid_height_l1994_199475

theorem great_pyramid_height (h w : ℝ) : 
  h > 500 → 
  w = h + 234 → 
  h + w = 1274 → 
  h - 500 = 20 := by
sorry

end great_pyramid_height_l1994_199475


namespace trick_decks_cost_l1994_199489

def price_per_deck (quantity : ℕ) : ℕ :=
  if quantity ≤ 3 then 8
  else if quantity ≤ 6 then 7
  else 6

def total_cost (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  victor_decks * price_per_deck victor_decks + friend_decks * price_per_deck friend_decks

theorem trick_decks_cost (victor_decks friend_decks : ℕ) 
  (h1 : victor_decks = 6) (h2 : friend_decks = 2) : 
  total_cost victor_decks friend_decks = 58 := by
  sorry

end trick_decks_cost_l1994_199489


namespace janous_inequality_l1994_199445

theorem janous_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
by sorry

end janous_inequality_l1994_199445


namespace system_consistency_solution_values_l1994_199431

def is_consistent (a : ℝ) : Prop :=
  ∃ x : ℝ, 3 * x^2 - x - a - 10 = 0 ∧ (a + 4) * x + a + 12 = 0

theorem system_consistency :
  ∀ a : ℝ, is_consistent a ↔ (a = -10 ∨ a = -8 ∨ a = 4) :=
sorry

theorem solution_values :
  (is_consistent (-10) ∧ ∃ x : ℝ, 3 * x^2 - x - (-10) - 10 = 0 ∧ (-10 + 4) * x + (-10) + 12 = 0 ∧ x = -1/3) ∧
  (is_consistent (-8) ∧ ∃ x : ℝ, 3 * x^2 - x - (-8) - 10 = 0 ∧ (-8 + 4) * x + (-8) + 12 = 0 ∧ x = -1) ∧
  (is_consistent 4 ∧ ∃ x : ℝ, 3 * x^2 - x - 4 - 10 = 0 ∧ (4 + 4) * x + 4 + 12 = 0 ∧ x = -2) :=
sorry

end system_consistency_solution_values_l1994_199431


namespace fountain_distance_is_30_l1994_199457

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def fountain_distance (total_distance : ℕ) (num_trips : ℕ) : ℚ :=
  (total_distance : ℚ) / (2 * num_trips)

/-- Theorem stating that the fountain distance is 30 feet -/
theorem fountain_distance_is_30 :
  fountain_distance 120 4 = 30 := by
  sorry

end fountain_distance_is_30_l1994_199457


namespace absolute_value_equation_unique_solution_l1994_199409

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x^2 + x - 2| :=
sorry

end absolute_value_equation_unique_solution_l1994_199409


namespace basket_count_l1994_199419

theorem basket_count (apples_per_basket : ℕ) (total_apples : ℕ) (h1 : apples_per_basket = 17) (h2 : total_apples = 629) :
  total_apples / apples_per_basket = 37 := by
sorry

end basket_count_l1994_199419


namespace perfect_square_condition_l1994_199472

/-- The expression (19a + b)^18 + (a + b)^18 + (a + 19b)^18 is a perfect square if and only if a = 0 and b = 0, where a and b are integers. -/
theorem perfect_square_condition (a b : ℤ) : 
  (∃ (k : ℤ), (19*a + b)^18 + (a + b)^18 + (a + 19*b)^18 = k^2) ↔ (a = 0 ∧ b = 0) := by
  sorry

#check perfect_square_condition

end perfect_square_condition_l1994_199472


namespace quadratic_root_square_implies_s_l1994_199410

theorem quadratic_root_square_implies_s (r s : ℝ) :
  (∃ x : ℂ, 3 * x^2 + r * x + s = 0 ∧ x^2 = 4 - 3*I) →
  s = 15 := by
sorry

end quadratic_root_square_implies_s_l1994_199410


namespace f_sum_positive_l1994_199413

def f (x : ℝ) : ℝ := x^2015

theorem f_sum_positive (a b : ℝ) (h1 : a + b > 0) (h2 : a * b < 0) : 
  f a + f b > 0 := by
  sorry

end f_sum_positive_l1994_199413


namespace most_reasonable_sampling_methods_l1994_199485

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a sampling survey --/
structure Survey where
  totalItems : ℕ
  sampleSize : ℕ
  hasStrata : Bool
  hasStructure : Bool

/-- Determines the most reasonable sampling method for a given survey --/
def mostReasonableSamplingMethod (s : Survey) : SamplingMethod :=
  if s.hasStrata then SamplingMethod.Stratified
  else if s.hasStructure then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

/-- The three surveys described in the problem --/
def survey1 : Survey := { totalItems := 15, sampleSize := 5, hasStrata := false, hasStructure := false }
def survey2 : Survey := { totalItems := 240, sampleSize := 20, hasStrata := true, hasStructure := false }
def survey3 : Survey := { totalItems := 25 * 38, sampleSize := 25, hasStrata := false, hasStructure := true }

/-- Theorem stating the most reasonable sampling methods for the given surveys --/
theorem most_reasonable_sampling_methods :
  (mostReasonableSamplingMethod survey1 = SamplingMethod.SimpleRandom) ∧
  (mostReasonableSamplingMethod survey2 = SamplingMethod.Stratified) ∧
  (mostReasonableSamplingMethod survey3 = SamplingMethod.Systematic) :=
sorry


end most_reasonable_sampling_methods_l1994_199485


namespace sheridan_fish_proof_l1994_199479

/-- The number of fish Mrs. Sheridan gave to her sister -/
def fish_given_away : ℝ := 22.0

/-- The number of fish Mrs. Sheridan has now -/
def fish_remaining : ℕ := 25

/-- The initial number of fish Mrs. Sheridan had -/
def initial_fish : ℝ := fish_given_away + fish_remaining

theorem sheridan_fish_proof : initial_fish = 47.0 := by
  sorry

end sheridan_fish_proof_l1994_199479


namespace floor_difference_equals_eight_l1994_199404

theorem floor_difference_equals_eight :
  ⌊(101^3 : ℝ) / (99 * 100) - (99^3 : ℝ) / (100 * 101)⌋ = 8 := by
  sorry

end floor_difference_equals_eight_l1994_199404


namespace floor_sqrt_50_squared_l1994_199451

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end floor_sqrt_50_squared_l1994_199451


namespace max_gcd_13n_plus_4_8n_plus_3_l1994_199443

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) = 11) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 11) :=
by sorry

end max_gcd_13n_plus_4_8n_plus_3_l1994_199443


namespace inequality_solution_set_l1994_199476

theorem inequality_solution_set (x : ℝ) : -x^2 + 4*x - 3 ≥ 0 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end inequality_solution_set_l1994_199476


namespace min_value_exponential_sum_equality_condition_l1994_199471

theorem min_value_exponential_sum (a b : ℝ) (h : a + 2 * b + 3 = 0) :
  2^a + 4^b ≥ Real.sqrt 2 / 2 :=
by sorry

theorem equality_condition (a b : ℝ) (h : a + 2 * b + 3 = 0) :
  ∃ (a₀ b₀ : ℝ), a₀ + 2 * b₀ + 3 = 0 ∧ 2^a₀ + 4^b₀ = Real.sqrt 2 / 2 :=
by sorry

end min_value_exponential_sum_equality_condition_l1994_199471


namespace geometric_sequence_min_value_l1994_199455

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = q * a n ∧ a n > 0

/-- The theorem statement -/
theorem geometric_sequence_min_value
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_eq : a 7 = a 6 + 2 * a 5)
  (h_sqrt : ∃ m n : ℕ, Real.sqrt (a m * a n) = 2 * a 1) :
  (∃ m n : ℕ, (1 : ℝ) / m + 9 / n = 4) ∧
  (∀ m n : ℕ, (1 : ℝ) / m + 9 / n ≥ 4) := by
  sorry


end geometric_sequence_min_value_l1994_199455


namespace complex_in_fourth_quadrant_l1994_199462

-- Define the complex number z
def z (a : ℝ) : ℂ := (2 - Complex.I) * (2 + a * Complex.I)

-- Define the condition for z to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem complex_in_fourth_quadrant :
  ∃ a : ℝ, in_fourth_quadrant (z a) ∧ a = -2 :=
sorry

end complex_in_fourth_quadrant_l1994_199462


namespace emily_small_gardens_l1994_199441

/-- The number of small gardens Emily had -/
def num_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Proof that Emily had 3 small gardens -/
theorem emily_small_gardens :
  num_small_gardens 41 29 4 = 3 := by
  sorry

end emily_small_gardens_l1994_199441


namespace total_rainfall_2011_2012_l1994_199406

/-- Represents the average monthly rainfall in millimeters for a given year. -/
def AverageMonthlyRainfall : ℕ → ℝ
  | 2010 => 50.0
  | 2011 => AverageMonthlyRainfall 2010 + 3
  | 2012 => AverageMonthlyRainfall 2011 + 4
  | _ => 0  -- Default case for other years

/-- Calculates the total yearly rainfall given the average monthly rainfall. -/
def YearlyRainfall (year : ℕ) : ℝ :=
  AverageMonthlyRainfall year * 12

/-- Theorem stating the total rainfall in Clouddale for 2011 and 2012. -/
theorem total_rainfall_2011_2012 :
  YearlyRainfall 2011 + YearlyRainfall 2012 = 1320.0 := by
  sorry

#eval YearlyRainfall 2011 + YearlyRainfall 2012

end total_rainfall_2011_2012_l1994_199406


namespace petunias_per_flat_is_8_l1994_199418

/-- Represents the number of petunias in each flat -/
def petunias_per_flat : ℕ := sorry

/-- The total number of flats of petunias -/
def petunia_flats : ℕ := 4

/-- The total number of flats of roses -/
def rose_flats : ℕ := 3

/-- The number of roses in each flat -/
def roses_per_flat : ℕ := 6

/-- The number of Venus flytraps -/
def venus_flytraps : ℕ := 2

/-- The amount of fertilizer needed for each petunia (in ounces) -/
def fertilizer_per_petunia : ℕ := 8

/-- The amount of fertilizer needed for each rose (in ounces) -/
def fertilizer_per_rose : ℕ := 3

/-- The amount of fertilizer needed for each Venus flytrap (in ounces) -/
def fertilizer_per_flytrap : ℕ := 2

/-- The total amount of fertilizer needed (in ounces) -/
def total_fertilizer : ℕ := 314

theorem petunias_per_flat_is_8 :
  petunias_per_flat = 8 :=
by sorry

end petunias_per_flat_is_8_l1994_199418


namespace expected_value_coin_flip_l1994_199432

/-- The expected value of a coin flip game -/
theorem expected_value_coin_flip :
  let p_heads : ℚ := 2/5
  let p_tails : ℚ := 3/5
  let win_amount : ℚ := 5
  let loss_amount : ℚ := 4
  let expected_value := p_heads * win_amount - p_tails * loss_amount
  expected_value = -2/5
:= by sorry

end expected_value_coin_flip_l1994_199432


namespace positive_integers_satisfying_conditions_l1994_199412

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_8 n ∧ sum_of_digits n = 7 ∧ product_of_digits n = 6

theorem positive_integers_satisfying_conditions :
  {n : ℕ | n > 0 ∧ satisfies_conditions n} = {1312, 3112} :=
sorry

end positive_integers_satisfying_conditions_l1994_199412
