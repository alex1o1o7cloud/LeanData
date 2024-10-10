import Mathlib

namespace set_union_problem_l1930_193041

theorem set_union_problem (x y : ℝ) :
  let A : Set ℝ := {x, y}
  let B : Set ℝ := {x + 1, 5}
  A ∩ B = {2} →
  A ∪ B = {1, 2, 5} := by
sorry

end set_union_problem_l1930_193041


namespace complex_division_simplification_l1930_193093

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  (8 - i) / (2 + i) = 3 - 2 * i := by
  sorry

end complex_division_simplification_l1930_193093


namespace vectors_collinear_l1930_193035

def a : Fin 3 → ℝ := ![3, -1, 6]
def b : Fin 3 → ℝ := ![5, 7, 10]
def c₁ : Fin 3 → ℝ := λ i => 4 * a i - 2 * b i
def c₂ : Fin 3 → ℝ := λ i => b i - 2 * a i

theorem vectors_collinear : ∃ (k : ℝ), k ≠ 0 ∧ (∀ i : Fin 3, c₁ i = k * c₂ i) := by
  sorry

end vectors_collinear_l1930_193035


namespace toy_cost_l1930_193055

theorem toy_cost (price_A price_B price_C : ℝ) 
  (h1 : 2 * price_A + price_B + 3 * price_C = 24)
  (h2 : 3 * price_A + 4 * price_B + 2 * price_C = 36) :
  price_A + price_B + price_C = 12 := by
  sorry

end toy_cost_l1930_193055


namespace absolute_value_inequality_l1930_193009

theorem absolute_value_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
by sorry

end absolute_value_inequality_l1930_193009


namespace sum_m_n_equals_five_l1930_193057

theorem sum_m_n_equals_five (m n : ℚ) (h : (m - 3) * Real.sqrt 5 + 2 - n = 0) : m + n = 5 := by
  sorry

end sum_m_n_equals_five_l1930_193057


namespace no_primes_between_factorial_plus_n_and_factorial_plus_2n_l1930_193066

theorem no_primes_between_factorial_plus_n_and_factorial_plus_2n (n : ℕ) (hn : n > 1) :
  ∀ p, Nat.Prime p → ¬(n! + n < p ∧ p < n! + 2*n) :=
sorry

end no_primes_between_factorial_plus_n_and_factorial_plus_2n_l1930_193066


namespace task_completion_time_l1930_193010

/-- Given workers A, B, and C with work rates a, b, and c respectively,
    if A and B together complete a task in 8 hours,
    A and C together complete it in 6 hours,
    and B and C together complete it in 4.8 hours,
    then A, B, and C working together will complete the task in 4 hours. -/
theorem task_completion_time (a b c : ℝ) 
  (hab : 8 * (a + b) = 1)
  (hac : 6 * (a + c) = 1)
  (hbc : 4.8 * (b + c) = 1) :
  (a + b + c)⁻¹ = 4 := by sorry

end task_completion_time_l1930_193010


namespace crazy_silly_school_series_l1930_193070

theorem crazy_silly_school_series (total_books : ℕ) (books_read : ℕ) (movies_watched : ℕ) :
  total_books = 11 →
  books_read = 7 →
  movies_watched = 21 →
  movies_watched = books_read + 14 →
  ∃ (total_movies : ℕ), total_movies = 7 ∧ total_movies = movies_watched - 14 :=
by
  sorry

end crazy_silly_school_series_l1930_193070


namespace a_plus_b_value_l1930_193084

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (A ∪ B a b = Set.univ) →
  (A ∩ B a b = Set.Ioc 3 4) →
  a + b = -7 := by
  sorry


end a_plus_b_value_l1930_193084


namespace diophantine_equation_solution_l1930_193017

theorem diophantine_equation_solution : ∃ (u v : ℤ), 364 * u + 154 * v = 14 ∧ u = 3 ∧ v = -7 := by
  sorry

end diophantine_equation_solution_l1930_193017


namespace ab_ab2_a_inequality_l1930_193033

theorem ab_ab2_a_inequality (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end ab_ab2_a_inequality_l1930_193033


namespace extremum_implies_deriv_root_exists_deriv_root_without_extremum_l1930_193049

-- Define a differentiable function on the real line
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for a function to have an extremum
def has_extremum (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

-- Define what it means for f'(x) = 0 to have a real root
def deriv_has_root (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, deriv f x = 0

-- Statement 1: If f has an extremum, then f'(x) = 0 has a real root
theorem extremum_implies_deriv_root :
  has_extremum f → deriv_has_root f :=
sorry

-- Statement 2: There exists a function f such that f'(x) = 0 has a real root,
-- but f does not have an extremum
theorem exists_deriv_root_without_extremum :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ deriv_has_root f ∧ ¬has_extremum f :=
sorry

end extremum_implies_deriv_root_exists_deriv_root_without_extremum_l1930_193049


namespace cuboid_surface_area_l1930_193063

/-- The total surface area of a cuboid with dimensions in the ratio 6:5:4 and actual dimensions 90 cm, 75 cm, and 60 cm is 33300 cm². -/
theorem cuboid_surface_area : 
  let length : ℝ := 90
  let breadth : ℝ := 75
  let height : ℝ := 60
  let ratio_length : ℝ := 6
  let ratio_breadth : ℝ := 5
  let ratio_height : ℝ := 4
  -- Ensure the dimensions are in the correct ratio
  length / ratio_length = breadth / ratio_breadth ∧ 
  breadth / ratio_breadth = height / ratio_height →
  -- Calculate the total surface area
  2 * (length * breadth + breadth * height + height * length) = 33300 := by
  sorry

end cuboid_surface_area_l1930_193063


namespace james_toy_cars_l1930_193060

/-- Proves that James buys 20 toy cars given the problem conditions -/
theorem james_toy_cars : 
  ∀ (cars soldiers : ℕ),
  soldiers = 2 * cars →
  cars + soldiers = 60 →
  cars = 20 := by
sorry

end james_toy_cars_l1930_193060


namespace second_number_value_l1930_193097

theorem second_number_value (x y : ℚ) 
  (h1 : (1 : ℚ) / 5 * x = (5 : ℚ) / 8 * y) 
  (h2 : x + 35 = 4 * y) : 
  y = 40 := by
sorry

end second_number_value_l1930_193097


namespace power_of_product_equals_product_of_powers_l1930_193096

theorem power_of_product_equals_product_of_powers (a : ℝ) : (2 * a^3)^3 = 8 * a^9 := by
  sorry

end power_of_product_equals_product_of_powers_l1930_193096


namespace jellybean_box_capacity_l1930_193045

theorem jellybean_box_capacity (bert_capacity : ℕ) (scale_factor : ℕ) : 
  bert_capacity = 150 → 
  scale_factor = 3 → 
  (scale_factor ^ 3 : ℕ) * bert_capacity = 4050 := by
sorry

end jellybean_box_capacity_l1930_193045


namespace valid_outfit_choices_l1930_193011

/-- Represents the number of types of each clothing item -/
def num_types : ℕ := 8

/-- Represents the number of colors available -/
def num_colors : ℕ := 8

/-- Calculates the total number of outfit combinations -/
def total_combinations : ℕ := num_types^4

/-- Calculates the number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Theorem: The number of valid outfit choices is 4088 -/
theorem valid_outfit_choices : 
  total_combinations - same_color_outfits = 4088 := by sorry

end valid_outfit_choices_l1930_193011


namespace unique_solution_l1930_193068

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_solution (a x y p : ℕ) : Prop :=
  is_single_digit a ∧ is_single_digit x ∧ is_single_digit y ∧ is_single_digit p ∧
  a ≠ x ∧ a ≠ y ∧ a ≠ p ∧ x ≠ y ∧ x ≠ p ∧ y ≠ p ∧
  10 * a + x + 10 * y + x = 100 * y + 10 * p + a

theorem unique_solution :
  ∀ a x y p : ℕ, is_solution a x y p → a = 8 ∧ x = 9 ∧ y = 1 ∧ p = 0 :=
by sorry

end unique_solution_l1930_193068


namespace complex_expression_sum_l1930_193042

theorem complex_expression_sum (z : ℂ) : 
  z = Complex.exp (4 * Real.pi * I / 7) →
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = -2 := by
  sorry

end complex_expression_sum_l1930_193042


namespace exponent_multiplication_l1930_193052

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l1930_193052


namespace factorization_of_difference_of_squares_l1930_193075

theorem factorization_of_difference_of_squares (a : ℝ) : 1 - a^2 = (1 + a) * (1 - a) := by
  sorry

end factorization_of_difference_of_squares_l1930_193075


namespace complex_equation_solution_l1930_193071

theorem complex_equation_solution : 
  ∃ (a b c d e : ℤ),
    (2 * (2 : ℝ)^(2/3) + (2 : ℝ)^(1/3) * a + 2 * b + (2 : ℝ)^(2/3) * c + (2 : ℝ)^(1/3) * d + e = 0) ∧
    (25 * (Complex.I * Real.sqrt 5) + 25 * a - 5 * (Complex.I * Real.sqrt 5) * b - 5 * c + (Complex.I * Real.sqrt 5) * d + e = 0) ∧
    (abs (a + b + c + d + e) = 7) :=
by sorry

end complex_equation_solution_l1930_193071


namespace greatest_sum_of_two_integers_l1930_193072

theorem greatest_sum_of_two_integers (n : ℤ) : 
  (∀ m : ℤ, m * (m + 2) < 500 → m ≤ n) →
  n * (n + 2) < 500 →
  n + (n + 2) = 44 := by
sorry

end greatest_sum_of_two_integers_l1930_193072


namespace at_most_two_special_numbers_l1930_193058

/-- A positive integer n is special if it can be expressed as 2^a * 3^b for some nonnegative integers a and b. -/
def is_special (n : ℕ+) : Prop :=
  ∃ a b : ℕ, n = 2^a * 3^b

/-- For any positive integer k, there are at most two special numbers in the range (k^2, k^2 + 2k + 1). -/
theorem at_most_two_special_numbers (k : ℕ+) :
  ∃ n₁ n₂ : ℕ+, ∀ n : ℕ+,
    k^2 < n ∧ n < k^2 + 2*k + 1 ∧ is_special n →
    n = n₁ ∨ n = n₂ :=
  sorry

end at_most_two_special_numbers_l1930_193058


namespace distance_traveled_l1930_193034

-- Define the speed in miles per hour
def speed : ℝ := 16

-- Define the time in hours
def time : ℝ := 5

-- Theorem to prove the distance traveled
theorem distance_traveled : speed * time = 80 := by
  sorry

end distance_traveled_l1930_193034


namespace max_value_2x_minus_y_l1930_193065

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0)
  (h2 : y + 1 ≥ 0)
  (h3 : x + y + 1 ≤ 0) :
  ∃ (max : ℝ), max = 1 ∧ ∀ x' y' : ℝ, 
    x' - y' + 1 ≥ 0 → y' + 1 ≥ 0 → x' + y' + 1 ≤ 0 → 
    2 * x' - y' ≤ max :=
by sorry

end max_value_2x_minus_y_l1930_193065


namespace hexagram_arrangement_exists_and_unique_l1930_193012

def Hexagram := Fin 7 → Fin 7

def is_valid_arrangement (h : Hexagram) : Prop :=
  (∀ i : Fin 7, ∃! j : Fin 7, h j = i) ∧
  (h 0 + h 1 + h 3 = 12) ∧
  (h 0 + h 2 + h 4 = 12) ∧
  (h 1 + h 2 + h 5 = 12) ∧
  (h 3 + h 4 + h 5 = 12) ∧
  (h 0 + h 6 + h 5 = 12) ∧
  (h 1 + h 6 + h 4 = 12) ∧
  (h 2 + h 6 + h 3 = 12)

theorem hexagram_arrangement_exists_and_unique :
  ∃! h : Hexagram, is_valid_arrangement h :=
sorry

end hexagram_arrangement_exists_and_unique_l1930_193012


namespace intersection_and_complement_when_m_is_3_intersection_equals_B_iff_m_in_range_l1930_193078

-- Define sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

-- Theorem 1
theorem intersection_and_complement_when_m_is_3 :
  (A ∩ B 3 = {x | 3 ≤ x ∧ x ≤ 4}) ∧
  (A ∩ (Set.univ \ B 3) = {x | 1 ≤ x ∧ x < 3}) := by sorry

-- Theorem 2
theorem intersection_equals_B_iff_m_in_range :
  ∀ m : ℝ, A ∩ B m = B m ↔ 1 ≤ m ∧ m ≤ 3 := by sorry

end intersection_and_complement_when_m_is_3_intersection_equals_B_iff_m_in_range_l1930_193078


namespace tan_eleven_pi_fourths_l1930_193087

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by sorry

end tan_eleven_pi_fourths_l1930_193087


namespace stratified_sample_size_l1930_193086

/-- Represents the ratio of students in each grade -/
structure GradeRatio where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the sample size and number of third grade students in the sample -/
structure Sample where
  size : ℕ
  thirdGrade : ℕ

/-- Theorem stating the sample size given the conditions -/
theorem stratified_sample_size 
  (ratio : GradeRatio) 
  (sample : Sample) 
  (h1 : ratio.first = 4)
  (h2 : ratio.second = 3)
  (h3 : ratio.third = 2)
  (h4 : sample.thirdGrade = 10) :
  (ratio.third : ℚ) / (ratio.first + ratio.second + ratio.third : ℚ) = 
  (sample.thirdGrade : ℚ) / (sample.size : ℚ) → 
  sample.size = 45 := by
sorry

end stratified_sample_size_l1930_193086


namespace prism_volume_l1930_193004

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 56) (h2 : b * c = 63) (h3 : 2 * a * c = 72) :
  a * b * c = 504 := by
  sorry

end prism_volume_l1930_193004


namespace cylinder_cone_sphere_volumes_l1930_193039

/-- Given a cylinder with volume 72π, prove the volumes of a cone and sphere with related dimensions. -/
theorem cylinder_cone_sphere_volumes (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  π * r^2 * h = 72 * π → 
  (1/3 : ℝ) * π * r^2 * h = 24 * π ∧ 
  (4/3 : ℝ) * π * (h/2)^3 = 12 * r * π := by
  sorry

end cylinder_cone_sphere_volumes_l1930_193039


namespace solve_for_k_l1930_193082

theorem solve_for_k : ∀ k : ℤ, 
  (∀ x : ℤ, 2*x - 3 = 3*x - 2 + k ↔ x = 2) → 
  k = -3 := by
  sorry

end solve_for_k_l1930_193082


namespace sufficient_not_necessary_condition_a_equals_two_sufficient_a_equals_two_not_necessary_a_equals_two_sufficient_not_necessary_l1930_193002

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + a = 0) ↔ a ≤ 9/4 :=
by sorry

theorem a_equals_two_sufficient (x : ℝ) :
  x^2 - 3*x + 2 = 0 → ∃ y : ℝ, y^2 - 3*y + 2 = 0 :=
by sorry

theorem a_equals_two_not_necessary :
  ∃ a : ℝ, a ≠ 2 ∧ (∃ x : ℝ, x^2 - 3*x + a = 0) :=
by sorry

theorem a_equals_two_sufficient_not_necessary :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → ∃ y : ℝ, y^2 - 3*y + 2 = 0) ∧
  (∃ a : ℝ, a ≠ 2 ∧ (∃ x : ℝ, x^2 - 3*x + a = 0)) :=
by sorry

end sufficient_not_necessary_condition_a_equals_two_sufficient_a_equals_two_not_necessary_a_equals_two_sufficient_not_necessary_l1930_193002


namespace simplify_expression_l1930_193030

theorem simplify_expression : 18 * (14 / 15) * (1 / 12) - (1 / 5) = 1 / 2 := by
  sorry

end simplify_expression_l1930_193030


namespace aku_birthday_cookies_l1930_193003

/-- Given the number of friends, packages, and cookies per package, 
    calculate the number of cookies each child will eat. -/
def cookies_per_child (friends : ℕ) (packages : ℕ) (cookies_per_package : ℕ) : ℕ :=
  (packages * cookies_per_package) / (friends + 1)

/-- Theorem stating that under the given conditions, each child will eat 15 cookies. -/
theorem aku_birthday_cookies : 
  cookies_per_child 4 3 25 = 15 := by
  sorry

end aku_birthday_cookies_l1930_193003


namespace treasure_burial_year_l1930_193091

def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8^i)) 0

theorem treasure_burial_year : 
  octal_to_decimal [1, 7, 6, 2] = 1465 := by
  sorry

end treasure_burial_year_l1930_193091


namespace binomial_2057_1_l1930_193099

theorem binomial_2057_1 : Nat.choose 2057 1 = 2057 := by
  sorry

end binomial_2057_1_l1930_193099


namespace closest_integer_to_cube_root_of_250_l1930_193079

theorem closest_integer_to_cube_root_of_250 : 
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |m ^ 3 - 250| ≥ |n ^ 3 - 250| :=
sorry

end closest_integer_to_cube_root_of_250_l1930_193079


namespace zoo_sandwiches_l1930_193098

theorem zoo_sandwiches (people : ℝ) (sandwiches_per_person : ℝ) :
  people = 219.0 →
  sandwiches_per_person = 3.0 →
  people * sandwiches_per_person = 657.0 := by
  sorry

end zoo_sandwiches_l1930_193098


namespace initial_alcohol_percentage_l1930_193050

/-- Proves that given a 6-liter solution with an unknown initial alcohol percentage,
    adding 1.8 liters of pure alcohol to create a 50% alcohol solution
    implies that the initial alcohol percentage was 35%. -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 6)
  (h2 : added_alcohol = 1.8)
  (h3 : final_percentage = 50)
  (h4 : final_percentage / 100 * (initial_volume + added_alcohol) = 
        (initial_volume * x / 100) + added_alcohol) :
  x = 35 :=
by sorry


end initial_alcohol_percentage_l1930_193050


namespace circle_area_when_radius_equals_six_times_reciprocal_of_circumference_l1930_193020

theorem circle_area_when_radius_equals_six_times_reciprocal_of_circumference :
  ∀ (r : ℝ), r > 0 → (6 * (1 / (2 * Real.pi * r)) = r) → (Real.pi * r^2 = 3) :=
by
  sorry

end circle_area_when_radius_equals_six_times_reciprocal_of_circumference_l1930_193020


namespace N_divisible_by_1980_l1930_193029

/-- The number formed by concatenating all two-digit numbers from 19 to 80 inclusive -/
def N : ℕ := sorry

/-- N is divisible by 1980 -/
theorem N_divisible_by_1980 : 1980 ∣ N := by sorry

end N_divisible_by_1980_l1930_193029


namespace paper_tray_height_l1930_193026

theorem paper_tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 120 →
  cut_distance = Real.sqrt 20 →
  cut_angle = π / 4 →
  ∃ (height : ℝ), height = (800 : ℝ) ^ (1/4 : ℝ) :=
by sorry

end paper_tray_height_l1930_193026


namespace isosceles_triangle_solution_l1930_193007

/-- Represents a system of linear equations in two variables -/
structure LinearSystem where
  eq1 : ℝ → ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ → ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  leg : ℝ
  base : ℝ

/-- The main theorem -/
theorem isosceles_triangle_solution (a : ℝ) : 
  let system : LinearSystem := {
    eq1 := fun x y a => 3 * x - y - (2 * a - 5)
    eq2 := fun x y a => x + 2 * y - (3 * a + 3)
  }
  let x := a - 1
  let y := a + 2
  (x > 0 ∧ y > 0) →
  (∃ t : IsoscelesTriangle, t.leg = x ∧ t.base = y ∧ 2 * t.leg + t.base = 12) →
  system.eq1 x y a = 0 ∧ system.eq2 x y a = 0 →
  a = 3 := by
  sorry

end isosceles_triangle_solution_l1930_193007


namespace range_of_a_l1930_193027

/-- The range of a given the conditions in the problem -/
theorem range_of_a (x a : ℝ) : 
  (∀ x, (1/2 ≤ x ∧ x ≤ 1) ↔ ¬((x < 1/2) ∨ (1 < x))) →
  (∀ x, ((x - a) * (x - a - 1) ≤ 0) ↔ (a ≤ x ∧ x ≤ a + 1)) →
  (∀ x, ¬((1/2 ≤ x ∧ x ≤ 1)) → ¬((x - a) * (x - a - 1) ≤ 0)) →
  (∃ x, ¬((1/2 ≤ x ∧ x ≤ 1)) ∧ ((x - a) * (x - a - 1) ≤ 0)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l1930_193027


namespace fraction_modification_result_l1930_193085

theorem fraction_modification_result (a b : ℤ) (h1 : a.gcd b = 1) 
  (h2 : (a - 1) / (b - 2) = (a + 1) / b) : (a - 1) / (b - 2) = 1 := by
  sorry

end fraction_modification_result_l1930_193085


namespace solve_exponential_equation_l1930_193074

theorem solve_exponential_equation (y : ℝ) : (1000 : ℝ)^4 = 10^y → y = 12 := by
  sorry

end solve_exponential_equation_l1930_193074


namespace housing_boom_calculation_l1930_193000

/-- The number of houses in Lawrence County before the housing boom. -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom. -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom. -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_calculation :
  houses_built = 574 :=
by sorry

end housing_boom_calculation_l1930_193000


namespace max_integer_values_quadratic_l1930_193014

/-- Given a quadratic function f(x) = ax² + bx + c where a > 100,
    the maximum number of integer x values satisfying |f(x)| ≤ 50 is 2 -/
theorem max_integer_values_quadratic (a b c : ℝ) (ha : a > 100) :
  (∃ (n : ℕ), ∀ (S : Finset ℤ),
    (∀ x ∈ S, |a * x^2 + b * x + c| ≤ 50) →
    S.card ≤ n) ∧
  (∃ (S : Finset ℤ), (∀ x ∈ S, |a * x^2 + b * x + c| ≤ 50) ∧ S.card = 2) :=
sorry

end max_integer_values_quadratic_l1930_193014


namespace better_fit_larger_R_squared_l1930_193064

-- Define the correlation index R²
def correlation_index (R : ℝ) : Prop := 0 ≤ R ∧ R ≤ 1

-- Define the concept of model fit
def model_fit (fit : ℝ) : Prop := 0 ≤ fit

-- Theorem stating that a larger R² indicates a better model fit
theorem better_fit_larger_R_squared 
  (R1 R2 fit1 fit2 : ℝ) 
  (h1 : correlation_index R1) 
  (h2 : correlation_index R2) 
  (h3 : model_fit fit1) 
  (h4 : model_fit fit2) 
  (h5 : R1 < R2) : 
  fit1 < fit2 := by
sorry


end better_fit_larger_R_squared_l1930_193064


namespace dress_design_count_l1930_193006

/-- The number of fabric colors available -/
def num_colors : Nat := 5

/-- The number of patterns available -/
def num_patterns : Nat := 5

/-- The number of fabric materials available -/
def num_materials : Nat := 2

/-- A dress design consists of exactly one color, one pattern, and one material -/
structure DressDesign where
  color : Fin num_colors
  pattern : Fin num_patterns
  material : Fin num_materials

/-- The total number of possible dress designs -/
def total_designs : Nat := num_colors * num_patterns * num_materials

theorem dress_design_count : total_designs = 50 := by
  sorry

end dress_design_count_l1930_193006


namespace pizza_cost_per_piece_l1930_193016

/-- 
Given that Luigi bought 4 pizzas for $80 and each pizza was cut into 5 pieces,
prove that each piece of pizza costs $4.
-/
theorem pizza_cost_per_piece 
  (num_pizzas : ℕ) 
  (total_cost : ℚ) 
  (pieces_per_pizza : ℕ) 
  (h1 : num_pizzas = 4) 
  (h2 : total_cost = 80) 
  (h3 : pieces_per_pizza = 5) : 
  total_cost / (num_pizzas * pieces_per_pizza : ℚ) = 4 := by
sorry

end pizza_cost_per_piece_l1930_193016


namespace book_cost_price_l1930_193037

/-- The cost price of a book satisfying given profit conditions -/
theorem book_cost_price : ∃ (C : ℝ), 
  (C > 0) ∧ 
  (1.15 * C = 1.10 * C + 100) ∧ 
  (C = 2000) := by
  sorry

end book_cost_price_l1930_193037


namespace simplify_expression_1_simplify_and_evaluate_expression_2_evaluate_expression_2_l1930_193005

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b := by sorry

-- Problem 2
theorem simplify_and_evaluate_expression_2 (x y : ℝ) :
  3 * x^2 * y - (2 * x * y^2 - 2 * (x * y - 1.5 * x^2 * y) + x * y) + 3 * x * y^2 = x * y^2 + x * y := by sorry

theorem evaluate_expression_2 :
  let x : ℝ := -3
  let y : ℝ := -2
  3 * x^2 * y - (2 * x * y^2 - 2 * (x * y - 1.5 * x^2 * y) + x * y) + 3 * x * y^2 = -6 := by sorry

end simplify_expression_1_simplify_and_evaluate_expression_2_evaluate_expression_2_l1930_193005


namespace triangle_area_l1930_193092

theorem triangle_area (x : ℝ) (α : ℝ) : 
  let BC := 4*x
  let CD := x
  let AC := 8*x*(Real.sqrt 2/Real.sqrt 3)
  let AD := (3/4 : ℝ)
  let cos_α := Real.sqrt 2/Real.sqrt 3
  let sin_α := 1/Real.sqrt 3
  (AD^2 = 33*x^2) →
  (1/2 * AC * BC * sin_α = Real.sqrt 2/11) :=
by sorry

end triangle_area_l1930_193092


namespace correct_calculation_l1930_193083

theorem correct_calculation (x : ℕ) (h : x - 6 = 51) : x * 6 = 342 := by
  sorry

#check correct_calculation

end correct_calculation_l1930_193083


namespace distance_to_place_l1930_193059

/-- The distance to the place -/
def distance : ℝ := 48

/-- The rowing speed in still water (km/h) -/
def rowing_speed : ℝ := 10

/-- The current velocity (km/h) -/
def current_velocity : ℝ := 2

/-- The wind speed (km/h) -/
def wind_speed : ℝ := 4

/-- The total time for the round trip (hours) -/
def total_time : ℝ := 15

/-- The effective speed towards the place (km/h) -/
def speed_to_place : ℝ := rowing_speed - wind_speed - current_velocity

/-- The effective speed returning from the place (km/h) -/
def speed_from_place : ℝ := rowing_speed + wind_speed + current_velocity

theorem distance_to_place : 
  distance = (total_time * speed_to_place * speed_from_place) / (speed_to_place + speed_from_place) :=
by sorry

end distance_to_place_l1930_193059


namespace absolute_value_equals_sqrt_of_square_l1930_193001

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end absolute_value_equals_sqrt_of_square_l1930_193001


namespace sqrt_3_times_sqrt_12_l1930_193023

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l1930_193023


namespace weight_of_replaced_person_l1930_193076

theorem weight_of_replaced_person
  (n : ℕ)
  (average_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : n = 10)
  (h2 : average_increase = 2.5)
  (h3 : new_person_weight = 90)
  : ∃ (replaced_weight : ℝ),
    replaced_weight = new_person_weight - n * average_increase :=
by
  sorry

end weight_of_replaced_person_l1930_193076


namespace carmina_coins_count_l1930_193048

theorem carmina_coins_count :
  ∀ (n d : ℕ),
  (5 * n + 10 * d = 360) →
  (10 * n + 5 * d = 540) →
  n + d = 60 :=
by
  sorry

end carmina_coins_count_l1930_193048


namespace arithmetic_sequence_tenth_term_l1930_193054

/-- An arithmetic sequence is a sequence where the difference between 
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℚ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third_term : a 3 = 5)
  (h_seventh_term : a 7 = 13) :
  a 10 = 19 := by
sorry

end arithmetic_sequence_tenth_term_l1930_193054


namespace robins_count_l1930_193090

theorem robins_count (total : ℕ) (robins penguins pigeons : ℕ) : 
  robins = 2 * total / 3 →
  penguins = total / 8 →
  pigeons = 5 →
  total = robins + penguins + pigeons →
  robins = 16 := by
sorry

end robins_count_l1930_193090


namespace lateral_edges_coplanar_iff_height_eq_edge_l1930_193032

/-- A cube with regular 4-sided pyramids on each face -/
structure PyramidCube where
  -- Edge length of the cube
  a : ℝ
  -- Height of the pyramids
  h : ℝ
  -- Assumption that a and h are positive
  a_pos : 0 < a
  h_pos : 0 < h

/-- The condition for lateral edges to lie in the same plane -/
def lateral_edges_coplanar (cube : PyramidCube) : Prop :=
  cube.h = cube.a

/-- Theorem stating the condition for lateral edges to be coplanar -/
theorem lateral_edges_coplanar_iff_height_eq_edge (cube : PyramidCube) :
  lateral_edges_coplanar cube ↔ cube.h = cube.a :=
sorry


end lateral_edges_coplanar_iff_height_eq_edge_l1930_193032


namespace equal_selection_probability_l1930_193047

/-- Represents the selection process for a student survey -/
structure StudentSurvey where
  total_students : ℕ
  selected_students : ℕ
  eliminated_students : ℕ
  remaining_students : ℕ

/-- The probability of a student being selected in the survey -/
def selection_probability (survey : StudentSurvey) : ℚ :=
  (survey.remaining_students : ℚ) / (survey.total_students : ℚ) *
  (survey.selected_students : ℚ) / (survey.remaining_students : ℚ)

/-- The specific survey described in the problem -/
def school_survey : StudentSurvey :=
  { total_students := 2012
  , selected_students := 50
  , eliminated_students := 12
  , remaining_students := 2000 }

/-- Theorem stating that the selection probability is equal for all students -/
theorem equal_selection_probability :
  ∀ (s1 s2 : StudentSurvey),
    s1 = school_survey → s2 = school_survey →
    selection_probability s1 = selection_probability s2 :=
by sorry

end equal_selection_probability_l1930_193047


namespace second_number_value_l1930_193021

theorem second_number_value (x y z : ℚ) : 
  x + y + z = 120 ∧ 
  x / y = 3 / 4 ∧ 
  y / z = 4 / 7 →
  y = 240 / 7 := by
sorry

end second_number_value_l1930_193021


namespace train_theorem_l1930_193044

def train_problem (initial : ℕ) 
  (stop1_off stop1_on : ℕ)
  (stop2_off stop2_on stop2_first_off : ℕ)
  (stop3_off stop3_on stop3_first_off : ℕ)
  (stop4_off stop4_on stop4_second_off : ℕ)
  (stop5_off stop5_on : ℕ) : Prop :=
  let after_stop1 := initial - stop1_off + stop1_on
  let after_stop2 := after_stop1 - stop2_off + stop2_on - stop2_first_off
  let after_stop3 := after_stop2 - stop3_off + stop3_on - stop3_first_off
  let after_stop4 := after_stop3 - stop4_off + stop4_on - stop4_second_off
  let final := after_stop4 - stop5_off + stop5_on
  final = 26

theorem train_theorem : train_problem 48 13 5 9 10 2 7 4 3 16 7 5 8 15 := by
  sorry

end train_theorem_l1930_193044


namespace max_mondays_in_45_days_l1930_193080

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days we're considering -/
def days_considered : ℕ := 45

/-- The maximum number of Mondays in the first 45 days of a year -/
def max_mondays : ℕ := 7

/-- Theorem: The maximum number of Mondays in the first 45 days of a year is 7 -/
theorem max_mondays_in_45_days : 
  ∀ (start_day : ℕ), start_day < days_in_week →
  (∃ (monday_count : ℕ), 
    monday_count ≤ max_mondays ∧
    monday_count = (days_considered / days_in_week) + 
      (if start_day = 0 then 1 else 0)) :=
sorry

end max_mondays_in_45_days_l1930_193080


namespace lcm_gcf_relations_l1930_193043

theorem lcm_gcf_relations :
  (∃! n : ℕ, Nat.lcm n 16 = 52 ∧ Nat.gcd n 16 = 8) ∧
  (¬ ∃ n : ℕ, Nat.lcm n 20 = 84 ∧ Nat.gcd n 20 = 4) ∧
  (∃! n : ℕ, Nat.lcm n 24 = 120 ∧ Nat.gcd n 24 = 6) := by
  sorry

end lcm_gcf_relations_l1930_193043


namespace video_game_lives_l1930_193028

theorem video_game_lives (initial_lives hard_part_lives next_level_lives : ℝ) :
  initial_lives + hard_part_lives + next_level_lives =
  initial_lives + (hard_part_lives + next_level_lives) :=
by
  sorry

-- Example usage
def tiffany_game (initial_lives hard_part_lives next_level_lives : ℝ) : ℝ :=
  initial_lives + hard_part_lives + next_level_lives

#eval tiffany_game 43.0 14.0 27.0

end video_game_lives_l1930_193028


namespace trapezoid_shorter_base_length_l1930_193061

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_line : ℝ

/-- The property that the line joining the midpoints of the diagonals is half the difference of the bases -/
def midpoint_line_property (t : Trapezoid) : Prop :=
  t.midpoint_line = (t.long_base - t.short_base) / 2

/-- Theorem: In a trapezoid where the line joining the midpoints of the diagonals has length 4
    and the longer base is 100, the shorter base has length 92 -/
theorem trapezoid_shorter_base_length :
  ∀ t : Trapezoid,
    t.long_base = 100 →
    t.midpoint_line = 4 →
    midpoint_line_property t →
    t.short_base = 92 := by
  sorry


end trapezoid_shorter_base_length_l1930_193061


namespace inscribed_sphere_radius_is_17_l1930_193089

structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

structure DistancesToFaces where
  ABC : ℝ
  ABD : ℝ
  ACD : ℝ
  BCD : ℝ

def ABCD : Tetrahedron := sorry

def X : Point := sorry
def Y : Point := sorry

def distances_X : DistancesToFaces := {
  ABC := 14,
  ABD := 11,
  ACD := 29,
  BCD := 8
}

def distances_Y : DistancesToFaces := {
  ABC := 15,
  ABD := 13,
  ACD := 25,
  BCD := 11
}

def inscribed_sphere_radius (t : Tetrahedron) : ℝ := sorry

theorem inscribed_sphere_radius_is_17 :
  inscribed_sphere_radius ABCD = 17 := by sorry

end inscribed_sphere_radius_is_17_l1930_193089


namespace range_of_f_l1930_193053

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | 0 ≤ y ∧ y ≤ 4 } := by sorry

end range_of_f_l1930_193053


namespace arithmetic_series_sum_l1930_193022

theorem arithmetic_series_sum (k : ℕ) : 
  let a₁ : ℕ := k^2 + k + 1
  let d : ℕ := 1
  let n : ℕ := 2*k + 3
  let S := n * (2*a₁ + (n-1)*d) / 2
  S = 2*k^3 + 7*k^2 + 10*k + 6 := by
sorry

end arithmetic_series_sum_l1930_193022


namespace cylinder_surface_area_l1930_193073

/-- The total surface area of a cylinder with height 8 and radius 5 is 130π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 8
  let r : ℝ := 5
  let circle_area := π * r^2
  let lateral_area := 2 * π * r * h
  circle_area * 2 + lateral_area = 130 * π :=
by sorry

end cylinder_surface_area_l1930_193073


namespace average_of_data_l1930_193013

def data : List ℝ := [4, 6, 5, 8, 7, 6]

theorem average_of_data :
  (data.sum / data.length : ℝ) = 6 := by
  sorry

end average_of_data_l1930_193013


namespace water_formed_l1930_193036

-- Define the substances involved in the reaction
structure Substance where
  name : String
  moles : ℝ
  molar_mass : ℝ

-- Define the reaction
def reaction (naoh : Substance) (hcl : Substance) (water : Substance) : Prop :=
  naoh.name = "Sodium hydroxide" ∧
  hcl.name = "Hydrochloric acid" ∧
  water.name = "Water" ∧
  naoh.moles = 1 ∧
  water.molar_mass = 18 ∧
  water.moles * water.molar_mass = 18

-- Theorem statement
theorem water_formed (naoh hcl water : Substance) :
  reaction naoh hcl water → water.moles = 1 := by
  sorry

end water_formed_l1930_193036


namespace fencing_required_l1930_193095

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 680 ∧ uncovered_side = 20 → 
  2 * (area / uncovered_side) + uncovered_side = 88 := by
  sorry

#check fencing_required

end fencing_required_l1930_193095


namespace two_numbers_difference_l1930_193015

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (square_diff : x^2 - y^2 = 48) : 
  |x - y| = 6 := by
sorry

end two_numbers_difference_l1930_193015


namespace seventh_diagram_shaded_fraction_l1930_193088

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Total number of triangles in the nth diagram -/
def total_triangles (n : ℕ) : ℕ := Nat.factorial n

/-- Fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ :=
  (fib n : ℚ) / (total_triangles n : ℚ)

/-- The main theorem -/
theorem seventh_diagram_shaded_fraction :
  shaded_fraction 7 = 13 / 5040 := by
  sorry

end seventh_diagram_shaded_fraction_l1930_193088


namespace number_of_factors_60_l1930_193025

/-- The number of positive factors of 60 is 12. -/
theorem number_of_factors_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end number_of_factors_60_l1930_193025


namespace largest_multiple_of_12_negation_greater_than_neg_150_l1930_193018

theorem largest_multiple_of_12_negation_greater_than_neg_150 :
  ∀ n : ℤ, 12 ∣ n ∧ -n > -150 → n ≤ 144 :=
by
  sorry

end largest_multiple_of_12_negation_greater_than_neg_150_l1930_193018


namespace jackie_cosmetics_purchase_l1930_193040

/-- The cost of a bottle of lotion -/
def lotion_cost : ℚ := 6

/-- The number of bottles of lotion purchased -/
def lotion_quantity : ℕ := 3

/-- The amount needed to reach the free shipping threshold -/
def additional_amount : ℚ := 12

/-- The free shipping threshold -/
def free_shipping_threshold : ℚ := 50

/-- The cost of a bottle of shampoo or conditioner -/
def shampoo_conditioner_cost : ℚ := 10

theorem jackie_cosmetics_purchase :
  2 * shampoo_conditioner_cost + lotion_cost * lotion_quantity + additional_amount = free_shipping_threshold := by
  sorry

end jackie_cosmetics_purchase_l1930_193040


namespace knicks_knacks_knocks_conversion_l1930_193008

/-- Given the conversion rates between knicks, knacks, and knocks, 
    this theorem proves that 36 knocks are equivalent to 40 knicks. -/
theorem knicks_knacks_knocks_conversion : 
  ∀ (knick knack knock : ℚ),
  (5 * knick = 3 * knack) →
  (4 * knack = 6 * knock) →
  (36 * knock = 40 * knick) := by
sorry

end knicks_knacks_knocks_conversion_l1930_193008


namespace last_digits_of_11_power_l1930_193031

theorem last_digits_of_11_power (n : ℕ) (h : n ≥ 1) :
  11^(10^n) ≡ 6 * 10^(n+1) + 1 [MOD 10^(n+2)] := by
sorry

end last_digits_of_11_power_l1930_193031


namespace sally_saturday_sandwiches_l1930_193019

/-- The number of sandwiches Sally eats on Saturday -/
def sandwiches_saturday : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sandwiches_sunday : ℕ := 1

/-- The number of pieces of bread used in each sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := 6

/-- Theorem stating that Sally eats 2 sandwiches on Saturday -/
theorem sally_saturday_sandwiches :
  sandwiches_saturday = (total_bread - sandwiches_sunday * bread_per_sandwich) / bread_per_sandwich :=
by sorry

end sally_saturday_sandwiches_l1930_193019


namespace largest_x_floor_div_l1930_193046

theorem largest_x_floor_div : 
  ∀ x : ℝ, (↑⌊x⌋ / x = 7 / 8) → x ≤ 48 / 7 :=
by sorry

end largest_x_floor_div_l1930_193046


namespace solution_in_interval_l1930_193024

open Real

theorem solution_in_interval :
  ∃! x₀ : ℝ, 2 < x₀ ∧ x₀ < 3 ∧ Real.log x₀ + x₀ = 4 := by
  sorry

end solution_in_interval_l1930_193024


namespace rainfall_difference_l1930_193081

def camping_days : ℕ := 14
def rainy_days : ℕ := 7
def friend_rainfall : ℕ := 65

def greg_rainfall : List ℕ := [3, 6, 5, 7, 4, 8, 9]

theorem rainfall_difference :
  friend_rainfall - (greg_rainfall.sum) = 23 :=
by sorry

end rainfall_difference_l1930_193081


namespace extreme_points_and_inequality_l1930_193051

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - x

theorem extreme_points_and_inequality (a : ℝ) (h : a > 1/2) :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    (∀ (y : ℝ), |y - x₁| < δ → f a y ≤ f a x₁ + ε) ∧
    (∀ (y : ℝ), |y - x₂| < δ → f a y ≥ f a x₂ - ε)) ∧
  f a x₂ < 1 + (Real.sin x₂ - x₂) / 2 :=
sorry

end extreme_points_and_inequality_l1930_193051


namespace quadratic_inequality_necessary_not_sufficient_l1930_193062

theorem quadratic_inequality_necessary_not_sufficient :
  (∀ x : ℝ, x > 2 → x^2 + 2*x - 8 > 0) ∧
  (∃ x : ℝ, x^2 + 2*x - 8 > 0 ∧ ¬(x > 2)) :=
by sorry

end quadratic_inequality_necessary_not_sufficient_l1930_193062


namespace point_on_line_l1930_193038

/-- Given that the point (x, -3) lies on the straight line joining (2, 10) and (6, 2) in the xy-plane, prove that x = 8.5 -/
theorem point_on_line (x : ℝ) :
  (∃ t : ℝ, t ∈ (Set.Icc 0 1) ∧
    x = 2 * (1 - t) + 6 * t ∧
    -3 = 10 * (1 - t) + 2 * t) →
  x = 8.5 := by
sorry

end point_on_line_l1930_193038


namespace hyperbola_sum_a_h_l1930_193056

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  -- Asymptote equations
  asymptote1 : ℝ → ℝ
  asymptote2 : ℝ → ℝ
  -- Point the hyperbola passes through
  point : ℝ × ℝ
  -- Standard form parameters
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  -- Conditions
  asymptote1_eq : ∀ x, asymptote1 x = 3 * x + 2
  asymptote2_eq : ∀ x, asymptote2 x = -3 * x + 8
  point_on_hyperbola : point = (1, 6)
  standard_form : ∀ x y, (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1
  positive_params : a > 0 ∧ b > 0

/-- Theorem: For the given hyperbola, a + h = 2 -/
theorem hyperbola_sum_a_h (hyp : Hyperbola) : hyp.a + hyp.h = 2 := by
  sorry

end hyperbola_sum_a_h_l1930_193056


namespace prism_volume_l1930_193094

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 18)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ x y z : ℝ, 
    x * y = side_area ∧ 
    y * z = front_area ∧ 
    x * z = bottom_area ∧ 
    x * y * z = 24 * Real.sqrt 3 := by
  sorry

end prism_volume_l1930_193094


namespace smallest_m_for_integral_solutions_l1930_193069

theorem smallest_m_for_integral_solutions : 
  (∀ m : ℕ, m > 0 ∧ m < 160 → ¬∃ x : ℤ, 10 * x^2 - m * x + 630 = 0) ∧ 
  (∃ x : ℤ, 10 * x^2 - 160 * x + 630 = 0) := by
  sorry

end smallest_m_for_integral_solutions_l1930_193069


namespace complex_sum_to_polar_l1930_193067

theorem complex_sum_to_polar : 
  15 * Complex.exp (Complex.I * Real.pi / 7) + 15 * Complex.exp (Complex.I * 5 * Real.pi / 7) = 
  (30 * Real.cos (3 * Real.pi / 14) * Real.cos (Real.pi / 14)) * Complex.exp (Complex.I * 3 * Real.pi / 7) := by
  sorry

end complex_sum_to_polar_l1930_193067


namespace costs_equal_at_60_l1930_193077

/-- Represents the pricing and discount options for appliances -/
structure AppliancePricing where
  washing_machine_price : ℕ
  cooker_price : ℕ
  option1_free_cookers : ℕ
  option2_discount : ℚ

/-- Calculates the cost for Option 1 -/
def option1_cost (p : AppliancePricing) (washing_machines : ℕ) (cookers : ℕ) : ℕ :=
  p.washing_machine_price * washing_machines + p.cooker_price * (cookers - p.option1_free_cookers)

/-- Calculates the cost for Option 2 -/
def option2_cost (p : AppliancePricing) (washing_machines : ℕ) (cookers : ℕ) : ℚ :=
  (p.washing_machine_price * washing_machines + p.cooker_price * cookers : ℚ) * p.option2_discount

/-- Theorem: Costs of Option 1 and Option 2 are equal when x = 60 -/
theorem costs_equal_at_60 (p : AppliancePricing) 
    (h1 : p.washing_machine_price = 800)
    (h2 : p.cooker_price = 200)
    (h3 : p.option1_free_cookers = 10)
    (h4 : p.option2_discount = 9/10) :
    (option1_cost p 10 60 : ℚ) = option2_cost p 10 60 := by
  sorry

end costs_equal_at_60_l1930_193077
