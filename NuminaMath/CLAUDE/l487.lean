import Mathlib

namespace product_remainder_mod_five_l487_48704

theorem product_remainder_mod_five : (1483 * 1773 * 1827 * 2001) % 5 = 3 := by
  sorry

end product_remainder_mod_five_l487_48704


namespace not_prime_sum_product_l487_48709

theorem not_prime_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a) 
  (h_eq : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : 
  ¬ Nat.Prime (a * b + c * d) := by
  sorry

end not_prime_sum_product_l487_48709


namespace expression_evaluation_l487_48723

theorem expression_evaluation : 
  let a : ℚ := -1/2
  (a - 2) * (a + 2) - (a + 1) * (a - 3) = -2 := by
sorry

end expression_evaluation_l487_48723


namespace updated_mean_after_correction_l487_48732

theorem updated_mean_after_correction (n : ℕ) (original_mean : ℝ) (decrement : ℝ) :
  n = 100 →
  original_mean = 350 →
  decrement = 63 →
  (n : ℝ) * original_mean + n * decrement = n * 413 := by
sorry

end updated_mean_after_correction_l487_48732


namespace part_one_part_two_l487_48726

-- Define the linear function f
def f (a b x : ℝ) : ℝ := a * x + b

-- Define the function g
def g (m x : ℝ) : ℝ := (x + m) * (4 * x + 1)

-- Theorem for part (I)
theorem part_one (a b : ℝ) :
  (∀ x y, x < y → f a b x < f a b y) →
  (∀ x, f a b (f a b x) = 16 * x + 5) →
  a = 4 ∧ b = 1 := by sorry

-- Theorem for part (II)
theorem part_two (m : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y → g m x < g m y) →
  m ≥ -9/4 := by sorry

end part_one_part_two_l487_48726


namespace express_y_in_terms_of_x_l487_48717

theorem express_y_in_terms_of_x (p : ℝ) (x y : ℝ) : 
  x = 3 + 2^p → y = 3 + 2^(-p) → y = (3*x - 8) / (x - 3) := by
  sorry

end express_y_in_terms_of_x_l487_48717


namespace f_derivative_l487_48766

-- Define the function f
def f (x : ℝ) : ℝ := (2*x - 1) * (x^2 + 3)

-- State the theorem
theorem f_derivative :
  deriv f = fun x => 6*x^2 - 2*x + 6 :=
by sorry

end f_derivative_l487_48766


namespace equal_intercept_line_equation_l487_48731

/-- A line passing through point A(1,2) with equal intercepts on both coordinate axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through point A(1,2) -/
  passes_through_A : m * 1 + b = 2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b = b / m

/-- The equation of an EqualInterceptLine is either 2x - y = 0 or x + y = 3 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 2 ∧ l.b = 0) ∨ (l.m = 1 ∧ l.b = 2) :=
sorry

end equal_intercept_line_equation_l487_48731


namespace roots_properties_l487_48746

def equation_roots (b : ℝ) (θ : ℝ) : Prop :=
  169 * (Real.sin θ)^2 - b * (Real.sin θ) + 60 = 0 ∧
  169 * (Real.cos θ)^2 - b * (Real.cos θ) + 60 = 0

theorem roots_properties (θ : ℝ) (h : π/4 < θ ∧ θ < 3*π/4) :
  ∃ b : ℝ, equation_roots b θ ∧
    b = 221 ∧
    (Real.sin θ / (1 - Real.cos θ)) + ((1 + Real.cos θ) / Real.sin θ) = 3 :=
by sorry

end roots_properties_l487_48746


namespace deer_weight_l487_48761

/-- Calculates the weight of each deer given hunting frequency, season length, and total kept weight --/
theorem deer_weight 
  (hunts_per_month : ℕ)
  (season_fraction : ℚ)
  (deer_per_hunt : ℕ)
  (kept_fraction : ℚ)
  (total_kept_weight : ℕ)
  (h1 : hunts_per_month = 6)
  (h2 : season_fraction = 1/4)
  (h3 : deer_per_hunt = 2)
  (h4 : kept_fraction = 1/2)
  (h5 : total_kept_weight = 10800) :
  (total_kept_weight / kept_fraction) / (hunts_per_month * (season_fraction * 12) * deer_per_hunt) = 600 := by
  sorry

end deer_weight_l487_48761


namespace infinite_solutions_c_equals_six_l487_48796

theorem infinite_solutions_c_equals_six :
  ∃! c : ℝ, ∀ y : ℝ, 2 * (4 + c * y) = 12 * y + 8 :=
by sorry

end infinite_solutions_c_equals_six_l487_48796


namespace sin_18_cos_12_plus_cos_18_sin_12_l487_48758

theorem sin_18_cos_12_plus_cos_18_sin_12 :
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end sin_18_cos_12_plus_cos_18_sin_12_l487_48758


namespace trail_mix_weight_l487_48739

theorem trail_mix_weight (peanuts chocolate_chips raisins : ℝ) 
  (h1 : peanuts = 0.17)
  (h2 : chocolate_chips = 0.17)
  (h3 : raisins = 0.08) :
  peanuts + chocolate_chips + raisins = 0.42 := by
  sorry

end trail_mix_weight_l487_48739


namespace complement_cardinality_l487_48735

def U : Finset Nat := {1,2,3,4,5,6}
def M : Finset Nat := {2,3,5}
def N : Finset Nat := {4,5}

theorem complement_cardinality : Finset.card (U \ (M ∪ N)) = 2 := by
  sorry

end complement_cardinality_l487_48735


namespace red_subsequence_2019th_element_l487_48741

/-- Represents the number of elements in the nth group of the red-colored subsequence -/
def group_size (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the last element of the nth group in the red-colored subsequence -/
def last_element_of_group (n : ℕ) : ℕ := n * (2 * n - 1)

/-- Represents the sum of elements in the first n groups of the red-colored subsequence -/
def sum_of_elements (n : ℕ) : ℕ := (1 + group_size n) * n / 2

/-- The group number containing the 2019th element -/
def target_group : ℕ := 45

/-- The position of the 2019th element within its group -/
def position_in_group : ℕ := 83

/-- The theorem stating that the 2019th number in the red-colored subsequence is 3993 -/
theorem red_subsequence_2019th_element : 
  last_element_of_group (target_group - 1) + 1 + (position_in_group - 1) * 2 = 3993 :=
sorry

end red_subsequence_2019th_element_l487_48741


namespace count_is_six_l487_48757

/-- A type representing the blocks of digits --/
inductive DigitBlock
  | two
  | fortyfive
  | sixtyeight

/-- The set of all possible permutations of the digit blocks --/
def permutations : List (List DigitBlock) :=
  [DigitBlock.two, DigitBlock.fortyfive, DigitBlock.sixtyeight].permutations

/-- The count of all possible 5-digit numbers formed by the digits 2, 45, 68 --/
def count_five_digit_numbers : Nat := permutations.length

/-- Theorem stating that the count of possible 5-digit numbers is 6 --/
theorem count_is_six : count_five_digit_numbers = 6 := by sorry

end count_is_six_l487_48757


namespace rectangle_area_is_75_l487_48711

/-- Represents a rectangle with length and breadth -/
structure Rectangle where
  length : ℝ
  breadth : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.breadth)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.breadth

/-- Theorem: A rectangle with length thrice its breadth and perimeter 40 has an area of 75 -/
theorem rectangle_area_is_75 (r : Rectangle) 
    (h1 : r.length = 3 * r.breadth) 
    (h2 : perimeter r = 40) : 
  area r = 75 := by
  sorry

end rectangle_area_is_75_l487_48711


namespace star_four_three_l487_48763

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 - a*b + b^2

-- Theorem statement
theorem star_four_three : star 4 3 = 13 := by
  sorry

end star_four_three_l487_48763


namespace total_food_eaten_l487_48772

/-- The amount of food Ella's dog eats relative to Ella -/
def dog_food_ratio : ℕ := 4

/-- The number of days -/
def days : ℕ := 10

/-- The amount of food Ella eats per day (in pounds) -/
def ella_food_per_day : ℕ := 20

/-- The total amount of food eaten by Ella and her dog (in pounds) -/
def total_food : ℕ := days * ella_food_per_day * (1 + dog_food_ratio)

theorem total_food_eaten :
  total_food = 1000 := by sorry

end total_food_eaten_l487_48772


namespace log_domain_condition_l487_48749

theorem log_domain_condition (a : ℝ) :
  (∀ x : ℝ, 0 < x^2 + 2*x + a) ↔ a > 1 := by sorry

end log_domain_condition_l487_48749


namespace min_value_quadratic_l487_48707

theorem min_value_quadratic (x : ℝ) :
  ∃ (z_min : ℝ), ∀ (z : ℝ), z = 3 * x^2 + 18 * x + 11 → z ≥ z_min ∧ ∃ (x_min : ℝ), 3 * x_min^2 + 18 * x_min + 11 = z_min :=
by sorry

end min_value_quadratic_l487_48707


namespace inscribed_circle_height_difference_l487_48712

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x^2 + 4 * x

/-- Represents a circle inscribed in the parabola -/
structure InscribedCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangentPoint : ℝ
  isTangent : (tangentPoint, parabola tangentPoint) ∈ frontier {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

/-- The height difference between the circle's center and a tangent point -/
def heightDifference (circle : InscribedCircle) : ℝ :=
  circle.center.2 - parabola circle.tangentPoint

theorem inscribed_circle_height_difference (circle : InscribedCircle) :
  heightDifference circle = -2 * circle.tangentPoint^2 - 4 * circle.tangentPoint + 2 :=
by sorry

end inscribed_circle_height_difference_l487_48712


namespace complex_product_magnitude_l487_48700

theorem complex_product_magnitude (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x = 6 * Real.sqrt 6 :=
by
  sorry

end complex_product_magnitude_l487_48700


namespace fermat_numbers_coprime_l487_48795

theorem fermat_numbers_coprime (m n : ℕ) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := by sorry

end fermat_numbers_coprime_l487_48795


namespace no_pentagon_cross_section_l487_48734

/-- A cube in 3D space -/
structure Cube

/-- A plane in 3D space -/
structure Plane

/-- Possible shapes that can result from the intersection of a plane and a cube -/
inductive CrossSection
  | EquilateralTriangle
  | Square
  | RegularPentagon
  | RegularHexagon

/-- The intersection of a plane and a cube -/
def intersect (c : Cube) (p : Plane) : CrossSection := sorry

/-- Theorem stating that a regular pentagon cannot be a cross-section of a cube -/
theorem no_pentagon_cross_section (c : Cube) (p : Plane) :
  intersect c p ≠ CrossSection.RegularPentagon := by sorry

end no_pentagon_cross_section_l487_48734


namespace warehouse_notebooks_l487_48764

/-- The number of notebooks in a warehouse --/
def total_notebooks (num_boxes : ℕ) (parts_per_box : ℕ) (notebooks_per_part : ℕ) : ℕ :=
  num_boxes * parts_per_box * notebooks_per_part

/-- Theorem: The total number of notebooks in the warehouse is 660 --/
theorem warehouse_notebooks : 
  total_notebooks 22 6 5 = 660 := by
  sorry

end warehouse_notebooks_l487_48764


namespace second_train_length_l487_48754

/-- Calculates the length of the second train given the parameters of two trains passing each other. -/
theorem second_train_length
  (first_train_length : ℝ)
  (first_train_speed : ℝ)
  (second_train_speed : ℝ)
  (initial_distance : ℝ)
  (crossing_time : ℝ)
  (h1 : first_train_length = 100)
  (h2 : first_train_speed = 10)
  (h3 : second_train_speed = 15)
  (h4 : initial_distance = 50)
  (h5 : crossing_time = 60)
  : ∃ (second_train_length : ℝ),
    second_train_length = 150 ∧
    second_train_length + first_train_length + initial_distance =
      (second_train_speed - first_train_speed) * crossing_time :=
by
  sorry


end second_train_length_l487_48754


namespace price_first_box_is_two_l487_48719

/-- The price of each movie in the first box -/
def price_first_box : ℝ := 2

/-- The number of movies bought from the first box -/
def num_first_box : ℕ := 10

/-- The number of movies bought from the second box -/
def num_second_box : ℕ := 5

/-- The price of each movie in the second box -/
def price_second_box : ℝ := 5

/-- The average price of all DVDs bought -/
def average_price : ℝ := 3

/-- The total number of movies bought -/
def total_movies : ℕ := num_first_box + num_second_box

theorem price_first_box_is_two :
  price_first_box * num_first_box + price_second_box * num_second_box = average_price * total_movies :=
by sorry

end price_first_box_is_two_l487_48719


namespace fourth_power_of_cube_of_third_smallest_prime_l487_48729

def third_smallest_prime : ℕ := 5

theorem fourth_power_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 4 = 244140625 := by
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l487_48729


namespace base_n_multiple_of_11_l487_48718

theorem base_n_multiple_of_11 : 
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 100 → 
  ¬(11 ∣ (7 + 4*n + 6*n^2 + 3*n^3 + 4*n^4 + 3*n^5)) := by
sorry

end base_n_multiple_of_11_l487_48718


namespace fraction_equality_l487_48773

theorem fraction_equality (a b c d : ℝ) (h : a / b = c / d) :
  (a + c) / (b + d) = a / b :=
by sorry

end fraction_equality_l487_48773


namespace x_minus_y_positive_l487_48774

theorem x_minus_y_positive (x y a : ℝ) 
  (h1 : x + y > 0) 
  (h2 : a < 0) 
  (h3 : a * y > 0) : 
  x - y > 0 := by sorry

end x_minus_y_positive_l487_48774


namespace ten_person_tournament_matches_l487_48750

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of matches in a 10-person round-robin tournament is 45 -/
theorem ten_person_tournament_matches :
  num_matches 10 = 45 := by
  sorry

/-- Lemma: The number of matches formula is valid for any number of players -/
lemma num_matches_formula_valid (n : ℕ) (n_ge_2 : n ≥ 2) :
  num_matches n = (n * (n - 1)) / 2 := by
  sorry

end ten_person_tournament_matches_l487_48750


namespace condition_sufficiency_for_increasing_f_l487_48788

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 3

theorem condition_sufficiency_for_increasing_f :
  (∀ x ≥ 2, Monotone (f 1)) ∧
  ¬(∀ a : ℝ, (∀ x ≥ 2, Monotone (f a)) → a = 1) :=
sorry

end condition_sufficiency_for_increasing_f_l487_48788


namespace square_area_from_rectangle_perimeter_l487_48760

/-- Given a square divided into 4 identical rectangles, each with a perimeter of 20,
    the area of the square is 1600/9. -/
theorem square_area_from_rectangle_perimeter :
  ∀ (s : ℝ), s > 0 →
  (∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * (l + w) = 20 ∧ 2 * l = s ∧ 2 * w = s) →
  s^2 = 1600 / 9 := by
sorry

end square_area_from_rectangle_perimeter_l487_48760


namespace snack_eaters_count_l487_48744

def snack_eaters_final (initial_attendees : ℕ) 
  (first_hour_eat_percent : ℚ) 
  (first_hour_not_eat_percent : ℚ)
  (second_hour_undecided_join_percent : ℚ)
  (second_hour_not_eat_join_percent : ℚ)
  (second_hour_newcomers : ℕ)
  (second_hour_newcomers_eat : ℕ)
  (second_hour_leave_percent : ℚ)
  (third_hour_increase_percent : ℚ)
  (third_hour_leave_percent : ℚ)
  (fourth_hour_latecomers : ℕ)
  (fourth_hour_latecomers_eat_percent : ℚ)
  (fourth_hour_workshop_leave : ℕ) : ℕ :=
  sorry

theorem snack_eaters_count : 
  snack_eaters_final 7500 (55/100) (35/100) (20/100) (15/100) 75 50 (40/100) (10/100) (1/2) 150 (60/100) 300 = 1347 := by
  sorry

end snack_eaters_count_l487_48744


namespace parabola_point_distance_l487_48716

/-- For a parabola y^2 = 2x, the x-coordinate of a point on the parabola
    that is at a distance of 3 from its focus is 5/2. -/
theorem parabola_point_distance (x y : ℝ) : 
  y^2 = 2*x →  -- parabola equation
  (x + 1/2)^2 + y^2 = 3^2 →  -- distance from focus is 3
  x = 5/2 := by
sorry

end parabola_point_distance_l487_48716


namespace complex_number_on_line_l487_48702

def complex_number (a : ℝ) : ℂ := (a : ℂ) - Complex.I

theorem complex_number_on_line (a : ℝ) :
  let z := 1 / complex_number a
  (z.re : ℝ) + 2 * (z.im : ℝ) = 0 → a = -2 :=
by sorry

end complex_number_on_line_l487_48702


namespace four_pepperoni_slices_left_l487_48727

/-- Represents the pizza sharing scenario -/
structure PizzaSharing where
  total_people : ℕ
  pepperoni_slices : ℕ
  cheese_slices : ℕ
  cheese_left : ℕ
  pepperoni_only_eaters : ℕ

/-- Calculate the number of pepperoni slices left -/
def pepperoni_left (ps : PizzaSharing) : ℕ :=
  let cheese_eaten := ps.cheese_slices - ps.cheese_left
  let slices_per_person := cheese_eaten / (ps.total_people - ps.pepperoni_only_eaters)
  let pepperoni_eaten := slices_per_person * (ps.total_people - ps.pepperoni_only_eaters) + 
                         slices_per_person * ps.pepperoni_only_eaters
  ps.pepperoni_slices - pepperoni_eaten

/-- Theorem stating that given the conditions, 4 pepperoni slices are left -/
theorem four_pepperoni_slices_left : 
  ∀ (ps : PizzaSharing), 
  ps.total_people = 4 ∧ 
  ps.pepperoni_slices = 16 ∧ 
  ps.cheese_slices = 16 ∧ 
  ps.cheese_left = 7 ∧ 
  ps.pepperoni_only_eaters = 1 →
  pepperoni_left ps = 4 := by
  sorry

end four_pepperoni_slices_left_l487_48727


namespace product_of_largest_primes_eq_679679_l487_48755

/-- The largest one-digit prime number -/
def largest_one_digit_prime : ℕ := 7

/-- The largest two-digit prime number -/
def largest_two_digit_prime : ℕ := 97

/-- The largest three-digit prime number -/
def largest_three_digit_prime : ℕ := 997

/-- The product of the largest one-digit, two-digit, and three-digit primes -/
def product_of_largest_primes : ℕ := 
  largest_one_digit_prime * largest_two_digit_prime * largest_three_digit_prime

theorem product_of_largest_primes_eq_679679 : 
  product_of_largest_primes = 679679 := by
  sorry

end product_of_largest_primes_eq_679679_l487_48755


namespace consecutive_years_product_l487_48742

theorem consecutive_years_product : (2014 - 2013) * (2013 - 2012) = 1 := by
  sorry

end consecutive_years_product_l487_48742


namespace document_word_count_l487_48706

/-- Calculates the approximate total number of words in a document -/
def approx_total_words (num_pages : ℕ) (avg_words_per_page : ℕ) : ℕ :=
  num_pages * avg_words_per_page

/-- Theorem stating that a document with 8 pages and an average of 605 words per page has approximately 4800 words in total -/
theorem document_word_count : approx_total_words 8 605 = 4800 := by
  sorry

end document_word_count_l487_48706


namespace min_positive_translation_for_symmetry_l487_48756

open Real

theorem min_positive_translation_for_symmetry (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = sin (2 * x + π / 4)) →
  (∀ x, f (x - φ) = f (-x)) →
  (φ > 0) →
  (∀ ψ, ψ > 0 → ψ < φ → ¬(∀ x, f (x - ψ) = f (-x))) →
  φ = 3 * π / 8 := by
sorry

end min_positive_translation_for_symmetry_l487_48756


namespace intersection_of_A_and_B_l487_48782

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end intersection_of_A_and_B_l487_48782


namespace misread_weight_l487_48730

theorem misread_weight (n : ℕ) (initial_avg correct_avg correct_weight : ℝ) :
  n = 20 →
  initial_avg = 58.4 →
  correct_avg = 58.9 →
  correct_weight = 66 →
  ∃ (misread_weight : ℝ),
    n * initial_avg + (correct_weight - misread_weight) = n * correct_avg ∧
    misread_weight = 56 :=
by sorry

end misread_weight_l487_48730


namespace solution_set_of_inequalities_l487_48721

theorem solution_set_of_inequalities :
  let S := {x : ℝ | x - 2 > 1 ∧ x < 4}
  S = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end solution_set_of_inequalities_l487_48721


namespace inheritance_problem_l487_48714

theorem inheritance_problem (S₁ S₂ S₃ S₄ D N : ℕ) :
  S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 ∧ D > 0 ∧ N > 0 →
  Nat.sqrt S₁ = S₂ / 2 →
  Nat.sqrt S₁ = S₃ - 2 →
  Nat.sqrt S₁ = S₄ + 2 →
  Nat.sqrt S₁ = 2 * D →
  Nat.sqrt S₁ = N * N →
  S₁ + S₂ + S₃ + S₄ + D + N < 1500 →
  S₁ + S₂ + S₃ + S₄ + D + N = 1464 :=
by sorry

#eval Nat.sqrt 1296  -- Should output 36
#eval 72 / 2         -- Should output 36
#eval 38 - 2         -- Should output 36
#eval 34 + 2         -- Should output 36
#eval 2 * 18         -- Should output 36
#eval 6 * 6          -- Should output 36
#eval 1296 + 72 + 38 + 34 + 18 + 6  -- Should output 1464

end inheritance_problem_l487_48714


namespace dodecagon_diagonals_and_triangles_l487_48701

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def number_of_triangles (n : ℕ) : ℕ := n.choose 3

theorem dodecagon_diagonals_and_triangles :
  let n : ℕ := 12
  number_of_diagonals n = 54 ∧ number_of_triangles n = 220 := by sorry

end dodecagon_diagonals_and_triangles_l487_48701


namespace zeros_before_first_nonzero_of_fraction_l487_48703

/-- The number of zeros after the decimal point and before the first non-zero digit
    in the decimal representation of 1/(2^3 * 5^6) -/
def zeros_before_first_nonzero : ℕ :=
  6

/-- The fraction we're considering -/
def fraction : ℚ :=
  1 / (2^3 * 5^6)

theorem zeros_before_first_nonzero_of_fraction :
  zeros_before_first_nonzero = 
    (fraction.den.factors.count 2 + fraction.den.factors.count 5).min
      (fraction.den.factors.count 5) :=
by
  sorry

end zeros_before_first_nonzero_of_fraction_l487_48703


namespace m_eq_2_sufficient_not_necessary_l487_48784

/-- Two vectors are collinear if their cross product is zero -/
def are_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Vector a is defined as (1, m-1) -/
def a (m : ℝ) : ℝ × ℝ := (1, m - 1)

/-- Vector b is defined as (m, 2) -/
def b (m : ℝ) : ℝ × ℝ := (m, 2)

/-- Theorem stating that m = 2 is a sufficient but not necessary condition for collinearity -/
theorem m_eq_2_sufficient_not_necessary :
  (∀ m : ℝ, m = 2 → are_collinear (a m) (b m)) ∧
  ¬(∀ m : ℝ, are_collinear (a m) (b m) → m = 2) :=
by sorry

end m_eq_2_sufficient_not_necessary_l487_48784


namespace quadratic_no_real_roots_l487_48770

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := by
  sorry

#check quadratic_no_real_roots

end quadratic_no_real_roots_l487_48770


namespace AB_range_l487_48743

/-- Represents an acute triangle ABC with specific properties -/
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real
  acute : A < 90 ∧ B < 90 ∧ C < 90
  angle_sum : A + B + C = 180
  angle_A : A = 60
  side_BC : BC = 6
  side_AB : AB > 0

/-- Theorem stating the range of AB in the specific acute triangle -/
theorem AB_range (t : AcuteTriangle) : 2 * Real.sqrt 3 < t.AB ∧ t.AB < 4 * Real.sqrt 3 := by
  sorry

#check AB_range

end AB_range_l487_48743


namespace min_sum_factors_2400_l487_48708

/-- The minimum sum of two positive integer factors of 2400 -/
theorem min_sum_factors_2400 : ∀ a b : ℕ+, a * b = 2400 → (∀ c d : ℕ+, c * d = 2400 → a + b ≤ c + d) → a + b = 98 := by
  sorry

end min_sum_factors_2400_l487_48708


namespace ratio_problem_l487_48778

theorem ratio_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : 2 * (a + b) = 3 * (a - b)) :
  a = 5 * b := by
sorry

end ratio_problem_l487_48778


namespace ticket_cost_calculation_l487_48736

/-- The total cost of tickets for various events -/
def total_cost (movie_price : ℚ) (football_price : ℚ) (concert_price : ℚ) (theater_price : ℚ) : ℚ :=
  8 * movie_price + 5 * football_price + 3 * concert_price + 4 * theater_price

/-- The theorem stating the total cost of tickets -/
theorem ticket_cost_calculation : ∃ (movie_price football_price concert_price theater_price : ℚ),
  (8 * movie_price = 2 * football_price) ∧
  (movie_price = 30) ∧
  (concert_price = football_price - 10) ∧
  (theater_price = 40 * (1 - 0.1)) ∧
  (total_cost movie_price football_price concert_price theater_price = 1314) :=
by
  sorry


end ticket_cost_calculation_l487_48736


namespace derivative_negative_two_exp_times_sin_l487_48715

theorem derivative_negative_two_exp_times_sin (x : ℝ) :
  deriv (λ x => -2 * Real.exp x * Real.sin x) x = -2 * Real.exp x * (Real.sin x + Real.cos x) := by
  sorry

end derivative_negative_two_exp_times_sin_l487_48715


namespace parabola_parameter_range_l487_48786

theorem parabola_parameter_range (a m n : ℝ) : 
  a ≠ 0 → 
  n = a * m^2 - 4 * a^2 * m - 3 →
  0 ≤ m → m ≤ 4 → 
  n ≤ -3 →
  (a ≥ 1 ∨ a < 0) :=
sorry

end parabola_parameter_range_l487_48786


namespace cake_price_l487_48737

theorem cake_price (num_cakes : ℕ) (num_pies : ℕ) (pie_price : ℚ) (total_revenue : ℚ) : 
  num_cakes = 453 → 
  num_pies = 126 → 
  pie_price = 7 → 
  total_revenue = 6318 → 
  ∃ (cake_price : ℚ), cake_price = 12 ∧ num_cakes * cake_price + num_pies * pie_price = total_revenue :=
by
  sorry

end cake_price_l487_48737


namespace polynomial_remainder_l487_48725

theorem polynomial_remainder (p : ℤ) : (p^11 - 3) % (p - 2) = 2045 := by
  sorry

end polynomial_remainder_l487_48725


namespace range_of_x_l487_48745

theorem range_of_x (x : ℝ) : 
  (6 - 3 * x ≥ 0) ∧ (1 / (x + 1) ≥ 0) → x ∈ Set.Icc (-1) 2 := by
  sorry

end range_of_x_l487_48745


namespace complex_magnitude_theorem_l487_48781

def complex_equation (z : ℂ) : Prop :=
  (1 - Complex.I) * z = (1 + Complex.I)^2

theorem complex_magnitude_theorem (z : ℂ) :
  complex_equation z → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_theorem_l487_48781


namespace f_derivative_at_zero_l487_48740

def f (x : ℝ) : ℝ := x^2 + 2*x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by sorry

end f_derivative_at_zero_l487_48740


namespace line_intersects_at_least_one_l487_48775

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of being skew lines
variable (skew : Line → Line → Prop)

-- Define the property of intersection
variable (intersects : Line → Line → Prop)

-- Define the property of a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (plane_intersection : Plane → Plane → Line)

-- State the theorem
theorem line_intersects_at_least_one 
  (m n l : Line) (α β : Plane) :
  skew m n →
  ¬(intersects l m) →
  ¬(intersects l n) →
  in_plane n β →
  plane_intersection α β = l →
  (intersects l m) ∨ (intersects l n) :=
sorry

end line_intersects_at_least_one_l487_48775


namespace cyclic_sum_inequality_l487_48767

theorem cyclic_sum_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  a*b/c + b*c/a + c*a/b ≥ Real.sqrt 3 := by
  sorry

end cyclic_sum_inequality_l487_48767


namespace property_implies_linear_l487_48785

/-- A function f: ℚ → ℚ satisfies the given property if
    f(x) + f(t) = f(y) + f(z) for all rational x < y < z < t
    that form an arithmetic progression -/
def SatisfiesProperty (f : ℚ → ℚ) : Prop :=
  ∀ (x y z t : ℚ) (d : ℚ), 0 < d → x < y ∧ y < z ∧ z < t →
  y = x + d ∧ z = y + d ∧ t = z + d →
  f x + f t = f y + f z

/-- A function f: ℚ → ℚ is linear if there exist rational m and b
    such that f(x) = mx + b for all rational x -/
def IsLinear (f : ℚ → ℚ) : Prop :=
  ∃ (m b : ℚ), ∀ (x : ℚ), f x = m * x + b

theorem property_implies_linear (f : ℚ → ℚ) :
  SatisfiesProperty f → IsLinear f := by
  sorry

end property_implies_linear_l487_48785


namespace circles_tangent_implies_m_equals_four_l487_48751

-- Define the circles
def circle_C (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 5 - m}
def circle_E : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 16}

-- Define the condition for external tangency
def externally_tangent (C E : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ C ∧ p ∈ E ∧ 
  ∀ (q : ℝ × ℝ), q ∈ C ∩ E → q = p

-- State the theorem
theorem circles_tangent_implies_m_equals_four :
  ∀ (m : ℝ), externally_tangent (circle_C m) circle_E → m = 4 := by
  sorry

end circles_tangent_implies_m_equals_four_l487_48751


namespace cookies_problem_l487_48705

theorem cookies_problem (mona jasmine rachel : ℕ) 
  (h1 : jasmine = mona - 5)
  (h2 : rachel = jasmine + 10)
  (h3 : mona + jasmine + rachel = 60) :
  mona = 20 := by
sorry

end cookies_problem_l487_48705


namespace no_divisors_between_30_and_40_l487_48710

theorem no_divisors_between_30_and_40 : ∀ n : ℕ, 30 < n → n < 40 → ¬(2^28 - 1) % n = 0 := by
  sorry

end no_divisors_between_30_and_40_l487_48710


namespace ounces_in_pound_l487_48752

/-- Represents the number of ounces in one pound -/
def ounces_per_pound : ℕ := sorry

theorem ounces_in_pound : 
  (2100 : ℕ) * 13 = 1680 * (16 + 4 / ounces_per_pound) → ounces_per_pound = 16 := by
  sorry

end ounces_in_pound_l487_48752


namespace paperware_cost_relationship_l487_48794

/-- Represents the cost of paper plates and cups -/
structure PaperwareCost where
  plate : ℝ
  cup : ℝ

/-- The total cost of a given number of plates and cups -/
def total_cost (c : PaperwareCost) (plates : ℝ) (cups : ℝ) : ℝ :=
  c.plate * plates + c.cup * cups

/-- Theorem stating the relationship between the costs of different quantities of plates and cups -/
theorem paperware_cost_relationship (c : PaperwareCost) :
  total_cost c 20 40 = 1.20 → total_cost c 100 200 = 6.00 := by
  sorry

end paperware_cost_relationship_l487_48794


namespace twelve_factorial_mod_thirteen_l487_48799

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

theorem twelve_factorial_mod_thirteen : factorial 12 % 13 = 12 := by
  sorry

end twelve_factorial_mod_thirteen_l487_48799


namespace pen_cost_l487_48789

theorem pen_cost (notebook pen case : ℝ) 
  (total_cost : notebook + pen + case = 3.50)
  (pen_triple : pen = 3 * notebook)
  (case_more : case = notebook + 0.50) :
  pen = 1.80 := by
  sorry

end pen_cost_l487_48789


namespace smoothie_combinations_l487_48787

theorem smoothie_combinations : 
  let num_flavors : ℕ := 5
  let num_toppings : ℕ := 8
  let topping_choices : ℕ := 3
  num_flavors * (Nat.choose num_toppings topping_choices) = 280 :=
by sorry

end smoothie_combinations_l487_48787


namespace min_sum_with_exponential_constraint_l487_48777

theorem min_sum_with_exponential_constraint (a b : ℝ) :
  a > 0 → b > 0 → (2 : ℝ)^a * 4^b = (2^a)^b →
  (∀ x y : ℝ, x > 0 → y > 0 → (2 : ℝ)^x * 4^y = (2^x)^y → a + b ≤ x + y) →
  a + b = 3 + 2 * Real.sqrt 2 :=
by sorry

end min_sum_with_exponential_constraint_l487_48777


namespace fixed_point_of_exponential_function_l487_48748

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2*x - 1) + 2
  f (1/2) = 3 := by sorry

end fixed_point_of_exponential_function_l487_48748


namespace sin_shift_l487_48776

theorem sin_shift (x : ℝ) : Real.sin (3 * x - π / 3) = Real.sin (3 * (x - π / 9)) := by
  sorry

end sin_shift_l487_48776


namespace car_initial_payment_l487_48769

/-- Calculates the initial payment for a car purchase given the total cost,
    monthly payment, and number of months. -/
def initial_payment (total_cost monthly_payment num_months : ℕ) : ℕ :=
  total_cost - monthly_payment * num_months

theorem car_initial_payment :
  initial_payment 13380 420 19 = 5400 := by
  sorry

end car_initial_payment_l487_48769


namespace notebook_calculation_sara_sister_notebooks_l487_48720

theorem notebook_calculation : ℕ → ℕ
  | initial =>
    let ordered := initial + (initial * 3 / 2)
    let after_loss := ordered - 2
    let after_sale := after_loss - (after_loss / 4)
    let final := after_sale - 3
    final

theorem sara_sister_notebooks :
  notebook_calculation 4 = 3 := by sorry

end notebook_calculation_sara_sister_notebooks_l487_48720


namespace mosaic_perimeter_l487_48779

/-- A mosaic constructed with a regular hexagon, squares, and equilateral triangles. -/
structure Mosaic where
  hexagon_side_length : ℝ
  num_squares : ℕ
  num_triangles : ℕ

/-- The outside perimeter of the mosaic. -/
def outside_perimeter (m : Mosaic) : ℝ :=
  (m.num_squares + m.num_triangles) * m.hexagon_side_length

/-- Theorem stating that the outside perimeter of the specific mosaic is 240 cm. -/
theorem mosaic_perimeter :
  ∀ (m : Mosaic),
    m.hexagon_side_length = 20 →
    m.num_squares = 6 →
    m.num_triangles = 6 →
    outside_perimeter m = 240 := by
  sorry

end mosaic_perimeter_l487_48779


namespace function_equality_implies_m_value_l487_48768

-- Define the functions f and g
def f (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + m
def g (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + 5*m

-- State the theorem
theorem function_equality_implies_m_value :
  ∀ m : ℚ, 3 * (f m 5) = 2 * (g m 5) → m = 10/7 := by
  sorry

end function_equality_implies_m_value_l487_48768


namespace luke_coin_count_l487_48762

theorem luke_coin_count (quarter_piles dime_piles coins_per_pile : ℕ) 
  (h1 : quarter_piles = 5)
  (h2 : dime_piles = 5)
  (h3 : coins_per_pile = 3) : 
  quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 30 := by
  sorry

end luke_coin_count_l487_48762


namespace smallest_sum_mn_l487_48791

theorem smallest_sum_mn (m n : ℕ) (hm : m > n) (h_div : (70^2 : ℕ) ∣ (2023^m - 2023^n)) : m + n ≥ 24 ∧ ∃ (m₀ n₀ : ℕ), m₀ + n₀ = 24 ∧ m₀ > n₀ ∧ (70^2 : ℕ) ∣ (2023^m₀ - 2023^n₀) :=
sorry

end smallest_sum_mn_l487_48791


namespace percentage_relation_l487_48713

theorem percentage_relation (A B x : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : A = (x / 100) * B) : 
  x = 100 * (A / B) := by
sorry

end percentage_relation_l487_48713


namespace polynomial_simplification_l487_48797

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 3 * x^2 + 8 * x - 5) - (x^3 + 6 * x^2 + 2 * x - 15) - (2 * x^3 + x^2 + 4 * x - 8) = 
  -4 * x^2 + 2 * x + 18 := by
  sorry

end polynomial_simplification_l487_48797


namespace no_eight_face_polyhedron_from_cube_cut_l487_48753

/-- Represents a polyhedron --/
structure Polyhedron where
  faces : ℕ

/-- Represents a cube --/
structure Cube where
  faces : ℕ
  faces_eq_six : faces = 6

/-- Represents the result of cutting a cube with a single plane --/
structure CubeCut where
  original : Cube
  piece1 : Polyhedron
  piece2 : Polyhedron
  single_cut : piece1.faces + piece2.faces = original.faces + 2

/-- Theorem stating that a polyhedron with 8 faces cannot be obtained from cutting a cube with a single plane --/
theorem no_eight_face_polyhedron_from_cube_cut (cut : CubeCut) :
  cut.piece1.faces ≠ 8 ∧ cut.piece2.faces ≠ 8 := by
  sorry

end no_eight_face_polyhedron_from_cube_cut_l487_48753


namespace production_reduction_for_breakeven_l487_48747

/-- The problem setup and proof statement --/
theorem production_reduction_for_breakeven (initial_production : ℕ) 
  (price_per_item : ℚ) (profit : ℚ) (variable_cost_per_item : ℚ) 
  (h1 : initial_production = 4000)
  (h2 : price_per_item = 6250)
  (h3 : profit = 2000000)
  (h4 : variable_cost_per_item = 3750) : 
  let constant_costs := initial_production * price_per_item - profit
  let breakeven_production := constant_costs / (price_per_item - variable_cost_per_item)
  (initial_production - breakeven_production) / initial_production = 1/5 := by
  sorry

end production_reduction_for_breakeven_l487_48747


namespace repeating_decimal_subtraction_l487_48790

/-- The value of a repeating decimal 0.abcabcabc... where a, b, c are digits -/
def repeating_decimal (a b c : Nat) : ℚ := (100 * a + 10 * b + c : ℚ) / 999

/-- Theorem stating that 0.246246246... - 0.135135135... - 0.012012012... = 1/9 -/
theorem repeating_decimal_subtraction :
  repeating_decimal 2 4 6 - repeating_decimal 1 3 5 - repeating_decimal 0 1 2 = 1 / 9 := by
  sorry

end repeating_decimal_subtraction_l487_48790


namespace perpendicular_planes_l487_48771

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (a b c : Line) 
  (α β γ : Plane) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h3 : perpendicular a α)
  (h4 : subset b β)
  (h5 : parallel a b) :
  plane_perpendicular α β :=
sorry

end perpendicular_planes_l487_48771


namespace right_triangle_square_distance_l487_48759

/-- Given a right triangle with hypotenuse forming the side of a square outside the triangle,
    and the sum of the legs being d, the distance from the right angle vertex to the center
    of the square is (d * √2) / 2. -/
theorem right_triangle_square_distance (d : ℝ) (h : d > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b = d ∧
    a^2 + b^2 = c^2 ∧
    (d * Real.sqrt 2) / 2 = c * Real.sqrt 2 / 2 :=
by sorry

end right_triangle_square_distance_l487_48759


namespace f_max_property_l487_48765

def f_properties (f : ℚ → ℚ) : Prop :=
  (f 0 = 0) ∧
  (∀ α : ℚ, α ≠ 0 → f α > 0) ∧
  (∀ α β : ℚ, f (α * β) = f α * f β) ∧
  (∀ α β : ℚ, f (α + β) ≤ f α + f β) ∧
  (∀ m : ℤ, f m ≤ 1989)

theorem f_max_property (f : ℚ → ℚ) (h : f_properties f) :
  ∀ α β : ℚ, f α ≠ f β → f (α + β) = max (f α) (f β) := by
  sorry

end f_max_property_l487_48765


namespace a_4_plus_a_5_l487_48783

def a : ℕ → ℕ
  | n => if n % 2 = 1 then 2 * n + 1 else 2^n

theorem a_4_plus_a_5 : a 4 + a 5 = 27 := by
  sorry

end a_4_plus_a_5_l487_48783


namespace trapezoid_balance_l487_48793

-- Define the shapes and their weights
variable (C P T : ℝ)

-- Define the balance conditions
axiom balance1 : C = 2 * P
axiom balance2 : T = C + P

-- Theorem to prove
theorem trapezoid_balance : T = 3 * P := by
  sorry

end trapezoid_balance_l487_48793


namespace prime_sum_less_than_ten_l487_48780

theorem prime_sum_less_than_ten (d e f : ℕ) : 
  Prime d → Prime e → Prime f →
  d < 10 → e < 10 → f < 10 →
  d + e = f →
  d < e →
  d = 2 := by
sorry

end prime_sum_less_than_ten_l487_48780


namespace hyperbola_ellipse_dot_product_l487_48728

-- Define the hyperbola C'
def hyperbola_C' (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

-- Define the dot product of AP and BP
def AP_dot_BP (x y : ℝ) : ℝ :=
  (x + 1) * (x - 1) + y * y

-- Define the range of AP⋅BP
def range_AP_dot_BP : Set ℝ :=
  {z : ℝ | 191/34 ≤ z ∧ z ≤ 24}

-- Theorem statement
theorem hyperbola_ellipse_dot_product :
  -- Conditions
  (∀ x y : ℝ, 3*x = 4*y ∨ 3*x = -4*y → ¬(hyperbola_C' x y)) →  -- Asymptotes
  (hyperbola_C' 5 (9/4)) →  -- Hyperbola passes through (5, 9/4)
  (∃ x₀ : ℝ, x₀ > 0 ∧ ellipse_M x₀ 0 ∧ hyperbola_C' x₀ 0) →  -- Shared focus/vertex
  (∀ x y : ℝ, ellipse_M x y → x ≤ 5 ∧ y ≤ 3) →  -- Bounds on ellipse
  -- Conclusions
  (∀ x y : ℝ, hyperbola_C' x y ↔ x^2 / 16 - y^2 / 9 = 1) ∧
  (∀ x y : ℝ, ellipse_M x y ↔ x^2 / 25 + y^2 / 9 = 1) ∧
  (∀ x y : ℝ, ellipse_M x y ∧ x ≥ 0 → AP_dot_BP x y ∈ range_AP_dot_BP) :=
by sorry

end hyperbola_ellipse_dot_product_l487_48728


namespace clothing_distribution_l487_48738

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 47)
  (h2 : first_load = 17)
  (h3 : num_small_loads = 5) :
  (total - first_load) / num_small_loads = 6 := by
  sorry

end clothing_distribution_l487_48738


namespace game_show_probability_l487_48722

theorem game_show_probability : 
  let num_tables : ℕ := 3
  let boxes_per_table : ℕ := 3
  let zonk_boxes_per_table : ℕ := 1
  let prob_no_zonk_per_table : ℚ := (boxes_per_table - zonk_boxes_per_table) / boxes_per_table
  (prob_no_zonk_per_table ^ num_tables : ℚ) = 8 / 27 := by sorry

end game_show_probability_l487_48722


namespace t_square_four_equal_parts_l487_48792

/-- A figure composed of three equal squares -/
structure TSquareFigure where
  square_area : ℝ
  total_area : ℝ
  h_total_area : total_area = 3 * square_area

/-- A division of the figure into four parts -/
structure FourPartDivision (fig : TSquareFigure) where
  part_area : ℝ
  h_part_count : ℕ
  h_part_count_eq : h_part_count = 4
  h_total_area : fig.total_area = h_part_count * part_area

/-- Theorem stating that a T-square figure can be divided into four equal parts -/
theorem t_square_four_equal_parts (fig : TSquareFigure) : 
  ∃ (div : FourPartDivision fig), div.part_area = 3 * fig.square_area / 4 := by
  sorry

end t_square_four_equal_parts_l487_48792


namespace cake_angle_theorem_l487_48733

theorem cake_angle_theorem (n : ℕ) (initial_angle : ℝ) (final_angle : ℝ) : 
  n = 10 →
  initial_angle = 360 / n →
  final_angle = 360 / (n - 1) →
  final_angle - initial_angle = 4 := by
  sorry

end cake_angle_theorem_l487_48733


namespace parallel_line_slope_l487_48798

/-- Given a line with equation 3x - 6y = 12, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) : 
  (3 * x - 6 * y = 12) → 
  (∃ (m b : ℝ), ∀ (x' y' : ℝ), y' = m * x' + b ∧ (3 * x' - 6 * y' = 12)) → 
  m = 1/2 := by
  sorry

end parallel_line_slope_l487_48798


namespace chessboard_rearrangement_3x3_no_chessboard_rearrangement_8x8_l487_48724

/-- Represents a chessboard of size N × N -/
structure Chessboard (N : ℕ) where
  size : ℕ
  size_eq : size = N

/-- Represents a position on the chessboard -/
structure Position (N : ℕ) where
  row : Fin N
  col : Fin N

/-- Knight's move distance between two positions -/
def knightDistance (N : ℕ) (p1 p2 : Position N) : ℕ :=
  sorry

/-- King's move distance between two positions -/
def kingDistance (N : ℕ) (p1 p2 : Position N) : ℕ :=
  sorry

/-- Represents a rearrangement of checkers on the board -/
def Rearrangement (N : ℕ) := Position N → Position N

/-- Checks if a rearrangement satisfies the problem condition -/
def isValidRearrangement (N : ℕ) (r : Rearrangement N) : Prop :=
  ∀ p1 p2 : Position N, knightDistance N p1 p2 = 1 → kingDistance N (r p1) (r p2) = 1

theorem chessboard_rearrangement_3x3 :
  ∃ (r : Rearrangement 3), isValidRearrangement 3 r :=
sorry

theorem no_chessboard_rearrangement_8x8 :
  ¬ ∃ (r : Rearrangement 8), isValidRearrangement 8 r :=
sorry

end chessboard_rearrangement_3x3_no_chessboard_rearrangement_8x8_l487_48724
