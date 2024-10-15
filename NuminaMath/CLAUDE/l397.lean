import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_theorem_l397_39773

theorem square_difference_theorem (a b M : ℝ) : 
  (a + 2*b)^2 = (a - 2*b)^2 + M → M = 8*a*b := by
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l397_39773


namespace NUMINAMATH_CALUDE_total_cost_after_increase_l397_39742

def price_increase : ℚ := 15 / 100

def original_orange_price : ℚ := 40
def original_mango_price : ℚ := 50

def new_orange_price : ℚ := original_orange_price * (1 + price_increase)
def new_mango_price : ℚ := original_mango_price * (1 + price_increase)

def total_cost : ℚ := 10 * new_orange_price + 10 * new_mango_price

theorem total_cost_after_increase :
  total_cost = 1035 := by sorry

end NUMINAMATH_CALUDE_total_cost_after_increase_l397_39742


namespace NUMINAMATH_CALUDE_total_price_after_increase_l397_39738

/-- Calculates the total price for a buyer purchasing jewelry and paintings 
    after a price increase. -/
theorem total_price_after_increase 
  (initial_jewelry_price : ℝ) 
  (initial_painting_price : ℝ)
  (jewelry_price_increase : ℝ)
  (painting_price_increase_percent : ℝ)
  (jewelry_quantity : ℕ)
  (painting_quantity : ℕ)
  (h1 : initial_jewelry_price = 30)
  (h2 : initial_painting_price = 100)
  (h3 : jewelry_price_increase = 10)
  (h4 : painting_price_increase_percent = 20)
  (h5 : jewelry_quantity = 2)
  (h6 : painting_quantity = 5) :
  let new_jewelry_price := initial_jewelry_price + jewelry_price_increase
  let new_painting_price := initial_painting_price * (1 + painting_price_increase_percent / 100)
  let total_price := new_jewelry_price * jewelry_quantity + new_painting_price * painting_quantity
  total_price = 680 := by
sorry


end NUMINAMATH_CALUDE_total_price_after_increase_l397_39738


namespace NUMINAMATH_CALUDE_shaded_area_between_squares_l397_39774

/-- The area of the shaded region between two squares -/
theorem shaded_area_between_squares (large_side small_side : ℝ) 
  (h1 : large_side = 10)
  (h2 : small_side = 5) :
  large_side^2 - small_side^2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_squares_l397_39774


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l397_39715

theorem decimal_to_fraction (x : ℚ) (h : x = 3.36) : 
  ∃ (n d : ℕ), d ≠ 0 ∧ x = n / d ∧ n = 84 ∧ d = 25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l397_39715


namespace NUMINAMATH_CALUDE_unique_valid_subset_l397_39710

def original_set : Finset ℕ := {1, 3, 4, 5, 7, 8, 9, 11, 12, 14}

def is_valid_subset (s : Finset ℕ) : Prop :=
  s.card = 2 ∧ 
  s ⊆ original_set ∧
  (Finset.sum (original_set \ s) id) / (original_set.card - 2 : ℚ) = 7

theorem unique_valid_subset : ∃! s : Finset ℕ, is_valid_subset s := by
  sorry

end NUMINAMATH_CALUDE_unique_valid_subset_l397_39710


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_range_l397_39792

/-- The range of k for which a hyperbola and parabola have at most two intersections -/
theorem hyperbola_parabola_intersection_range :
  ∀ k : ℝ,
  (∀ x y : ℝ, x^2 - y^2 + 1 = 0 ∧ y^2 = (k - 1) * x →
    (∃! p q : ℝ × ℝ, (p.1^2 - p.2^2 + 1 = 0 ∧ p.2^2 = (k - 1) * p.1) ∧
                     (q.1^2 - q.2^2 + 1 = 0 ∧ q.2^2 = (k - 1) * q.1) ∧
                     p ≠ q) ∨
    (∃! p : ℝ × ℝ, p.1^2 - p.2^2 + 1 = 0 ∧ p.2^2 = (k - 1) * p.1) ∨
    (∀ x y : ℝ, x^2 - y^2 + 1 ≠ 0 ∨ y^2 ≠ (k - 1) * x)) →
  -1 ≤ k ∧ k < 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_range_l397_39792


namespace NUMINAMATH_CALUDE_dave_train_books_l397_39779

/-- The number of books about trains Dave bought -/
def num_train_books (num_animal_books num_space_books cost_per_book total_spent : ℕ) : ℕ :=
  (total_spent - (num_animal_books + num_space_books) * cost_per_book) / cost_per_book

theorem dave_train_books :
  num_train_books 8 6 6 102 = 3 :=
sorry

end NUMINAMATH_CALUDE_dave_train_books_l397_39779


namespace NUMINAMATH_CALUDE_N_value_is_negative_twelve_point_five_l397_39794

/-- Represents a grid with arithmetic sequences -/
structure ArithmeticGrid :=
  (row_first : ℚ)
  (col1_second : ℚ)
  (col1_third : ℚ)
  (col2_last : ℚ)
  (num_columns : ℕ)
  (num_rows : ℕ)

/-- Calculates the value of N in the arithmetic grid -/
def calculate_N (grid : ArithmeticGrid) : ℚ :=
  sorry

/-- Theorem stating that N equals -12.5 for the given grid -/
theorem N_value_is_negative_twelve_point_five :
  let grid : ArithmeticGrid := {
    row_first := 18,
    col1_second := 15,
    col1_third := 21,
    col2_last := -14,
    num_columns := 7,
    num_rows := 2
  }
  calculate_N grid = -12.5 := by sorry

end NUMINAMATH_CALUDE_N_value_is_negative_twelve_point_five_l397_39794


namespace NUMINAMATH_CALUDE_product_of_integers_l397_39733

theorem product_of_integers (A B C D : ℕ+) : 
  A + B + C + D = 100 →
  2^(A:ℕ) = B - 4 →
  C + 6 = D →
  B + C = D + 10 →
  A * B * C * D = 33280 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l397_39733


namespace NUMINAMATH_CALUDE_cards_satisfy_conditions_l397_39784

def card1 : Finset Nat := {1, 4, 7}
def card2 : Finset Nat := {2, 3, 4}
def card3 : Finset Nat := {2, 5, 7}

theorem cards_satisfy_conditions : 
  (card1 ∩ card2).card = 1 ∧ 
  (card1 ∩ card3).card = 1 ∧ 
  (card2 ∩ card3).card = 1 := by
  sorry

end NUMINAMATH_CALUDE_cards_satisfy_conditions_l397_39784


namespace NUMINAMATH_CALUDE_infinite_triplets_exist_l397_39735

theorem infinite_triplets_exist : 
  ∀ x : ℝ, ∃ a b c : ℝ, a + b + c = 0 ∧ a^4 + b^4 + c^4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_infinite_triplets_exist_l397_39735


namespace NUMINAMATH_CALUDE_vector_properties_l397_39719

/-- Given vectors a and b, prove properties about vector c and scalar t. -/
theorem vector_properties (a b : ℝ × ℝ) (h_a : a = (1, 0)) (h_b : b = (-1, 2)) :
  -- Part 1
  (∃ c : ℝ × ℝ, ‖c‖ = 1 ∧ ∃ k : ℝ, c = k • (a - b)) →
  (∃ c : ℝ × ℝ, c = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) ∨ c = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2)) ∧
  -- Part 2
  (∃ t : ℝ, (2 * t • a - b) • (3 • a + t • b) = 0) →
  (∃ t : ℝ, t = -1 ∨ t = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l397_39719


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l397_39787

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 25 cm and height 15 cm is 375 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 25 15 = 375 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l397_39787


namespace NUMINAMATH_CALUDE_part1_min_max_part2_t_range_l397_39763

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := x^2 + 2*t*x + t - 1

-- Part 1
theorem part1_min_max :
  let t := 2
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc (-3) 1, f t x ≥ min) ∧
    (∃ x ∈ Set.Icc (-3) 1, f t x = min) ∧
    (∀ x ∈ Set.Icc (-3) 1, f t x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 1, f t x = max) ∧
    min = -3 ∧ max = 6 :=
sorry

-- Part 2
theorem part2_t_range :
  {t : ℝ | ∀ x ∈ Set.Icc 1 2, f t x > 0} = Set.Ioi 0 :=
sorry

end NUMINAMATH_CALUDE_part1_min_max_part2_t_range_l397_39763


namespace NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_four_l397_39714

theorem least_positive_integer_for_multiple_of_four :
  ∃ (n : ℕ), n > 0 ∧ (575 + n) % 4 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (575 + m) % 4 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_four_l397_39714


namespace NUMINAMATH_CALUDE_jeff_fills_130_boxes_l397_39713

/-- Calculates the number of boxes Jeff can fill with remaining donuts --/
def donut_boxes : ℕ :=
  let total_donuts := 50 * 30
  let jeff_eats := 3 * 30
  let friends_eat := 10 + 12 + 8
  let given_away := 25 + 50
  let unavailable := jeff_eats + friends_eat + given_away
  let remaining := total_donuts - unavailable
  remaining / 10

/-- Theorem stating that Jeff can fill 130 boxes with remaining donuts --/
theorem jeff_fills_130_boxes : donut_boxes = 130 := by
  sorry

end NUMINAMATH_CALUDE_jeff_fills_130_boxes_l397_39713


namespace NUMINAMATH_CALUDE_dvd_average_price_l397_39748

/-- Calculates the average price of DVDs bought from two different price groups -/
theorem dvd_average_price (n1 : ℕ) (p1 : ℚ) (n2 : ℕ) (p2 : ℚ) : 
  n1 = 10 → p1 = 2 → n2 = 5 → p2 = 5 → 
  (n1 * p1 + n2 * p2) / (n1 + n2 : ℚ) = 3 := by
sorry

end NUMINAMATH_CALUDE_dvd_average_price_l397_39748


namespace NUMINAMATH_CALUDE_total_water_in_boxes_l397_39769

theorem total_water_in_boxes (num_boxes : ℕ) (bottles_per_box : ℕ) (bottle_capacity : ℚ) (fill_ratio : ℚ) : 
  num_boxes = 10 →
  bottles_per_box = 50 →
  bottle_capacity = 12 →
  fill_ratio = 3/4 →
  (num_boxes * bottles_per_box * bottle_capacity * fill_ratio : ℚ) = 4500 := by
sorry

end NUMINAMATH_CALUDE_total_water_in_boxes_l397_39769


namespace NUMINAMATH_CALUDE_pipe_filling_time_l397_39767

/-- Given a tank and two pipes, prove that if one pipe takes T minutes to fill the tank,
    another pipe takes 12 minutes, and both pipes together take 4.8 minutes,
    then T = 8 minutes. -/
theorem pipe_filling_time (T : ℝ) : 
  (T > 0) →  -- T is positive (implied by the context)
  (1 / T + 1 / 12 = 1 / 4.8) →  -- Combined rate equation
  T = 8 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l397_39767


namespace NUMINAMATH_CALUDE_min_omega_for_50_maxima_l397_39729

theorem min_omega_for_50_maxima (ω : ℝ) : ω > 0 → (∀ x ∈ Set.Icc 0 1, ∃ y, y = Real.sin (ω * x)) →
  (∃ (maxima : Finset ℝ), maxima.card ≥ 50 ∧ 
    ∀ t ∈ maxima, t ∈ Set.Icc 0 1 ∧ 
    (∀ h ∈ Set.Icc 0 1, Real.sin (ω * t) ≥ Real.sin (ω * h))) →
  ω ≥ 197 * Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_for_50_maxima_l397_39729


namespace NUMINAMATH_CALUDE_intersection_implies_range_l397_39728

def A (a : ℝ) : Set ℝ := {x | |x - a| < 2}

def B : Set ℝ := {x | (2*x - 1) / (x + 2) < 1}

theorem intersection_implies_range (a : ℝ) : A a ∩ B = A a → a ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_range_l397_39728


namespace NUMINAMATH_CALUDE_parallel_dot_product_perpendicular_angle_l397_39720

noncomputable section

/-- Two vectors in a plane with given magnitudes and angle between them -/
structure VectorPair where
  a : ℝ × ℝ
  b : ℝ × ℝ
  mag_a : Real.sqrt (a.1^2 + a.2^2) = 1
  mag_b : Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2
  θ : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: If vectors are parallel, their dot product is ±√2 -/
theorem parallel_dot_product (vp : VectorPair) 
    (h_parallel : ∃ (k : ℝ), vp.a = k • vp.b ∨ vp.b = k • vp.a) : 
    dot_product vp.a vp.b = Real.sqrt 2 ∨ dot_product vp.a vp.b = -Real.sqrt 2 := by
  sorry

/-- Theorem: If a - b is perpendicular to a, then θ = 45° -/
theorem perpendicular_angle (vp : VectorPair) 
    (h_perp : dot_product (vp.a.1 - vp.b.1, vp.a.2 - vp.b.2) vp.a = 0) : 
    vp.θ = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_dot_product_perpendicular_angle_l397_39720


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l397_39726

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l397_39726


namespace NUMINAMATH_CALUDE_milk_for_cookies_l397_39702

/-- Given the ratio of cookies to milk, calculate the cups of milk needed for a given number of cookies -/
def milkNeeded (cookiesReference : ℕ) (quartsReference : ℕ) (cupsPerQuart : ℕ) (cookiesTarget : ℕ) : ℚ :=
  (quartsReference * cupsPerQuart : ℚ) * cookiesTarget / cookiesReference

theorem milk_for_cookies :
  milkNeeded 15 5 4 6 = 8 := by
  sorry

#eval milkNeeded 15 5 4 6

end NUMINAMATH_CALUDE_milk_for_cookies_l397_39702


namespace NUMINAMATH_CALUDE_sum_of_fractions_minus_seven_equals_negative_one_sixty_fourth_l397_39717

theorem sum_of_fractions_minus_seven_equals_negative_one_sixty_fourth : 
  10 * 56 * (3/2 + 5/4 + 9/8 + 17/16 + 33/32 + 65/64 - 7) = -1/64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_minus_seven_equals_negative_one_sixty_fourth_l397_39717


namespace NUMINAMATH_CALUDE_quadratic_root_range_l397_39722

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + (a^2 - 1)*x + a - 2 = 0 ∧ y^2 + (a^2 - 1)*y + a - 2 = 0 ∧ x > 1 ∧ y < 1) 
  → a > -2 ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l397_39722


namespace NUMINAMATH_CALUDE_olympic_year_zodiac_l397_39711

/-- The zodiac signs in order -/
inductive ZodiacSign
| Rat | Ox | Tiger | Rabbit | Dragon | Snake | Horse | Goat | Monkey | Rooster | Dog | Pig

/-- Function to get the zodiac sign for a given year -/
def getZodiacSign (year : Int) : ZodiacSign :=
  match (year - 1) % 12 with
  | 0 => ZodiacSign.Rooster
  | 1 => ZodiacSign.Dog
  | 2 => ZodiacSign.Pig
  | 3 => ZodiacSign.Rat
  | 4 => ZodiacSign.Ox
  | 5 => ZodiacSign.Tiger
  | 6 => ZodiacSign.Rabbit
  | 7 => ZodiacSign.Dragon
  | 8 => ZodiacSign.Snake
  | 9 => ZodiacSign.Horse
  | 10 => ZodiacSign.Goat
  | _ => ZodiacSign.Monkey

theorem olympic_year_zodiac :
  getZodiacSign 2008 = ZodiacSign.Rabbit :=
by sorry

end NUMINAMATH_CALUDE_olympic_year_zodiac_l397_39711


namespace NUMINAMATH_CALUDE_inequality_proof_l397_39741

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l397_39741


namespace NUMINAMATH_CALUDE_determinant_specific_matrix_l397_39703

theorem determinant_specific_matrix :
  let matrix : Matrix (Fin 2) (Fin 2) ℝ := !![3, Real.sin (π / 6); 5, Real.cos (π / 3)]
  Matrix.det matrix = -1 := by
  sorry

end NUMINAMATH_CALUDE_determinant_specific_matrix_l397_39703


namespace NUMINAMATH_CALUDE_three_digit_cube_sum_l397_39704

theorem three_digit_cube_sum : ∃ (n : ℕ), 
  100 ≤ n ∧ n < 1000 ∧ 
  (n = (n / 100)^3 + ((n / 10) % 10)^3 + (n % 10)^3) ∧
  n = 153 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_cube_sum_l397_39704


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l397_39778

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x² + (5 + 1/5)x - 2/5 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/5
def c : ℚ := -2/5

theorem quadratic_discriminant : discriminant a b c = 876/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l397_39778


namespace NUMINAMATH_CALUDE_jakes_weight_l397_39745

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 15 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 132) : 
  jake_weight = 93 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l397_39745


namespace NUMINAMATH_CALUDE_max_value_theorem_l397_39789

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 → x + y^2 + z^4 ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l397_39789


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_sum_l397_39772

theorem smallest_n_for_integer_sum : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    ¬∃ (j : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / m = j) ∧
  n = 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_sum_l397_39772


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_3_to_1987_l397_39786

theorem rightmost_three_digits_of_3_to_1987 : 3^1987 % 1000 = 187 := by sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_3_to_1987_l397_39786


namespace NUMINAMATH_CALUDE_divisors_odd_iff_perfect_square_l397_39766

theorem divisors_odd_iff_perfect_square (n : ℕ) : 
  Odd (Finset.card (Nat.divisors n)) ↔ ∃ m : ℕ, n = m ^ 2 := by
sorry

end NUMINAMATH_CALUDE_divisors_odd_iff_perfect_square_l397_39766


namespace NUMINAMATH_CALUDE_square_root_equality_l397_39712

theorem square_root_equality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x = a + 2 ∧ Real.sqrt x = 2*a - 5) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l397_39712


namespace NUMINAMATH_CALUDE_quiz_answer_key_count_l397_39793

/-- The number of ways to arrange true and false answers for 6 questions 
    with an equal number of true and false answers -/
def true_false_arrangements : ℕ := Nat.choose 6 3

/-- The number of ways to choose answers for 4 multiple-choice questions 
    with 5 options each -/
def multiple_choice_arrangements : ℕ := 5^4

/-- The total number of ways to create an answer key for the quiz -/
def total_arrangements : ℕ := true_false_arrangements * multiple_choice_arrangements

theorem quiz_answer_key_count : total_arrangements = 12500 := by
  sorry

end NUMINAMATH_CALUDE_quiz_answer_key_count_l397_39793


namespace NUMINAMATH_CALUDE_square_root_of_64_l397_39797

theorem square_root_of_64 : ∃ x : ℝ, x^2 = 64 ↔ x = 8 ∨ x = -8 := by sorry

end NUMINAMATH_CALUDE_square_root_of_64_l397_39797


namespace NUMINAMATH_CALUDE_angle_trig_sum_l397_39751

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and a point P on its terminal side, prove that 2sinα + cosα = -2/5 -/
theorem angle_trig_sum (α : Real) (m : Real) (h1 : m < 0) :
  let P : Prod Real Real := (-4 * m, 3 * m)
  2 * Real.sin α + Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_trig_sum_l397_39751


namespace NUMINAMATH_CALUDE_inequality_solutions_l397_39701

theorem inequality_solutions (x : ℝ) : 
  ((-x^2 + x + 6 ≤ 0) ↔ (x ≤ -2 ∨ x ≥ 3)) ∧
  ((x^2 - 2*x - 5 < 2*x) ↔ (-1 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l397_39701


namespace NUMINAMATH_CALUDE_trent_tears_per_three_onions_l397_39771

def tears_per_three_onions (pots : ℕ) (onions_per_pot : ℕ) (total_tears : ℕ) : ℚ :=
  (3 * total_tears : ℚ) / (pots * onions_per_pot : ℚ)

theorem trent_tears_per_three_onions :
  tears_per_three_onions 6 4 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trent_tears_per_three_onions_l397_39771


namespace NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l397_39788

def original_budget : ℝ := 940
def desired_reduction : ℝ := 658

theorem magazine_budget_cut_percentage :
  (original_budget - desired_reduction) / original_budget * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l397_39788


namespace NUMINAMATH_CALUDE_solution_set_of_f_neg_x_l397_39756

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

-- State the theorem
theorem solution_set_of_f_neg_x (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, f a b (-x) < 0 ↔ x < -3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_neg_x_l397_39756


namespace NUMINAMATH_CALUDE_non_collinear_triples_count_l397_39740

/-- The total number of points -/
def total_points : ℕ := 60

/-- The number of collinear triples -/
def collinear_triples : ℕ := 30

/-- The number of ways to choose three points from the total points -/
def total_triples : ℕ := total_points.choose 3

/-- The number of ways to choose three non-collinear points -/
def non_collinear_triples : ℕ := total_triples - collinear_triples

theorem non_collinear_triples_count : non_collinear_triples = 34190 := by
  sorry

end NUMINAMATH_CALUDE_non_collinear_triples_count_l397_39740


namespace NUMINAMATH_CALUDE_diana_age_is_22_l397_39718

-- Define the ages as natural numbers
def anna_age : ℕ := 48

-- Define the relationships between ages
def brianna_age : ℕ := anna_age / 2
def caitlin_age : ℕ := brianna_age - 5
def diana_age : ℕ := caitlin_age + 3

-- Theorem to prove Diana's age
theorem diana_age_is_22 : diana_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_diana_age_is_22_l397_39718


namespace NUMINAMATH_CALUDE_children_on_bus_after_stop_l397_39753

/-- The number of children on a bus after a stop -/
theorem children_on_bus_after_stop 
  (initial_children : ℕ) 
  (children_who_got_on : ℕ) 
  (h1 : initial_children = 18) 
  (h2 : children_who_got_on = 7) :
  initial_children + children_who_got_on = 25 := by
  sorry

end NUMINAMATH_CALUDE_children_on_bus_after_stop_l397_39753


namespace NUMINAMATH_CALUDE_three_digit_square_last_three_l397_39730

theorem three_digit_square_last_three (n : ℕ) : 
  (100 ≤ n ∧ n < 1000) → (n = n^2 % 1000 ↔ n = 376 ∨ n = 625) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_last_three_l397_39730


namespace NUMINAMATH_CALUDE_red_cows_produce_more_milk_l397_39749

/-- The daily milk production of a black cow -/
def black_cow_milk : ℝ := sorry

/-- The daily milk production of a red cow -/
def red_cow_milk : ℝ := sorry

/-- The total milk production of 4 black cows and 3 red cows in 5 days -/
def milk_production_1 : ℝ := 5 * (4 * black_cow_milk + 3 * red_cow_milk)

/-- The total milk production of 3 black cows and 5 red cows in 4 days -/
def milk_production_2 : ℝ := 4 * (3 * black_cow_milk + 5 * red_cow_milk)

theorem red_cows_produce_more_milk :
  milk_production_1 = milk_production_2 → red_cow_milk > black_cow_milk := by
  sorry

end NUMINAMATH_CALUDE_red_cows_produce_more_milk_l397_39749


namespace NUMINAMATH_CALUDE_smallest_sum_of_exponents_l397_39783

theorem smallest_sum_of_exponents (m n : ℕ+) (h1 : m > n) 
  (h2 : 2012^(m.val) % 1000 = 2012^(n.val) % 1000) : 
  ∃ (k l : ℕ+), k.val + l.val = 104 ∧ k > l ∧ 
  2012^(k.val) % 1000 = 2012^(l.val) % 1000 ∧
  ∀ (p q : ℕ+), p > q → 2012^(p.val) % 1000 = 2012^(q.val) % 1000 → 
  p.val + q.val ≥ 104 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_exponents_l397_39783


namespace NUMINAMATH_CALUDE_combination_permutation_relation_combination_symmetry_pascal_identity_permutation_recursive_l397_39734

-- Define C_n_m as the number of combinations of n items taken m at a time
def C (n m : ℕ) : ℕ := Nat.choose n m

-- Define A_n_m as the number of permutations of n items taken m at a time
def A (n m : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - m)

theorem combination_permutation_relation (n m : ℕ) (h : m ≤ n) :
  C n m = A n m / Nat.factorial m := by sorry

theorem combination_symmetry (n m : ℕ) (h : m ≤ n) :
  C n m = C n (n - m) := by sorry

theorem pascal_identity (n r : ℕ) (h : r ≤ n) :
  C (n + 1) r = C n r + C n (r - 1) := by sorry

theorem permutation_recursive (n m : ℕ) (h : m ≤ n) :
  A (n + 2) (m + 2) = (n + 2) * (n + 1) * A n m := by sorry

end NUMINAMATH_CALUDE_combination_permutation_relation_combination_symmetry_pascal_identity_permutation_recursive_l397_39734


namespace NUMINAMATH_CALUDE_expression_is_perfect_square_l397_39760

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Define the expression
def expression : ℕ := 3^4 * 4^6 * 7^4

-- Define the prime factorization of 4
axiom four_factorization : 4 = 2^2

-- Theorem to prove
theorem expression_is_perfect_square : is_perfect_square expression := by
  sorry

end NUMINAMATH_CALUDE_expression_is_perfect_square_l397_39760


namespace NUMINAMATH_CALUDE_comic_book_arrangement_count_comic_book_arrangement_count_is_correct_l397_39709

/-- The number of ways to arrange comic books from different publishers in a stack -/
theorem comic_book_arrangement_count : Nat :=
  let marvel_books : Nat := 8
  let dc_books : Nat := 6
  let image_books : Nat := 5
  let publisher_groups : Nat := 3

  let marvel_arrangements := Nat.factorial marvel_books
  let dc_arrangements := Nat.factorial dc_books
  let image_arrangements := Nat.factorial image_books
  let group_arrangements := Nat.factorial publisher_groups

  marvel_arrangements * dc_arrangements * image_arrangements * group_arrangements

/-- Proof that the number of arrangements is 20,901,888,000 -/
theorem comic_book_arrangement_count_is_correct : 
  comic_book_arrangement_count = 20901888000 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_arrangement_count_comic_book_arrangement_count_is_correct_l397_39709


namespace NUMINAMATH_CALUDE_sin_2theta_value_l397_39743

theorem sin_2theta_value (θ : Real) 
  (h : Real.exp (2 * Real.log 2 * ((-2 : Real) + 2 * Real.sin θ)) + 3 = 
       Real.exp (Real.log 2 * ((1 / 2 : Real) + Real.sin θ))) : 
  Real.sin (2 * θ) = 3 * Real.sqrt 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l397_39743


namespace NUMINAMATH_CALUDE_inverse_inequality_implies_reverse_l397_39790

theorem inverse_inequality_implies_reverse (a b : ℝ) :
  (1 / a < 1 / b) ∧ (1 / b < 0) → a > b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_implies_reverse_l397_39790


namespace NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l397_39706

/-- Given a total of 5x + 10 problems and x + 2 missed problems, 
    the percentage of correctly answered problems is 80%. -/
theorem lucky_lacy_correct_percentage (x : ℕ) : 
  let total := 5 * x + 10
  let missed := x + 2
  let correct := total - missed
  (correct : ℚ) / total * 100 = 80 :=
by sorry

end NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l397_39706


namespace NUMINAMATH_CALUDE_total_tickets_won_l397_39764

/-- Represents the number of tickets Dave used for toys. -/
def tickets_for_toys : ℕ := 8

/-- Represents the number of tickets Dave used for clothes. -/
def tickets_for_clothes : ℕ := 18

/-- Represents the difference in tickets used for clothes versus toys. -/
def difference_clothes_toys : ℕ := 10

/-- Theorem stating that the total number of tickets Dave won is the sum of
    tickets used for toys and clothes. -/
theorem total_tickets_won (hw : tickets_for_clothes = tickets_for_toys + difference_clothes_toys) :
  tickets_for_toys + tickets_for_clothes = 26 := by
  sorry

#check total_tickets_won

end NUMINAMATH_CALUDE_total_tickets_won_l397_39764


namespace NUMINAMATH_CALUDE_triangle_count_on_circle_l397_39727

theorem triangle_count_on_circle (n : ℕ) (h : n = 10) : 
  Nat.choose n 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_on_circle_l397_39727


namespace NUMINAMATH_CALUDE_jeds_change_l397_39798

/-- Given the conditions of Jed's board game purchase, prove the number of $5 bills received as change. -/
theorem jeds_change (num_games : ℕ) (game_cost : ℕ) (payment : ℕ) (change_bill : ℕ) : 
  num_games = 6 → 
  game_cost = 15 → 
  payment = 100 → 
  change_bill = 5 → 
  (payment - num_games * game_cost) / change_bill = 2 := by
sorry

end NUMINAMATH_CALUDE_jeds_change_l397_39798


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l397_39799

theorem divisibility_by_nine (D E : Nat) : 
  D ≤ 9 → E ≤ 9 → (D * 100000 + 864000 + E * 100 + 72) % 9 = 0 →
  (D + E = 0 ∨ D + E = 9 ∨ D + E = 18) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l397_39799


namespace NUMINAMATH_CALUDE_largest_number_l397_39762

/-- Represents a real number with a repeating decimal expansion -/
structure RepeatingDecimal where
  integerPart : ℤ
  nonRepeatingPart : List ℕ
  repeatingPart : List ℕ

/-- Convert a RepeatingDecimal to a real number -/
def toReal (d : RepeatingDecimal) : ℝ :=
  sorry

/-- Define the five numbers in the problem -/
def a : ℝ := 8.23456
def b : RepeatingDecimal := ⟨8, [2, 3, 4], [5]⟩
def c : RepeatingDecimal := ⟨8, [2, 3], [4, 5]⟩
def d : RepeatingDecimal := ⟨8, [2], [3, 4, 5]⟩
def e : RepeatingDecimal := ⟨8, [], [2, 3, 4, 5]⟩

theorem largest_number :
  toReal b > a ∧
  toReal b > toReal c ∧
  toReal b > toReal d ∧
  toReal b > toReal e :=
sorry

end NUMINAMATH_CALUDE_largest_number_l397_39762


namespace NUMINAMATH_CALUDE_regular_milk_students_l397_39759

/-- Proof that the number of students who selected regular milk is 3 -/
theorem regular_milk_students (chocolate_milk : ℕ) (strawberry_milk : ℕ) (total_milk : ℕ) :
  chocolate_milk = 2 →
  strawberry_milk = 15 →
  total_milk = 20 →
  total_milk - (chocolate_milk + strawberry_milk) = 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_milk_students_l397_39759


namespace NUMINAMATH_CALUDE_max_profit_year_l397_39777

/-- Represents the financial model of the environmentally friendly building materials factory. -/
structure FactoryFinances where
  initialInvestment : ℕ
  firstYearOperatingCosts : ℕ
  annualOperatingCostsIncrease : ℕ
  annualRevenue : ℕ

/-- Calculates the net profit for a given year. -/
def netProfitAtYear (f : FactoryFinances) (year : ℕ) : ℤ :=
  (f.annualRevenue * year : ℤ) -
  (f.initialInvestment : ℤ) -
  (f.firstYearOperatingCosts * year : ℤ) -
  (f.annualOperatingCostsIncrease * (year * (year - 1) / 2) : ℤ)

/-- Theorem stating that the net profit reaches its maximum in the 10th year. -/
theorem max_profit_year (f : FactoryFinances)
  (h1 : f.initialInvestment = 720000)
  (h2 : f.firstYearOperatingCosts = 120000)
  (h3 : f.annualOperatingCostsIncrease = 40000)
  (h4 : f.annualRevenue = 500000) :
  ∀ y : ℕ, y ≠ 10 → netProfitAtYear f y ≤ netProfitAtYear f 10 :=
sorry

end NUMINAMATH_CALUDE_max_profit_year_l397_39777


namespace NUMINAMATH_CALUDE_first_player_wins_l397_39754

/-- Represents a chessboard --/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a piece on the chessboard --/
structure Piece :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a player in the game --/
inductive Player
| First
| Second

/-- Defines the game state --/
structure GameState :=
  (board : Chessboard)
  (firstPiece : Piece)
  (secondPiece : Piece)
  (currentPlayer : Player)

/-- Defines a winning strategy for a player --/
def WinningStrategy (player : Player) (game : GameState) : Prop :=
  ∃ (strategy : GameState → ℕ × ℕ), 
    ∀ (opponent_move : ℕ × ℕ), 
      player = game.currentPlayer → 
      ∃ (next_state : GameState), 
        next_state.currentPlayer ≠ player ∧ 
        (∃ (final_state : GameState), final_state.currentPlayer = player ∧ ¬∃ (move : ℕ × ℕ), true)

/-- The main theorem stating that the first player has a winning strategy --/
theorem first_player_wins (game : GameState) : 
  game.board.rows = 3 ∧ 
  game.board.cols = 1000 ∧ 
  game.firstPiece.width = 1 ∧ 
  game.firstPiece.height = 2 ∧ 
  game.secondPiece.width = 2 ∧ 
  game.secondPiece.height = 1 ∧ 
  game.currentPlayer = Player.First → 
  WinningStrategy Player.First game :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l397_39754


namespace NUMINAMATH_CALUDE_f_derivative_l397_39796

noncomputable def f (x : ℝ) : ℝ := 2 + x * Real.cos x

theorem f_derivative : 
  deriv f = λ x => Real.cos x - x * Real.sin x :=
sorry

end NUMINAMATH_CALUDE_f_derivative_l397_39796


namespace NUMINAMATH_CALUDE_lioness_weight_l397_39776

/-- The weight of a lioness given the weights of her cubs -/
theorem lioness_weight (L F M : ℝ) : 
  L = 6 * F →  -- The weight of the lioness is six times the weight of her female cub
  L = 4 * M →  -- The weight of the lioness is four times the weight of her male cub
  M - F = 14 → -- The difference between the weights of the male and female cub is 14 kg
  L = 168 :=   -- The weight of the lioness is 168 kg
by sorry

end NUMINAMATH_CALUDE_lioness_weight_l397_39776


namespace NUMINAMATH_CALUDE_double_iced_cubes_count_l397_39737

/-- Represents a 3D coordinate in the cake --/
structure Coordinate where
  x : Nat
  y : Nat
  z : Nat

/-- The size of the cake --/
def cakeSize : Nat := 5

/-- Checks if a coordinate is on an edge with exactly two iced sides --/
def isDoubleIcedEdge (c : Coordinate) : Bool :=
  -- Top edge (front)
  (c.z = cakeSize - 1 && c.y = 0 && c.x > 0 && c.x < cakeSize - 1) ||
  -- Top edge (left)
  (c.z = cakeSize - 1 && c.x = 0 && c.y > 0 && c.y < cakeSize - 1) ||
  -- Front-left edge
  (c.x = 0 && c.y = 0 && c.z > 0 && c.z < cakeSize - 1)

/-- Counts the number of cubes with icing on exactly two sides --/
def countDoubleIcedCubes : Nat :=
  let coords := List.range cakeSize >>= fun x =>
                List.range cakeSize >>= fun y =>
                List.range cakeSize >>= fun z =>
                [{x := x, y := y, z := z}]
  (coords.filter isDoubleIcedEdge).length

/-- The main theorem to prove --/
theorem double_iced_cubes_count :
  countDoubleIcedCubes = 31 := by
  sorry


end NUMINAMATH_CALUDE_double_iced_cubes_count_l397_39737


namespace NUMINAMATH_CALUDE_pizza_fraction_l397_39746

theorem pizza_fraction (total_slices : ℕ) (whole_slices : ℕ) (shared_slice : ℚ) :
  total_slices = 16 →
  whole_slices = 2 →
  shared_slice = 1/3 →
  (whole_slices : ℚ) / total_slices + shared_slice / total_slices = 7/48 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_l397_39746


namespace NUMINAMATH_CALUDE_solve_carlas_drink_problem_l397_39721

/-- The amount of water Carla drank, given the conditions of the problem -/
def carlas_water_amount (s w : ℝ) : Prop :=
  s = 3 * w - 6 ∧ s + w = 54 → w = 15

theorem solve_carlas_drink_problem :
  ∀ s w : ℝ, carlas_water_amount s w :=
by
  sorry

end NUMINAMATH_CALUDE_solve_carlas_drink_problem_l397_39721


namespace NUMINAMATH_CALUDE_ralphs_peanuts_l397_39781

/-- Given Ralph's initial peanuts, lost peanuts, number of bags bought, and peanuts per bag,
    prove that Ralph ends up with the correct number of peanuts. -/
theorem ralphs_peanuts (initial : ℕ) (lost : ℕ) (bags : ℕ) (per_bag : ℕ)
    (h1 : initial = 2650)
    (h2 : lost = 1379)
    (h3 : bags = 4)
    (h4 : per_bag = 450) :
    initial - lost + bags * per_bag = 3071 := by
  sorry

end NUMINAMATH_CALUDE_ralphs_peanuts_l397_39781


namespace NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l397_39785

theorem polygon_sides_from_exterior_angle :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 30 →
    (n : ℝ) * exterior_angle = 360 →
    n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l397_39785


namespace NUMINAMATH_CALUDE_yahs_to_bahs_500_l397_39795

/-- Represents the exchange rate between bahs and rahs -/
def bah_to_rah_rate : ℚ := 16 / 10

/-- Represents the exchange rate between rahs and yahs -/
def rah_to_yah_rate : ℚ := 10 / 6

/-- Converts yahs to bahs -/
def yahs_to_bahs (yahs : ℚ) : ℚ :=
  yahs * (1 / rah_to_yah_rate) * (1 / bah_to_rah_rate)

theorem yahs_to_bahs_500 :
  yahs_to_bahs 500 = 187.5 := by sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_500_l397_39795


namespace NUMINAMATH_CALUDE_rainfall_sum_l397_39780

theorem rainfall_sum (monday1 wednesday1 friday monday2 wednesday2 : ℝ)
  (h1 : monday1 = 0.17)
  (h2 : wednesday1 = 0.42)
  (h3 : friday = 0.08)
  (h4 : monday2 = 0.37)
  (h5 : wednesday2 = 0.51) :
  monday1 + wednesday1 + friday + monday2 + wednesday2 = 1.55 := by
sorry

end NUMINAMATH_CALUDE_rainfall_sum_l397_39780


namespace NUMINAMATH_CALUDE_circle_line_disjoint_radius_l397_39707

-- Define the circle and line
def Circle (O : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}
def Line (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define the distance function
def distance (O : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_line_disjoint_radius (O : ℝ × ℝ) (l : Set (ℝ × ℝ)) (r : ℝ) :
  (∃ (a b c : ℝ), l = Line a b c) →
  (distance O l)^2 - (distance O l) - 20 = 0 →
  (distance O l > 0) →
  (∀ p ∈ Circle O r, p ∉ l) →
  r = 4 := by sorry

end NUMINAMATH_CALUDE_circle_line_disjoint_radius_l397_39707


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l397_39775

/-- Given two quadratic equations, where the roots of one are three less than the roots of the other,
    prove that the constant term of the second equation is zero. -/
theorem quadratic_root_relation (b c : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 8 * r + 6 = 0 ∧ 2 * s^2 - 8 * s + 6 = 0) →
  (∃ x y : ℝ, x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0) →
  (∀ r s x y : ℝ, 
    (2 * r^2 - 8 * r + 6 = 0 ∧ 2 * s^2 - 8 * s + 6 = 0) →
    (x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0) →
    x = r - 3 ∧ y = s - 3) →
  c = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l397_39775


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l397_39791

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  α ≠ β →
  perpendicular m α →
  parallel_lines m n →
  parallel_line_plane n β →
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l397_39791


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l397_39739

/-- Given ratios between w, x, y, and z, prove the ratio of w to y -/
theorem ratio_w_to_y 
  (h_wx : (w : ℚ) / x = 5 / 2)
  (h_yz : (y : ℚ) / z = 4 / 1)
  (h_zx : (z : ℚ) / x = 2 / 5) :
  w / y = 25 / 16 :=
by sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l397_39739


namespace NUMINAMATH_CALUDE_sqrt_sum_zero_implies_power_l397_39768

theorem sqrt_sum_zero_implies_power (a b : ℝ) : 
  Real.sqrt (a + 3) + Real.sqrt (2 - b) = 0 → a^b = 9 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_zero_implies_power_l397_39768


namespace NUMINAMATH_CALUDE_g_inv_f_10_l397_39716

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- Assume f and g are bijective
variable (hf : Function.Bijective f)
variable (hg : Function.Bijective g)

-- Define the relationship between f and g
axiom fg_relation : ∀ x, f_inv (g x) = 3 * x - 1

-- Define the inverse functions
axiom f_inverse : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f
axiom g_inverse : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g

-- State the theorem
theorem g_inv_f_10 : g_inv (f 10) = 11 / 3 := by sorry

end NUMINAMATH_CALUDE_g_inv_f_10_l397_39716


namespace NUMINAMATH_CALUDE_hanks_pancakes_l397_39731

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- The number of customers who ordered short stack pancakes -/
def short_stack_orders : ℕ := 9

/-- The number of customers who ordered big stack pancakes -/
def big_stack_orders : ℕ := 6

/-- The total number of pancakes Hank needs to make -/
def total_pancakes : ℕ := short_stack_orders * short_stack + big_stack_orders * big_stack

theorem hanks_pancakes : total_pancakes = 57 := by
  sorry

end NUMINAMATH_CALUDE_hanks_pancakes_l397_39731


namespace NUMINAMATH_CALUDE_square_perimeter_l397_39705

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) :
  area = 675 →
  side * side = area →
  perimeter = 4 * side →
  1.5 * perimeter = 90 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l397_39705


namespace NUMINAMATH_CALUDE_new_people_weight_sum_l397_39747

/-- Given a group of 8 people with average weight W kg, prove that when two people weighing 68 kg each
    leave and are replaced by two new people, causing the average weight to increase by 5.5 kg,
    the sum of the weights of the two new people is 180 kg. -/
theorem new_people_weight_sum (W : ℝ) : 
  let original_total := 8 * W
  let remaining_total := original_total - 2 * 68
  let new_total := 8 * (W + 5.5)
  new_total - remaining_total = 180 := by
  sorry

/-- The sum of the weights of the two new people is no more than 180 kg. -/
axiom new_people_weight_bound (x y : ℝ) : x + y ≤ 180

/-- Each of the new people weighs more than the original average weight. -/
axiom new_people_weight_lower_bound (x y : ℝ) (W : ℝ) : x > W ∧ y > W

end NUMINAMATH_CALUDE_new_people_weight_sum_l397_39747


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l397_39758

/-- Represents the sum of elements in the nth set of a specific sequence of sets of consecutive integers -/
def S (n : ℕ) : ℕ :=
  let first := 1 + (n * (n - 1)) / 2
  let last := first + n - 1
  n * (first + last) / 2

/-- The theorem stating that the sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l397_39758


namespace NUMINAMATH_CALUDE_exists_solution_for_prime_l397_39736

theorem exists_solution_for_prime (p : ℕ) (hp : Prime p) :
  ∃ (x y z w : ℤ), x^2 + y^2 + z^2 - w * ↑p = 0 ∧ 0 < w ∧ w < ↑p :=
by sorry

end NUMINAMATH_CALUDE_exists_solution_for_prime_l397_39736


namespace NUMINAMATH_CALUDE_parabola_shift_l397_39757

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift amount
def shift : ℝ := 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x - shift)^2

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - shift) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l397_39757


namespace NUMINAMATH_CALUDE_olympic_arrangements_correct_l397_39723

/-- The number of ways to arrange athletes in Olympic lanes -/
def olympicArrangements : ℕ := 2520

/-- The number of lanes -/
def numLanes : ℕ := 8

/-- The number of countries -/
def numCountries : ℕ := 4

/-- The number of athletes per country -/
def athletesPerCountry : ℕ := 2

/-- Theorem: The number of ways to arrange the athletes is correct -/
theorem olympic_arrangements_correct :
  olympicArrangements = (numLanes.choose athletesPerCountry) *
                        ((numLanes - athletesPerCountry).choose athletesPerCountry) *
                        ((numLanes - 2 * athletesPerCountry).choose athletesPerCountry) *
                        ((numLanes - 3 * athletesPerCountry).choose athletesPerCountry) :=
by sorry

end NUMINAMATH_CALUDE_olympic_arrangements_correct_l397_39723


namespace NUMINAMATH_CALUDE_largest_common_divisor_l397_39732

theorem largest_common_divisor : ∃ (n : ℕ), n = 60 ∧ 
  n ∣ 660 ∧ n < 100 ∧ n ∣ 120 ∧ 
  ∀ (m : ℕ), m ∣ 660 ∧ m < 100 ∧ m ∣ 120 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l397_39732


namespace NUMINAMATH_CALUDE_intersection_A_B_l397_39744

-- Define set A
def A : Set ℝ := {x | |x| < 1}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l397_39744


namespace NUMINAMATH_CALUDE_incorrect_expression_l397_39761

/-- A repeating decimal with non-repeating part X and repeating part Y -/
structure RepeatingDecimal where
  X : ℕ  -- non-repeating part
  Y : ℕ  -- repeating part
  t : ℕ  -- number of digits in X
  u : ℕ  -- number of digits in Y

/-- The value of a repeating decimal -/
def value (E : RepeatingDecimal) : ℚ :=
  sorry

/-- The statement that the expression is incorrect -/
theorem incorrect_expression (E : RepeatingDecimal) :
  ¬(10^E.t * (10^E.u - 1) * value E = E.Y * (E.X - 10)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_expression_l397_39761


namespace NUMINAMATH_CALUDE_leahs_coins_value_l397_39765

theorem leahs_coins_value (n p : ℕ) : 
  n + p = 13 →                   -- Total number of coins is 13
  n + 1 = p →                    -- One more nickel would equal pennies
  5 * n + p = 37                 -- Total value in cents
  := by sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l397_39765


namespace NUMINAMATH_CALUDE_expression_value_l397_39700

theorem expression_value (a : ℚ) (h : a = 1/3) : 
  (3 * a⁻¹ + (2 * a⁻¹) / 3) / a = 33 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l397_39700


namespace NUMINAMATH_CALUDE_additional_amount_for_free_shipping_l397_39770

-- Define the book prices and discount
def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00
def discount_rate : ℝ := 0.25
def free_shipping_threshold : ℝ := 50.00

-- Calculate the discounted prices for books 1 and 2
def discounted_book1_price : ℝ := book1_price * (1 - discount_rate)
def discounted_book2_price : ℝ := book2_price * (1 - discount_rate)

-- Calculate the total cost of all four books
def total_cost : ℝ := discounted_book1_price + discounted_book2_price + book3_price + book4_price

-- Define the additional amount needed for free shipping
def additional_amount : ℝ := free_shipping_threshold - total_cost

-- Theorem to prove
theorem additional_amount_for_free_shipping :
  additional_amount = 9.00 := by sorry

end NUMINAMATH_CALUDE_additional_amount_for_free_shipping_l397_39770


namespace NUMINAMATH_CALUDE_hannah_reading_finish_day_l397_39725

def days_to_read (n : ℕ) : ℕ := n * (n + 1) / 2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem hannah_reading_finish_day (start_day : ℕ) (num_books : ℕ) :
  start_day = 5 →  -- Friday is represented as 5 (0 = Sunday, 1 = Monday, etc.)
  num_books = 20 →
  day_of_week start_day (days_to_read num_books) = start_day :=
by sorry

end NUMINAMATH_CALUDE_hannah_reading_finish_day_l397_39725


namespace NUMINAMATH_CALUDE_system_solution_unique_l397_39752

theorem system_solution_unique :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧
    x^4 + y^4 - x^2*y^2 = 13 ∧
    x^2 - y^2 + 2*x*y = 1 ∧
    x = 1 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l397_39752


namespace NUMINAMATH_CALUDE_plot_perimeter_is_220_l397_39755

/-- Represents a rectangular plot with the given conditions -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ
  lengthWidthRelation : length = width + 10
  fencingCostRelation : fencingCostPerMeter * (2 * (length + width)) = totalFencingCost

/-- The perimeter of the rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.width)

/-- Theorem stating that the perimeter of the plot is 220 meters -/
theorem plot_perimeter_is_220 (plot : RectangularPlot) 
    (h1 : plot.fencingCostPerMeter = 6.5)
    (h2 : plot.totalFencingCost = 1430) : 
  perimeter plot = 220 := by
  sorry

end NUMINAMATH_CALUDE_plot_perimeter_is_220_l397_39755


namespace NUMINAMATH_CALUDE_solve_flower_problem_l397_39750

def flower_problem (minyoung_flowers : ℕ) (ratio : ℕ) : Prop :=
  let yoojung_flowers := minyoung_flowers / ratio
  minyoung_flowers + yoojung_flowers = 30

theorem solve_flower_problem :
  flower_problem 24 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_flower_problem_l397_39750


namespace NUMINAMATH_CALUDE_f_has_six_zeros_l397_39708

noncomputable def f (x : ℝ) : ℝ :=
  (1 + x - x^2/2 + x^3/3 - x^4/4 - x^2018/2018 + x^2019/2019) * Real.cos (2*x)

theorem f_has_six_zeros :
  ∃ (S : Finset ℝ), S.card = 6 ∧ 
  (∀ x ∈ S, x ∈ Set.Icc (-3) 4 ∧ f x = 0) ∧
  (∀ x ∈ Set.Icc (-3) 4, f x = 0 → x ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_f_has_six_zeros_l397_39708


namespace NUMINAMATH_CALUDE_roboducks_order_l397_39782

theorem roboducks_order (shelves_percentage : ℚ) (storage_count : ℕ) : 
  shelves_percentage = 30 / 100 →
  storage_count = 140 →
  ∃ total : ℕ, total = 200 ∧ (1 - shelves_percentage) * total = storage_count :=
by
  sorry

end NUMINAMATH_CALUDE_roboducks_order_l397_39782


namespace NUMINAMATH_CALUDE_min_value_of_z_l397_39724

/-- Given a set of constraints on x and y, prove that the minimum value of z = 3x - 4y is -1 -/
theorem min_value_of_z (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y - 2 ≤ 0) (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3 * x - 4 * y ∧ z ≥ -1 ∧ ∀ (w : ℝ), w = 3 * x - 4 * y → w ≥ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l397_39724
