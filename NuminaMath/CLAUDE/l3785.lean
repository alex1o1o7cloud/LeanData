import Mathlib

namespace NUMINAMATH_CALUDE_cameron_typing_difference_l3785_378566

theorem cameron_typing_difference (
  speed_before : ℕ) 
  (speed_after : ℕ) 
  (time : ℕ) 
  (h1 : speed_before = 10) 
  (h2 : speed_after = 8) 
  (h3 : time = 5) : 
  speed_before * time - speed_after * time = 10 :=
by sorry

end NUMINAMATH_CALUDE_cameron_typing_difference_l3785_378566


namespace NUMINAMATH_CALUDE_max_abs_z_l3785_378535

theorem max_abs_z (z : ℂ) (h : Complex.abs (z + 3 + 4 * I) ≤ 2) :
  ∃ (max_val : ℝ), max_val = 7 ∧ ∀ w : ℂ, Complex.abs (w + 3 + 4 * I) ≤ 2 → Complex.abs w ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_l3785_378535


namespace NUMINAMATH_CALUDE_right_triangle_area_l3785_378504

theorem right_triangle_area (a b c : ℝ) (ha : a^2 = 100) (hb : b^2 = 64) (hc : c^2 = 121)
  (h_right : a^2 + b^2 = c^2) : (1/2) * a * b = 40 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3785_378504


namespace NUMINAMATH_CALUDE_find_divisor_l3785_378578

theorem find_divisor (divisor : ℕ) : 
  (∃ k : ℕ, (228712 + 5) = divisor * k) ∧ 
  (∀ n < 5, ¬∃ m : ℕ, (228712 + n) = divisor * m) →
  divisor = 3 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3785_378578


namespace NUMINAMATH_CALUDE_existence_of_a_value_of_a_l3785_378543

-- Define the sets A, B, and C as functions of real numbers
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + 4*a^2 - 3 = 0}
def B : Set ℝ := {x | x^2 - x - 2 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem 1
theorem existence_of_a : ∃ a : ℝ, A a = B ∧ a = 1/2 := by sorry

-- Theorem 2
theorem value_of_a (a : ℝ) : (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -1 := by sorry

end NUMINAMATH_CALUDE_existence_of_a_value_of_a_l3785_378543


namespace NUMINAMATH_CALUDE_complex_binomial_sum_zero_l3785_378558

theorem complex_binomial_sum_zero :
  let x : ℂ := 2 * Complex.I / (1 - Complex.I)
  let n : ℕ := 2016
  (Finset.sum (Finset.range n) (fun k => Nat.choose n (k + 1) * x ^ (k + 1))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_binomial_sum_zero_l3785_378558


namespace NUMINAMATH_CALUDE_grape_juice_in_drink_l3785_378569

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ

/-- Calculates the amount of grape juice in the drink -/
def grape_juice_amount (drink : FruitDrink) : ℝ :=
  drink.total * (1 - drink.orange_percent - drink.watermelon_percent)

/-- Theorem stating the amount of grape juice in the specific drink -/
theorem grape_juice_in_drink : 
  let drink : FruitDrink := { total := 150, orange_percent := 0.35, watermelon_percent := 0.35 }
  grape_juice_amount drink = 45 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_in_drink_l3785_378569


namespace NUMINAMATH_CALUDE_percentage_of_part_to_whole_l3785_378574

theorem percentage_of_part_to_whole (total : ℝ) (part : ℝ) : 
  total > 0 → part ≥ 0 → part ≤ total → (part / total) * 100 = 25 → total = 400 ∧ part = 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_part_to_whole_l3785_378574


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l3785_378542

theorem ellipse_eccentricity_range (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (x y : ℝ), 
    (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
    ((x + c)^2 + y^2) * ((x - c)^2 + y^2) = (2*c^2)^2 →
    (1/2 : ℝ) ≤ c/a ∧ c/a ≤ (Real.sqrt 3)/3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l3785_378542


namespace NUMINAMATH_CALUDE_vicente_spent_25_l3785_378573

/-- The total amount Vicente spent on rice and meat -/
def total_spent (rice_kg : ℕ) (rice_price : ℚ) (meat_lb : ℕ) (meat_price : ℚ) : ℚ :=
  (rice_kg : ℚ) * rice_price + (meat_lb : ℚ) * meat_price

/-- Proof that Vicente spent $25 on his purchase -/
theorem vicente_spent_25 :
  total_spent 5 2 3 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_vicente_spent_25_l3785_378573


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l3785_378531

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Definition of the first line: y = -3x -/
def line1 (x y : ℚ) : Prop := y = -3 * x

/-- Definition of the second line: y - 3 = 9x -/
def line2 (x y : ℚ) : Prop := y - 3 = 9 * x

/-- Theorem stating that (-1/4, 3/4) is the unique intersection point of the two lines -/
theorem intersection_point_of_lines :
  ∃! p : IntersectionPoint, line1 p.x p.y ∧ line2 p.x p.y ∧ p.x = -1/4 ∧ p.y = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l3785_378531


namespace NUMINAMATH_CALUDE_count_four_digit_integers_eq_sixteen_l3785_378506

/-- The number of four-digit positive integers composed only of digits 2 and 5 -/
def count_four_digit_integers : ℕ :=
  let digit_choices := 2  -- number of choices for each digit (2 or 5)
  let num_digits := 4     -- number of digits in the integer
  digit_choices ^ num_digits

/-- Theorem stating that the count of four-digit positive integers
    composed only of digits 2 and 5 is equal to 16 -/
theorem count_four_digit_integers_eq_sixteen :
  count_four_digit_integers = 16 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_integers_eq_sixteen_l3785_378506


namespace NUMINAMATH_CALUDE_salesman_pear_sales_l3785_378598

theorem salesman_pear_sales (morning_sales afternoon_sales total_sales : ℕ) :
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 510 →
  afternoon_sales = 340 :=
by
  sorry

end NUMINAMATH_CALUDE_salesman_pear_sales_l3785_378598


namespace NUMINAMATH_CALUDE_vectors_form_basis_l3785_378588

def vector_a : Fin 2 → ℝ := λ i => if i = 0 then 2 else -3
def vector_b : Fin 2 → ℝ := λ i => if i = 0 then 6 else 9

def is_basis (v w : Fin 2 → ℝ) : Prop :=
  LinearIndependent ℝ (![v, w]) ∧ Submodule.span ℝ {v, w} = ⊤

theorem vectors_form_basis : is_basis vector_a vector_b := by sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l3785_378588


namespace NUMINAMATH_CALUDE_john_max_books_l3785_378540

def john_money : ℕ := 2545  -- in cents
def initial_book_price : ℕ := 285  -- in cents
def discounted_book_price : ℕ := 250  -- in cents
def discount_threshold : ℕ := 10

def max_books_buyable (money : ℕ) (price : ℕ) (discount_price : ℕ) (threshold : ℕ) : ℕ :=
  if money < threshold * price then
    money / price
  else
    threshold + (money - threshold * price) / discount_price

theorem john_max_books :
  max_books_buyable john_money initial_book_price discounted_book_price discount_threshold = 8 :=
sorry

end NUMINAMATH_CALUDE_john_max_books_l3785_378540


namespace NUMINAMATH_CALUDE_negation_of_conditional_l3785_378536

theorem negation_of_conditional (x : ℝ) :
  ¬(x > 1 → x^2 > x) ↔ (x ≤ 1 → x^2 ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l3785_378536


namespace NUMINAMATH_CALUDE_probability_of_blue_ball_l3785_378599

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of blue balls
def blue_balls : ℕ := 6

-- Define the probability of drawing a blue ball
def prob_blue_ball : ℚ := blue_balls / total_balls

-- Theorem statement
theorem probability_of_blue_ball : prob_blue_ball = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_blue_ball_l3785_378599


namespace NUMINAMATH_CALUDE_sum_of_roots_l3785_378538

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a = 1) 
  (hb : b^3 - 3*b^2 + 5*b = 5) : 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3785_378538


namespace NUMINAMATH_CALUDE_integer_ratio_problem_l3785_378544

theorem integer_ratio_problem (a b : ℤ) : 
  1996 * a + b / 96 = a + b → b / a = 2016 ∨ a / b = 1 / 2016 := by
  sorry

end NUMINAMATH_CALUDE_integer_ratio_problem_l3785_378544


namespace NUMINAMATH_CALUDE_sandy_painting_area_l3785_378591

/-- The area Sandy needs to paint on her bedroom wall -/
def area_to_paint (wall_height wall_length bookshelf_width bookshelf_height : ℝ) : ℝ :=
  wall_height * wall_length - bookshelf_width * bookshelf_height

/-- Theorem stating that Sandy needs to paint 135 square feet -/
theorem sandy_painting_area :
  area_to_paint 10 15 3 5 = 135 := by
  sorry

#eval area_to_paint 10 15 3 5

end NUMINAMATH_CALUDE_sandy_painting_area_l3785_378591


namespace NUMINAMATH_CALUDE_min_marked_points_l3785_378579

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A configuration of points in a plane -/
structure PointConfiguration where
  points : Finset Point
  unique_distances : ∀ p q r s : Point, p ∈ points → q ∈ points → r ∈ points → s ∈ points →
    p ≠ q → r ≠ s → (p.x - q.x)^2 + (p.y - q.y)^2 ≠ (r.x - s.x)^2 + (r.y - s.y)^2

/-- The set of points marked as closest to at least one other point -/
def marked_points (config : PointConfiguration) : Finset Point :=
  sorry

/-- The theorem stating the minimum number of marked points -/
theorem min_marked_points (config : PointConfiguration) :
  config.points.card = 2018 →
  (marked_points config).card ≥ 404 :=
sorry

end NUMINAMATH_CALUDE_min_marked_points_l3785_378579


namespace NUMINAMATH_CALUDE_books_read_in_common_l3785_378568

theorem books_read_in_common (tony_books dean_books breanna_books total_books : ℕ) 
  (h1 : tony_books = 23)
  (h2 : dean_books = 12)
  (h3 : breanna_books = 17)
  (h4 : total_books = 47)
  (h5 : ∃ (common : ℕ), common > 0 ∧ common ≤ min tony_books dean_books)
  (h6 : ∃ (all_common : ℕ), all_common > 0 ∧ all_common ≤ min tony_books (min dean_books breanna_books)) :
  ∃ (x : ℕ), x = 3 ∧ 
    tony_books + dean_books + breanna_books - x - 1 = total_books :=
by sorry

end NUMINAMATH_CALUDE_books_read_in_common_l3785_378568


namespace NUMINAMATH_CALUDE_last_three_average_l3785_378582

theorem last_three_average (numbers : List ℝ) : 
  numbers.length = 5 →
  numbers.sum / numbers.length = 54 →
  (numbers.take 2).sum / 2 = 48 →
  (numbers.drop 2).sum / 3 = 58 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l3785_378582


namespace NUMINAMATH_CALUDE_triangular_prism_float_l3785_378501

theorem triangular_prism_float (x : ℝ) : ¬ (0 < x ∧ x < Real.sqrt 3 / 2 ∧ (Real.sqrt 3 / 4) * x = x * (1 - x / Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_float_l3785_378501


namespace NUMINAMATH_CALUDE_consecutive_even_sum_l3785_378584

theorem consecutive_even_sum (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4) →  -- a, b, c are consecutive even integers
  (a + b + c = 246) →                              -- their sum is 246
  (c = 84) :=                                      -- the third number is 84
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_l3785_378584


namespace NUMINAMATH_CALUDE_determinant_evaluation_l3785_378555

theorem determinant_evaluation (x : ℝ) : 
  Matrix.det !![x + 2, x - 1, x; x - 1, x + 2, x; x, x, x + 3] = 14 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_determinant_evaluation_l3785_378555


namespace NUMINAMATH_CALUDE_twentieth_term_is_negative_49_l3785_378513

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  firstTerm : ℤ
  commonDiff : ℤ

/-- The nth term of an arithmetic sequence. -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.firstTerm + (n - 1 : ℤ) * seq.commonDiff

/-- The theorem stating that the 20th term of the specific arithmetic sequence is -49. -/
theorem twentieth_term_is_negative_49 :
  let seq := ArithmeticSequence.mk 8 (-3)
  nthTerm seq 20 = -49 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_negative_49_l3785_378513


namespace NUMINAMATH_CALUDE_triangle_problem_l3785_378556

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (2*b - c) * Real.cos A - a * Real.cos C = 0 →
  a = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ b = 2 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3785_378556


namespace NUMINAMATH_CALUDE_cos_negative_300_degrees_l3785_378580

theorem cos_negative_300_degrees : Real.cos (-300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_300_degrees_l3785_378580


namespace NUMINAMATH_CALUDE_john_arcade_spend_l3785_378550

/-- The amount of money John spent at the arcade -/
def arcade_spend (total_time minutes_per_break num_breaks cost_per_interval minutes_per_interval : ℕ) : ℚ :=
  let total_minutes := total_time
  let break_minutes := minutes_per_break * num_breaks
  let playing_minutes := total_minutes - break_minutes
  let num_intervals := playing_minutes / minutes_per_interval
  (num_intervals : ℚ) * cost_per_interval

theorem john_arcade_spend :
  arcade_spend 275 10 5 (3/4) 5 = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_john_arcade_spend_l3785_378550


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3785_378581

theorem arithmetic_expression_evaluation : 2 - (3 - 4) - (5 - 6 - 7) = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3785_378581


namespace NUMINAMATH_CALUDE_female_democrats_count_l3785_378592

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 840 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 140 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l3785_378592


namespace NUMINAMATH_CALUDE_intersection_of_P_and_M_l3785_378597

def P : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x : ℝ | |x| ≤ 3}

theorem intersection_of_P_and_M : P ∩ M = {x : ℝ | 0 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_M_l3785_378597


namespace NUMINAMATH_CALUDE_dining_bill_share_l3785_378517

theorem dining_bill_share (total_bill : ℝ) (num_people : ℕ) (tip_percentage : ℝ) :
  total_bill = 139 ∧ num_people = 5 ∧ tip_percentage = 0.1 →
  (total_bill * (1 + tip_percentage)) / num_people = 30.58 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_share_l3785_378517


namespace NUMINAMATH_CALUDE_final_result_proof_l3785_378534

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 1376) :
  (chosen_number / 8 : ℚ) - 160 = 12 := by
  sorry

end NUMINAMATH_CALUDE_final_result_proof_l3785_378534


namespace NUMINAMATH_CALUDE_fraction_inequality_l3785_378587

theorem fraction_inequality (x y z a b c r : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) :
  (x + y + a + b) / (x + y + a + b + c + r) + (y + z + b + c) / (y + z + a + b + c + r) >
  (x + z + a + c) / (x + z + a + b + c + r) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3785_378587


namespace NUMINAMATH_CALUDE_floor_length_proof_l3785_378526

/-- Given two rectangular floors X and Y with equal areas, 
    where X is 10 feet by 18 feet and Y is 9 feet wide, 
    prove that the length of floor Y is 20 feet. -/
theorem floor_length_proof (area_x area_y length_x width_x width_y : ℝ) : 
  area_x = area_y → 
  length_x = 10 → 
  width_x = 18 → 
  width_y = 9 → 
  area_x = length_x * width_x → 
  area_y = width_y * (area_y / width_y) → 
  area_y / width_y = 20 := by
  sorry

#check floor_length_proof

end NUMINAMATH_CALUDE_floor_length_proof_l3785_378526


namespace NUMINAMATH_CALUDE_range_of_a_l3785_378508

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x - 7 else Real.sqrt x

-- State the theorem
theorem range_of_a (a : ℝ) (h : f a < 1) : -3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3785_378508


namespace NUMINAMATH_CALUDE_zeros_of_f_range_of_m_l3785_378521

-- Define the function f
def f (a x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

-- Theorem for part (I)
theorem zeros_of_f'_depend_on_a (a : ℝ) :
  ∃ n : ℕ, n ∈ ({1, 2} : Set ℕ) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-1) 3 ∧ x₂ ∈ Set.Icc (-1) 3 ∧ 
   f' a x₁ = 0 ∧ f' a x₂ = 0 ∧ 
   ∀ x ∈ Set.Icc (-1) 3, f' a x = 0 → x = x₁ ∨ x = x₂) :=
sorry

-- Theorem for part (II)
theorem range_of_m (a : ℝ) (h : a ∈ Set.Icc (-3) 0) :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → 
    m - a * m^2 ≥ |f a x₁ - f a x₂|) → 
  m ∈ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_range_of_m_l3785_378521


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l3785_378583

theorem rectangle_longer_side (a : ℝ) (h1 : a > 0) : 
  (a * (0.8 * a) = 81/20) → a = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l3785_378583


namespace NUMINAMATH_CALUDE_solve_for_y_l3785_378562

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3785_378562


namespace NUMINAMATH_CALUDE_coordinate_sum_of_A_l3785_378524

-- Define the points
def B : ℝ × ℝ := (2, 8)
def C : ℝ × ℝ := (0, 2)

-- Define the theorem
theorem coordinate_sum_of_A (A : ℝ × ℝ) :
  (A.1 - C.1) / (B.1 - C.1) = 1/3 ∧
  (A.2 - C.2) / (B.2 - C.2) = 1/3 →
  A.1 + A.2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_A_l3785_378524


namespace NUMINAMATH_CALUDE_probability_heart_spade_club_standard_deck_l3785_378545

/-- A standard deck of cards. -/
structure Deck :=
  (total : Nat)
  (hearts : Nat)
  (spades : Nat)
  (clubs : Nat)
  (diamonds : Nat)

/-- The probability of drawing a heart, then a spade, then a club from a standard deck. -/
def probability_heart_spade_club (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.total *
  (d.spades : ℚ) / (d.total - 1) *
  (d.clubs : ℚ) / (d.total - 2)

/-- Theorem stating the probability of drawing a heart, then a spade, then a club
    from a standard 52-card deck. -/
theorem probability_heart_spade_club_standard_deck :
  let standard_deck : Deck := ⟨52, 13, 13, 13, 13⟩
  probability_heart_spade_club standard_deck = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_spade_club_standard_deck_l3785_378545


namespace NUMINAMATH_CALUDE_circle_relationship_l3785_378520

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Theorem about the relationship between two circles -/
theorem circle_relationship (R₁ R₂ d : ℝ) (c₁ c₂ : Circle) 
  (h₁ : c₁.radius = R₁)
  (h₂ : c₂.radius = R₂)
  (h₃ : R₁ ≠ R₂)
  (h₄ : ∃ x : ℝ, x^2 - 2*R₁*x + R₂^2 - d*(R₂ - R₁) = 0 ∧ 
        ∀ y : ℝ, y^2 - 2*R₁*y + R₂^2 - d*(R₂ - R₁) = 0 → y = x) :
  R₁ + R₂ = d ∧ (∀ p : ℝ × ℝ, ‖p‖ ≠ R₁ ∨ ‖p - (d, 0)‖ ≠ R₂) := by
sorry

end NUMINAMATH_CALUDE_circle_relationship_l3785_378520


namespace NUMINAMATH_CALUDE_power_function_unique_m_l3785_378515

/-- A function f: ℝ → ℝ is increasing on (0, +∞) if for all x₁, x₂ ∈ (0, +∞),
    x₁ < x₂ implies f(x₁) < f(x₂) -/
def IncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f x₁ < f x₂

/-- A function f: ℝ → ℝ is a power function if there exist constants a and b
    such that f(x) = a * x^b for all x > 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, x > 0 → f x = a * x^b

theorem power_function_unique_m :
  ∃! m : ℝ, IsPowerFunction (fun x ↦ (m^2 - m - 1) * x^(m^2 - 3*m - 3)) ∧
            IncreasingOn (fun x ↦ (m^2 - m - 1) * x^(m^2 - 3*m - 3)) ∧
            m = -1 :=
sorry

end NUMINAMATH_CALUDE_power_function_unique_m_l3785_378515


namespace NUMINAMATH_CALUDE_cafeteria_choices_l3785_378507

theorem cafeteria_choices (num_dishes : ℕ) (num_students : ℕ) : 
  num_dishes = 5 → num_students = 3 → (num_dishes ^ num_students) = 125 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_choices_l3785_378507


namespace NUMINAMATH_CALUDE_rhombus_diagonals_l3785_378577

theorem rhombus_diagonals (area : ℝ) (perimeter : ℝ) (d1 d2 : ℝ) : 
  area = 117 →
  perimeter = 31 →
  area = (1/2) * d1 * d2 →
  perimeter = 2 * ((1/2) * d1 + (1/2) * d2) →
  (d1 = 18 ∧ d2 = 13) ∨ (d1 = 13 ∧ d2 = 18) := by
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_l3785_378577


namespace NUMINAMATH_CALUDE_son_work_time_l3785_378571

-- Define the work rates
def man_rate : ℚ := 1 / 6
def combined_rate : ℚ := 1 / 3

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem to prove
theorem son_work_time : (1 : ℚ) / son_rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l3785_378571


namespace NUMINAMATH_CALUDE_wednesday_bags_is_nine_l3785_378594

/-- Represents the leaf raking business scenario -/
structure LeafRakingBusiness where
  charge_per_bag : ℕ
  monday_bags : ℕ
  tuesday_bags : ℕ
  total_earnings : ℕ

/-- Calculates the number of bags raked on Wednesday -/
def bags_on_wednesday (business : LeafRakingBusiness) : ℕ :=
  (business.total_earnings - business.charge_per_bag * (business.monday_bags + business.tuesday_bags)) / business.charge_per_bag

/-- Theorem stating that the number of bags raked on Wednesday is 9 -/
theorem wednesday_bags_is_nine (business : LeafRakingBusiness)
  (h1 : business.charge_per_bag = 4)
  (h2 : business.monday_bags = 5)
  (h3 : business.tuesday_bags = 3)
  (h4 : business.total_earnings = 68) :
  bags_on_wednesday business = 9 := by
  sorry

#eval bags_on_wednesday { charge_per_bag := 4, monday_bags := 5, tuesday_bags := 3, total_earnings := 68 }

end NUMINAMATH_CALUDE_wednesday_bags_is_nine_l3785_378594


namespace NUMINAMATH_CALUDE_average_english_score_of_dropped_students_l3785_378548

/-- Represents the problem of calculating the average English quiz score of dropped students -/
theorem average_english_score_of_dropped_students
  (total_students : ℕ)
  (remaining_students : ℕ)
  (initial_average : ℝ)
  (new_average : ℝ)
  (h1 : total_students = 16)
  (h2 : remaining_students = 13)
  (h3 : initial_average = 62.5)
  (h4 : new_average = 62.0) :
  let dropped_students := total_students - remaining_students
  let total_score := total_students * initial_average
  let remaining_score := remaining_students * new_average
  let dropped_score := total_score - remaining_score
  abs ((dropped_score / dropped_students) - 64.67) < 0.01 := by
  sorry

#check average_english_score_of_dropped_students

end NUMINAMATH_CALUDE_average_english_score_of_dropped_students_l3785_378548


namespace NUMINAMATH_CALUDE_parabola_shift_l3785_378576

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x + 3)^2 + 2

-- Theorem stating that the shifted parabola is correct
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 3) + 2 :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_shift_l3785_378576


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l3785_378593

theorem consecutive_odd_squares_difference (n : ℕ) : 
  ∃ k : ℤ, (2*n + 1)^2 - (2*n - 1)^2 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l3785_378593


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l3785_378511

theorem same_color_plate_probability : 
  let total_plates : ℕ := 7 + 5
  let red_plates : ℕ := 7
  let blue_plates : ℕ := 5
  let total_combinations : ℕ := Nat.choose total_plates 3
  let red_combinations : ℕ := Nat.choose red_plates 3
  let blue_combinations : ℕ := Nat.choose blue_plates 3
  let same_color_combinations : ℕ := red_combinations + blue_combinations
  (same_color_combinations : ℚ) / total_combinations = 9 / 44 := by
sorry

end NUMINAMATH_CALUDE_same_color_plate_probability_l3785_378511


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3785_378549

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_slope_at_one
  (h1 : Differentiable ℝ f)
  (h2 : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f 1 - f (1 - 2*Δx)) / (2*Δx) + 1| < ε) :
  deriv f 1 = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3785_378549


namespace NUMINAMATH_CALUDE_milk_cost_per_liter_l3785_378546

/-- Represents the milkman's milk mixture problem -/
def MilkProblem (total_milk pure_milk water_added mixture_price profit : ℝ) : Prop :=
  total_milk = 30 ∧
  pure_milk = 20 ∧
  water_added = 5 ∧
  (pure_milk + water_added) * mixture_price - pure_milk * mixture_price = profit ∧
  profit = 35

/-- The cost of pure milk per liter is 7 rupees -/
theorem milk_cost_per_liter (total_milk pure_milk water_added mixture_price profit : ℝ) 
  (h : MilkProblem total_milk pure_milk water_added mixture_price profit) : 
  mixture_price = 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_cost_per_liter_l3785_378546


namespace NUMINAMATH_CALUDE_solution_is_ray_iff_a_is_pm1_l3785_378518

/-- The polynomial function in x parameterized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - (a^2 + a + 1)*x^2 + (a^3 + a^2 + a)*x - a^3

/-- The set of solutions for the inequality -/
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x ≥ 0}

/-- Definition of a ray (half-line) in ℝ -/
def is_ray (S : Set ℝ) : Prop :=
  ∃ (x₀ : ℝ), S = {x : ℝ | x ≥ x₀} ∨ S = {x : ℝ | x ≤ x₀}

/-- The main theorem -/
theorem solution_is_ray_iff_a_is_pm1 :
  ∀ a : ℝ, is_ray (solution_set a) ↔ (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_solution_is_ray_iff_a_is_pm1_l3785_378518


namespace NUMINAMATH_CALUDE_cosine_expression_equals_negative_one_l3785_378554

theorem cosine_expression_equals_negative_one :
  (Real.cos (64 * π / 180) * Real.cos (4 * π / 180) - Real.cos (86 * π / 180) * Real.cos (26 * π / 180)) /
  (Real.cos (71 * π / 180) * Real.cos (41 * π / 180) - Real.cos (49 * π / 180) * Real.cos (19 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_expression_equals_negative_one_l3785_378554


namespace NUMINAMATH_CALUDE_rongcheng_sample_points_l3785_378552

/-- Represents the number of observation points in each county -/
structure ObservationPoints where
  xiongxian : ℕ
  rongcheng : ℕ
  anxin : ℕ

/-- Checks if three numbers form an arithmetic sequence -/
def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- Checks if three numbers form a geometric sequence -/
def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Calculates the number of data points for stratified sampling -/
def stratified_sample (total_samples : ℕ) (points : ObservationPoints) (county : ℕ) : ℕ :=
  (county * total_samples) / (points.xiongxian + points.rongcheng + points.anxin)

theorem rongcheng_sample_points :
  ∀ (points : ObservationPoints),
    points.xiongxian = 6 →
    is_arithmetic_sequence points.xiongxian points.rongcheng points.anxin →
    is_geometric_sequence points.xiongxian points.rongcheng (points.anxin + 6) →
    stratified_sample 12 points points.rongcheng = 4 :=
by sorry

end NUMINAMATH_CALUDE_rongcheng_sample_points_l3785_378552


namespace NUMINAMATH_CALUDE_complex_number_properties_l3785_378572

def i : ℂ := Complex.I

theorem complex_number_properties (z : ℂ) (h : z * (2 - i) = i ^ 2020) :
  (Complex.im z = 1/5) ∧ (Complex.re z > 0 ∧ Complex.im z > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3785_378572


namespace NUMINAMATH_CALUDE_abs_inequality_l3785_378514

theorem abs_inequality (x y : ℝ) 
  (h1 : |x + y + 1| ≤ 1/3) 
  (h2 : |y - 1/3| ≤ 2/3) : 
  |2/3 * x + 1| ≥ 4/9 := by
sorry

end NUMINAMATH_CALUDE_abs_inequality_l3785_378514


namespace NUMINAMATH_CALUDE_systematic_sampling_l3785_378505

/-- Systematic sampling problem -/
theorem systematic_sampling 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (groups : ℕ) 
  (interval : ℕ) 
  (group_15_num : ℕ) 
  (h1 : total_students = 160) 
  (h2 : sample_size = 20) 
  (h3 : groups = 20) 
  (h4 : interval = 8) 
  (h5 : group_15_num = 116) :
  ∃ (first_group_num : ℕ), 
    first_group_num + interval * (15 - 1) = group_15_num ∧ 
    first_group_num = 4 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l3785_378505


namespace NUMINAMATH_CALUDE_roots_and_p_value_l3785_378551

-- Define the polynomial
def f (p : ℝ) (x : ℝ) : ℝ := x^3 + 7*x^2 + 14*x - p

-- Define the condition of three distinct roots in geometric progression
def has_three_distinct_roots_in_gp (p : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    f p a = 0 ∧ f p b = 0 ∧ f p c = 0 ∧
    ∃ (r : ℝ), r ≠ 0 ∧ r ≠ 1 ∧ b = a * r ∧ c = b * r

-- Theorem statement
theorem roots_and_p_value (p : ℝ) :
  has_three_distinct_roots_in_gp p →
  p = -8 ∧ f p (-1) = 0 ∧ f p (-2) = 0 ∧ f p (-4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_and_p_value_l3785_378551


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_90_l3785_378575

theorem thirty_percent_less_than_90 (x : ℝ) : x + (1/4) * x = 63 ↔ x = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_90_l3785_378575


namespace NUMINAMATH_CALUDE_equation_one_solutions_l3785_378509

theorem equation_one_solutions (x : ℝ) : x^2 - 9 = 0 ↔ x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l3785_378509


namespace NUMINAMATH_CALUDE_range_of_f_l3785_378570

def f (x : ℝ) : ℝ := x^2 - 6*x + 10

theorem range_of_f :
  Set.range f = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3785_378570


namespace NUMINAMATH_CALUDE_notebook_pen_equation_l3785_378523

theorem notebook_pen_equation (x : ℝ) : 
  (5 * (x - 2) + 3 * x = 14) ↔ 
  (∃ (notebook_price : ℝ), 
    notebook_price = x - 2 ∧ 
    5 * notebook_price + 3 * x = 14) :=
by sorry

end NUMINAMATH_CALUDE_notebook_pen_equation_l3785_378523


namespace NUMINAMATH_CALUDE_jasper_refreshments_difference_l3785_378537

/-- Proves that Jasper sold 12 more drinks than hot dogs -/
theorem jasper_refreshments_difference : 
  ∀ (chips hot_dogs drinks : ℕ), 
    chips = 27 → 
    hot_dogs = chips - 8 → 
    drinks = 31 → 
    drinks - hot_dogs = 12 := by
  sorry

end NUMINAMATH_CALUDE_jasper_refreshments_difference_l3785_378537


namespace NUMINAMATH_CALUDE_b_not_unique_l3785_378502

-- Define the line equation
def line_equation (y : ℝ) : ℝ := 8 * y + 5

-- Define the points on the line
def point1 (m B : ℝ) : ℝ × ℝ := (m, B)
def point2 (m B : ℝ) : ℝ × ℝ := (m + 2, B + 0.25)

-- Theorem stating that B cannot be uniquely determined
theorem b_not_unique (m B : ℝ) : 
  line_equation B = (point1 m B).1 ∧ 
  line_equation (B + 0.25) = (point2 m B).1 → 
  ∃ (B' : ℝ), B' ≠ B ∧ 
    line_equation B' = (point1 m B').1 ∧ 
    line_equation (B' + 0.25) = (point2 m B').1 :=
by
  sorry


end NUMINAMATH_CALUDE_b_not_unique_l3785_378502


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_count_l3785_378586

/-- The number of intersection points of diagonals in a regular decagon -/
def decagon_diagonal_intersections : ℕ :=
  Nat.choose 10 4

/-- Theorem stating that the number of interior intersection points of diagonals
    in a regular decagon is equal to the number of ways to choose 4 vertices from 10 -/
theorem decagon_diagonal_intersections_count :
  decagon_diagonal_intersections = 210 := by
  sorry

#eval decagon_diagonal_intersections

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_count_l3785_378586


namespace NUMINAMATH_CALUDE_line_through_circle_centers_l3785_378564

/-- Given two circles that pass through (1, 1), prove the equation of the line through their centers -/
theorem line_through_circle_centers (D₁ E₁ D₂ E₂ : ℝ) : 
  (1^2 + 1^2 + D₁*1 + E₁*1 + 3 = 0) →
  (1^2 + 1^2 + D₂*1 + E₂*1 + 3 = 0) →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x = D₁ ∧ y = E₁) ∨ (x = D₂ ∧ y = E₂) → x + y + 5 = k := by
  sorry

#check line_through_circle_centers

end NUMINAMATH_CALUDE_line_through_circle_centers_l3785_378564


namespace NUMINAMATH_CALUDE_inequality_range_l3785_378539

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3785_378539


namespace NUMINAMATH_CALUDE_jason_shelves_needed_l3785_378500

/-- Calculates the number of shelves needed to store books -/
def shelves_needed (regular_books : ℕ) (large_books : ℕ) : ℕ :=
  let regular_shelves := (regular_books + 44) / 45
  let large_shelves := (large_books + 29) / 30
  regular_shelves + large_shelves

/-- Theorem stating that Jason needs 9 shelves to store all his books -/
theorem jason_shelves_needed : shelves_needed 240 75 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jason_shelves_needed_l3785_378500


namespace NUMINAMATH_CALUDE_kira_away_time_l3785_378561

/-- Represents the rate at which the cat eats kibble in hours per pound -/
def eating_rate : ℝ := 4

/-- Represents the initial amount of kibble in the bowl in pounds -/
def initial_kibble : ℝ := 3

/-- Represents the amount of kibble left when Kira returns in pounds -/
def remaining_kibble : ℝ := 1

/-- Represents the time Kira was away from home in hours -/
def time_away : ℝ := eating_rate * (initial_kibble - remaining_kibble)

theorem kira_away_time : time_away = 8 := by
  sorry

end NUMINAMATH_CALUDE_kira_away_time_l3785_378561


namespace NUMINAMATH_CALUDE_and_implies_or_but_not_conversely_l3785_378565

-- Define propositions p and q
variable (p q : Prop)

-- State the theorem
theorem and_implies_or_but_not_conversely :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) :=
by
  sorry


end NUMINAMATH_CALUDE_and_implies_or_but_not_conversely_l3785_378565


namespace NUMINAMATH_CALUDE_number_puzzle_l3785_378522

theorem number_puzzle (x y : ℝ) (h1 : x + y = 25) (h2 : x - y = 15) : x^2 - y^3 = 275 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3785_378522


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3785_378585

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  ab + bc + 2*c*a ≤ 9/2 ∧ ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧ ab + bc + 2*c*a = 9/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3785_378585


namespace NUMINAMATH_CALUDE_fibonacci_identity_l3785_378525

def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_identity (n : ℕ) (h : n ≥ 1) :
  fib (n - 1) * fib (n + 1) - fib n ^ 2 = (-1) ^ n := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_identity_l3785_378525


namespace NUMINAMATH_CALUDE_project_hours_difference_l3785_378547

/-- Given a project with three contributors (Pat, Kate, and Mark) with specific charging ratios,
    prove the difference in hours charged between Mark and Kate. -/
theorem project_hours_difference (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 198 →
  kate_hours + 2 * kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 110 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l3785_378547


namespace NUMINAMATH_CALUDE_w_value_l3785_378557

-- Define the coefficients of the first polynomial
def a : ℝ := 1
def b : ℝ := 5
def c : ℝ := 6
def d : ℝ := -7

-- Define the roots of the first polynomial
variable (p q r : ℝ)

-- Define the coefficients of the second polynomial
variable (u v w : ℝ)

-- Axioms based on the problem conditions
axiom root_condition : a * p^3 + b * p^2 + c * p + d = 0 ∧
                       a * q^3 + b * q^2 + c * q + d = 0 ∧
                       a * r^3 + b * r^2 + c * r + d = 0

axiom new_root_condition : (p + q)^3 + u * (p + q)^2 + v * (p + q) + w = 0 ∧
                           (q + r)^3 + u * (q + r)^2 + v * (q + r) + w = 0 ∧
                           (r + p)^3 + u * (r + p)^2 + v * (r + p) + w = 0

-- Theorem to prove
theorem w_value : w = 37 := by
  sorry

end NUMINAMATH_CALUDE_w_value_l3785_378557


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l3785_378560

def original_soup_price : ℚ := 7.50 / 3
def original_bread_price : ℚ := 5 / 2
def new_soup_price : ℚ := 8 / 4
def new_bread_price : ℚ := 6 / 3

def original_bundle_avg : ℚ := (original_soup_price + original_bread_price) / 2
def new_bundle_avg : ℚ := (new_soup_price + new_bread_price) / 2

theorem price_decrease_percentage :
  (original_bundle_avg - new_bundle_avg) / original_bundle_avg * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l3785_378560


namespace NUMINAMATH_CALUDE_banknote_sum_divisibility_l3785_378595

theorem banknote_sum_divisibility
  (a b : ℕ)
  (h_distinct : a % 101 ≠ b % 101)
  (h_total : ℕ)
  (h_count : h_total = 100) :
  ∃ (m n : ℕ), m + n ≤ h_total ∧ (m * a + n * b) % 101 = 0 :=
sorry

end NUMINAMATH_CALUDE_banknote_sum_divisibility_l3785_378595


namespace NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l3785_378532

theorem arithmetic_sequence_divisibility (a : ℕ) :
  ∃! k : Fin 7, ∃ n : ℕ, n = a + k * 30 ∧ n % 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l3785_378532


namespace NUMINAMATH_CALUDE_a_4_equals_8_l3785_378530

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ :=
  if n = 0 then S 0
  else S n - S (n-1)

theorem a_4_equals_8 : a 4 = 8 := by sorry

end NUMINAMATH_CALUDE_a_4_equals_8_l3785_378530


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l3785_378529

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {-3+a, 2*a-1, a^2+1}

theorem set_intersection_and_union (a : ℝ) :
  (A a) ∩ (B a) = {-3} →
  a = -1 ∧ (A a) ∪ (B a) = {-4, -3, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l3785_378529


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l3785_378533

theorem partial_fraction_decomposition_sum (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l3785_378533


namespace NUMINAMATH_CALUDE_interior_angles_sum_l3785_378590

theorem interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 1440) →
  (180 * ((n + 3) - 2) = 1980) :=
by sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l3785_378590


namespace NUMINAMATH_CALUDE_alia_markers_count_l3785_378589

theorem alia_markers_count : ∀ (steve austin alia : ℕ),
  steve = 60 →
  austin = steve / 3 →
  alia = 2 * austin →
  alia = 40 := by
sorry

end NUMINAMATH_CALUDE_alia_markers_count_l3785_378589


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3785_378563

def repeating_decimal (a b : ℕ) : ℚ := (a : ℚ) / (99 : ℚ) + (b : ℚ) / (100 : ℚ)

theorem decimal_to_fraction :
  ∃ (n d : ℕ), d ≠ 0 ∧ 
  repeating_decimal 36 0 = (n : ℚ) / (d : ℚ) ∧
  ∀ (n' d' : ℕ), d' ≠ 0 → repeating_decimal 36 0 = (n' : ℚ) / (d' : ℚ) → d ≤ d' ∧
  d = 11 :=
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3785_378563


namespace NUMINAMATH_CALUDE_prop_analysis_l3785_378559

-- Define the original proposition
def original_prop (x y : ℝ) : Prop := (x + y = 5) → (x = 3 ∧ y = 2)

-- Define the converse
def converse (x y : ℝ) : Prop := (x = 3 ∧ y = 2) → (x + y = 5)

-- Define the inverse
def inverse (x y : ℝ) : Prop := (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2)

-- Define the contrapositive
def contrapositive (x y : ℝ) : Prop := (x ≠ 3 ∨ y ≠ 2) → (x + y ≠ 5)

-- Theorem stating the truth values of converse, inverse, and contrapositive
theorem prop_analysis :
  (∀ x y : ℝ, converse x y) ∧
  (¬ ∀ x y : ℝ, inverse x y) ∧
  (∀ x y : ℝ, contrapositive x y) :=
by sorry

end NUMINAMATH_CALUDE_prop_analysis_l3785_378559


namespace NUMINAMATH_CALUDE_sara_second_book_cost_l3785_378527

/-- The cost of Sara's second book -/
def second_book_cost (first_book_cost bill_given change_received : ℝ) : ℝ :=
  bill_given - change_received - first_book_cost

/-- Theorem stating the cost of Sara's second book -/
theorem sara_second_book_cost :
  second_book_cost 5.5 20 8 = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_sara_second_book_cost_l3785_378527


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l3785_378512

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function has period T if f(x + T) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_period : HasPeriod f 5) 
    (h1 : f 1 = 1) 
    (h2 : f 2 = 2) : 
  f 3 - f 4 = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l3785_378512


namespace NUMINAMATH_CALUDE_main_theorem_l3785_378516

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, Differentiable ℝ f ∧ f x + (deriv^[2] f) x > 0

/-- The main theorem -/
theorem main_theorem (f : ℝ → ℝ) (hf : satisfies_condition f) :
  ∀ a b : ℝ, a > b ↔ Real.exp a * f a > Real.exp b * f b :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l3785_378516


namespace NUMINAMATH_CALUDE_parabola_homothety_transform_l3785_378510

/-- A homothety transformation centered at (0,0) with ratio k > 0 -/
structure Homothety where
  k : ℝ
  h_pos : k > 0

/-- The equation of a parabola in the form 2py = x^2 -/
def Parabola (p : ℝ) (x y : ℝ) : Prop := 2 * p * y = x^2

theorem parabola_homothety_transform (p : ℝ) (h_p : p ≠ 0) :
  ∃ (h : Homothety), ∀ (x y : ℝ),
    Parabola p x y ↔ y = x^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_homothety_transform_l3785_378510


namespace NUMINAMATH_CALUDE_floor_divisibility_l3785_378553

theorem floor_divisibility (n : ℕ) : 
  (2^(n+1) : ℤ) ∣ ⌊((1 : ℝ) + Real.sqrt 3)^(2*n+1)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_divisibility_l3785_378553


namespace NUMINAMATH_CALUDE_log_stack_sum_l3785_378596

theorem log_stack_sum (n : ℕ) (a l : ℕ) (h1 : n = 12) (h2 : a = 15) (h3 : l = 4) :
  n * (a + l) / 2 = 114 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l3785_378596


namespace NUMINAMATH_CALUDE_f_property_l3785_378519

def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem f_property (a b : ℝ) : f a b (-2) = 3 → f a b 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l3785_378519


namespace NUMINAMATH_CALUDE_second_number_value_l3785_378541

theorem second_number_value (x y : ℝ) 
  (h1 : (1/5) * x = (5/8) * y) 
  (h2 : x + 35 = 4 * y) : 
  y = 40 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l3785_378541


namespace NUMINAMATH_CALUDE_sum_greater_than_product_l3785_378528

theorem sum_greater_than_product (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : Real.arctan x + Real.arctan y + Real.arctan z < π) : 
  x + y + z > x * y * z := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_product_l3785_378528


namespace NUMINAMATH_CALUDE_no_solution_implies_b_bounded_l3785_378567

theorem no_solution_implies_b_bounded (a b : ℝ) :
  (∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) →
  abs b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_b_bounded_l3785_378567


namespace NUMINAMATH_CALUDE_square_of_complex_number_l3785_378503

theorem square_of_complex_number :
  let z : ℂ := 5 - 3*I
  z^2 = 16 - 30*I := by sorry

end NUMINAMATH_CALUDE_square_of_complex_number_l3785_378503
