import Mathlib

namespace NUMINAMATH_CALUDE_num_parallel_planes_zero_or_one_l2674_267478

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew (a b : Line3D) : Prop := sorry

/-- A point is outside a line if it does not lie on the line. -/
def is_outside (P : Point3D) (l : Line3D) : Prop := sorry

/-- A plane is parallel to a line if they do not intersect. -/
def plane_parallel_to_line (π : Plane3D) (l : Line3D) : Prop := sorry

/-- The number of planes passing through a point and parallel to two lines. -/
def num_parallel_planes (P : Point3D) (a b : Line3D) : ℕ := sorry

theorem num_parallel_planes_zero_or_one 
  (P : Point3D) (a b : Line3D) 
  (h_skew : are_skew a b) 
  (h_outside_a : is_outside P a) 
  (h_outside_b : is_outside P b) : 
  num_parallel_planes P a b = 0 ∨ num_parallel_planes P a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_num_parallel_planes_zero_or_one_l2674_267478


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2674_267481

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2674_267481


namespace NUMINAMATH_CALUDE_ratio_of_65_to_13_l2674_267462

theorem ratio_of_65_to_13 (certain_number : ℚ) (h : certain_number = 65) : 
  certain_number / 13 = 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_65_to_13_l2674_267462


namespace NUMINAMATH_CALUDE_S_in_quadrants_I_and_II_l2674_267473

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > 2 * p.1 ∧ p.2 > 4 - p.1}

-- Define quadrants I and II
def quadrantI : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}
def quadrantII : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

-- Theorem stating that S is contained in quadrants I and II
theorem S_in_quadrants_I_and_II : S ⊆ quadrantI ∪ quadrantII := by
  sorry


end NUMINAMATH_CALUDE_S_in_quadrants_I_and_II_l2674_267473


namespace NUMINAMATH_CALUDE_lcm_problem_l2674_267447

theorem lcm_problem (m n : ℕ) : 
  m > 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  Nat.lcm m n = 690 ∧ 
  ¬(3 ∣ n) ∧ 
  ¬(2 ∣ m) → 
  n = 230 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l2674_267447


namespace NUMINAMATH_CALUDE_sum_of_primes_even_l2674_267419

/-- A number is prime if it's greater than 1 and has no positive divisors other than 1 and itself -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem sum_of_primes_even 
  (A B C : ℕ) 
  (hA : isPrime A) 
  (hB : isPrime B) 
  (hC : isPrime C) 
  (hAB_minus : isPrime (A - B)) 
  (hAB_plus : isPrime (A + B)) 
  (hABC : isPrime (A + B + C)) : 
  Even (A + B + C + (A - B) + (A + B) + (A + B + C)) := by
sorry

end NUMINAMATH_CALUDE_sum_of_primes_even_l2674_267419


namespace NUMINAMATH_CALUDE_f_g_inequality_l2674_267457

/-- The function f(x) = -x³ + x² + x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + x^2 + x + a

/-- The function g(x) = 2a - x³ -/
def g (a : ℝ) (x : ℝ) : ℝ := 2*a - x^3

/-- Theorem: If g(x) ≥ f(x) for all x ∈ [0, 1], then a ≥ 2 -/
theorem f_g_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, g a x ≥ f a x) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_g_inequality_l2674_267457


namespace NUMINAMATH_CALUDE_optimal_pricing_and_profit_daily_profit_function_l2674_267453

/-- Represents the daily profit function for a product --/
def daily_profit (x : ℝ) : ℝ := -3 * x^2 + 252 * x - 4860

/-- Represents the constraint on the selling price --/
def price_constraint (x : ℝ) : Prop := 30 ≤ x ∧ x ≤ 54

/-- The theorem stating the optimal selling price and maximum profit --/
theorem optimal_pricing_and_profit :
  ∃ (x : ℝ), price_constraint x ∧ 
    (∀ y, price_constraint y → daily_profit y ≤ daily_profit x) ∧
    x = 42 ∧ daily_profit x = 432 := by
  sorry

/-- The theorem stating the form of the daily profit function --/
theorem daily_profit_function (x : ℝ) :
  daily_profit x = (x - 30) * (162 - 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_optimal_pricing_and_profit_daily_profit_function_l2674_267453


namespace NUMINAMATH_CALUDE_family_reunion_food_l2674_267469

/-- The amount of food Peter buys for the family reunion -/
def total_food (chicken hamburger hotdog side : ℝ) : ℝ :=
  chicken + hamburger + hotdog + side

theorem family_reunion_food : ∃ (chicken hamburger hotdog side : ℝ),
  chicken = 16 ∧
  hamburger = chicken / 2 ∧
  hotdog = hamburger + 2 ∧
  side = hotdog / 2 ∧
  total_food chicken hamburger hotdog side = 39 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_food_l2674_267469


namespace NUMINAMATH_CALUDE_value_added_to_numbers_l2674_267466

theorem value_added_to_numbers (n : ℕ) (initial_avg final_avg x : ℚ) : 
  n = 15 → initial_avg = 40 → final_avg = 52 → 
  n * final_avg = n * initial_avg + n * x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_numbers_l2674_267466


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l2674_267485

theorem min_value_theorem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  4 * a^3 + 8 * b^3 + 27 * c^3 + 64 * d^3 + 2 / (a * b * c * d) ≥ 16 * Real.sqrt 3 :=
by
  sorry

theorem min_value_achievable :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    4 * a^3 + 8 * b^3 + 27 * c^3 + 64 * d^3 + 2 / (a * b * c * d) = 16 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l2674_267485


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2674_267443

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 5 = 0.3 →
  arithmetic_sequence a₁ d 12 = 3.1 →
  a₁ = -1.3 ∧ d = 0.4 ∧
  (arithmetic_sequence a₁ d 18 +
   arithmetic_sequence a₁ d 19 +
   arithmetic_sequence a₁ d 20 +
   arithmetic_sequence a₁ d 21 +
   arithmetic_sequence a₁ d 22) = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2674_267443


namespace NUMINAMATH_CALUDE_negation_of_for_all_leq_zero_l2674_267448

theorem negation_of_for_all_leq_zero :
  (¬ ∀ x : ℝ, Real.exp x - 2 * Real.sin x + 4 ≤ 0) ↔
  (∃ x : ℝ, Real.exp x - 2 * Real.sin x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_for_all_leq_zero_l2674_267448


namespace NUMINAMATH_CALUDE_max_digits_product_l2674_267405

theorem max_digits_product (a b : ℕ) : 
  1000 ≤ a ∧ a < 10000 → 10000 ≤ b ∧ b < 100000 → 
  a * b < 1000000000 :=
sorry

end NUMINAMATH_CALUDE_max_digits_product_l2674_267405


namespace NUMINAMATH_CALUDE_kendall_driving_distance_l2674_267418

theorem kendall_driving_distance (distance_with_mother distance_with_father : Real) 
  (h1 : distance_with_mother = 0.17)
  (h2 : distance_with_father = 0.5) : 
  distance_with_mother + distance_with_father = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_kendall_driving_distance_l2674_267418


namespace NUMINAMATH_CALUDE_line_segment_length_l2674_267454

theorem line_segment_length (volume : ℝ) (radius : ℝ) (length : ℝ) : 
  volume = 432 * Real.pi →
  radius = 4 →
  volume = (Real.pi * radius^2 * length) + (2 * (2/3) * Real.pi * radius^3) →
  length = 50/3 := by
sorry

end NUMINAMATH_CALUDE_line_segment_length_l2674_267454


namespace NUMINAMATH_CALUDE_intersection_implies_outside_circle_l2674_267402

theorem intersection_implies_outside_circle (a b : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  a^2 + b^2 > 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_outside_circle_l2674_267402


namespace NUMINAMATH_CALUDE_triangle_side_length_l2674_267409

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  a = 3 →
  b = Real.sqrt 6 →
  B = π / 4 →
  c = (3 * Real.sqrt 2 + Real.sqrt 6) / 2 ∨ c = (3 * Real.sqrt 2 - Real.sqrt 6) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2674_267409


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2674_267456

/-- A parabola defined by y = ax² where a > 0 -/
structure Parabola where
  a : ℝ
  a_pos : a > 0

/-- A line with slope 1 -/
structure Line where
  b : ℝ

/-- Intersection points of a parabola and a line -/
structure Intersection (p : Parabola) (l : Line) where
  x₁ : ℝ
  x₂ : ℝ
  y₁ : ℝ
  y₂ : ℝ
  eq₁ : y₁ = p.a * x₁^2
  eq₂ : y₂ = p.a * x₂^2
  eq₃ : y₁ = x₁ + l.b
  eq₄ : y₂ = x₂ + l.b

/-- The theorem to be proved -/
theorem parabola_focus_directrix_distance 
  (p : Parabola) (l : Line) (i : Intersection p l) :
  (i.x₁ + i.x₂) / 2 = 1 → 1 / (4 * p.a) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2674_267456


namespace NUMINAMATH_CALUDE_min_value_and_max_value_l2674_267410

theorem min_value_and_max_value :
  (∀ x : ℝ, x > 1 → (x + 1 / (x - 1)) ≥ 3) ∧
  (∃ x : ℝ, x > 1 ∧ (x + 1 / (x - 1)) = 3) ∧
  (∀ x : ℝ, 0 < x ∧ x < 10 → Real.sqrt (x * (10 - x)) ≤ 5) ∧
  (∃ x : ℝ, 0 < x ∧ x < 10 ∧ Real.sqrt (x * (10 - x)) = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_max_value_l2674_267410


namespace NUMINAMATH_CALUDE_sqrt_three_simplification_l2674_267468

theorem sqrt_three_simplification : 3 * Real.sqrt 3 - 2 * Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_simplification_l2674_267468


namespace NUMINAMATH_CALUDE_f_plus_g_is_linear_l2674_267458

/-- Represents a cubic function ax³ + bx² + cx + d -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The function resulting from reflecting and translating a cubic function -/
def reflected_translated (cf : CubicFunction) (x : ℝ) : ℝ :=
  cf.a * (x - 10)^3 + cf.b * (x - 10)^2 + cf.c * (x - 10) + cf.d

/-- The function resulting from reflecting about x-axis, then translating a cubic function -/
def reflected_translated_negative (cf : CubicFunction) (x : ℝ) : ℝ :=
  -cf.a * (x + 10)^3 - cf.b * (x + 10)^2 - cf.c * (x + 10) - cf.d

/-- The sum of the two reflected and translated functions -/
def f_plus_g (cf : CubicFunction) (x : ℝ) : ℝ :=
  reflected_translated cf x + reflected_translated_negative cf x

/-- Theorem stating that f_plus_g is a non-horizontal linear function -/
theorem f_plus_g_is_linear (cf : CubicFunction) :
  ∃ m k, m ≠ 0 ∧ ∀ x, f_plus_g cf x = m * x + k :=
sorry

end NUMINAMATH_CALUDE_f_plus_g_is_linear_l2674_267458


namespace NUMINAMATH_CALUDE_quadratic_equation_determination_l2674_267496

theorem quadratic_equation_determination (b c : ℝ) :
  (∀ x : ℝ, x^2 + b*x + c = 0 → x = 5 ∨ x = 3 ∨ x = -6 ∨ x = -4) →
  (5 + 3 = -b) →
  ((-6) * (-4) = c) →
  (b = -8 ∧ c = 24) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_determination_l2674_267496


namespace NUMINAMATH_CALUDE_no_three_digit_odd_divisible_by_six_l2674_267430

theorem no_three_digit_odd_divisible_by_six : 
  ¬ ∃ n : ℕ, 
    (100 ≤ n ∧ n ≤ 999) ∧ 
    (∀ d, d ∈ n.digits 10 → d % 2 = 1 ∧ d > 4) ∧ 
    n % 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_three_digit_odd_divisible_by_six_l2674_267430


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_sequence_l2674_267425

def arithmetic_sequence (a₁ a₂ a₃ a₄ : ℚ) : Prop :=
  ∃ (d : ℚ), a₂ = a₁ + d ∧ a₃ = a₂ + d ∧ a₄ = a₃ + d

def fifth_term (a₁ a₂ a₃ a₄ : ℚ) : ℚ :=
  a₁ + 4 * (a₂ - a₁)

theorem fifth_term_of_specific_sequence (x y : ℚ) :
  arithmetic_sequence (x + 2*y) (x - 2*y) (3*x*y) (x/(2*y)) →
  fifth_term (x + 2*y) (x - 2*y) (3*x*y) (x/(2*y)) = (20*y^2)/(2*y-1) - 14*y :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_sequence_l2674_267425


namespace NUMINAMATH_CALUDE_base7_to_base10_l2674_267442

/-- Converts a list of digits in base b to its decimal (base 10) representation -/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The digits of 45321 in base 7, in reverse order (least significant digit first) -/
def digits : List Nat := [1, 2, 3, 5, 4]

/-- The statement that 45321 in base 7 is equal to 11481 in base 10 -/
theorem base7_to_base10 : toDecimal digits 7 = 11481 := by sorry

end NUMINAMATH_CALUDE_base7_to_base10_l2674_267442


namespace NUMINAMATH_CALUDE_unique_albums_count_l2674_267461

/-- Represents the album collections of Andrew, John, and Samantha -/
structure AlbumCollections where
  andrew_total : ℕ
  john_total : ℕ
  samantha_total : ℕ
  andrew_john_shared : ℕ
  andrew_samantha_shared : ℕ
  john_samantha_shared : ℕ

/-- Calculates the number of unique albums given the album collections -/
def uniqueAlbums (c : AlbumCollections) : ℕ :=
  (c.andrew_total - c.andrew_john_shared - c.andrew_samantha_shared) +
  (c.john_total - c.andrew_john_shared - c.john_samantha_shared) +
  (c.samantha_total - c.andrew_samantha_shared - c.john_samantha_shared)

/-- Theorem stating that the number of unique albums is 26 for the given collection -/
theorem unique_albums_count :
  let c : AlbumCollections := {
    andrew_total := 23,
    john_total := 20,
    samantha_total := 15,
    andrew_john_shared := 12,
    andrew_samantha_shared := 3,
    john_samantha_shared := 5
  }
  uniqueAlbums c = 26 := by
  sorry

end NUMINAMATH_CALUDE_unique_albums_count_l2674_267461


namespace NUMINAMATH_CALUDE_ab_value_l2674_267494

theorem ab_value (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 80) : a * b = 32 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2674_267494


namespace NUMINAMATH_CALUDE_olaf_boat_crew_size_l2674_267437

/-- Proves the number of men on Olaf's boat given the travel conditions -/
theorem olaf_boat_crew_size :
  ∀ (total_distance : ℝ) 
    (boat_speed : ℝ) 
    (water_per_man_per_day : ℝ) 
    (total_water : ℝ),
  total_distance = 4000 →
  boat_speed = 200 →
  water_per_man_per_day = 1/2 →
  total_water = 250 →
  (total_water / ((total_distance / boat_speed) * water_per_man_per_day) : ℝ) = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_olaf_boat_crew_size_l2674_267437


namespace NUMINAMATH_CALUDE_garden_scale_drawing_l2674_267407

/-- Represents the length in feet given a scale drawing measurement -/
def actualLength (scale : ℝ) (drawingLength : ℝ) : ℝ :=
  scale * drawingLength

theorem garden_scale_drawing :
  let scale : ℝ := 500  -- 1 inch represents 500 feet
  let drawingLength : ℝ := 6.5  -- length in the drawing is 6.5 inches
  actualLength scale drawingLength = 3250 := by
  sorry

end NUMINAMATH_CALUDE_garden_scale_drawing_l2674_267407


namespace NUMINAMATH_CALUDE_implication_not_equivalence_l2674_267463

theorem implication_not_equivalence :
  ∃ (a : ℝ), (∀ (x : ℝ), (abs (5 * x - 1) > a) → (x^2 - (3/2) * x + 1/2 > 0)) ∧
             (∃ (y : ℝ), (y^2 - (3/2) * y + 1/2 > 0) ∧ (abs (5 * y - 1) ≤ a)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_implication_not_equivalence_l2674_267463


namespace NUMINAMATH_CALUDE_probability_one_defective_six_two_l2674_267403

/-- The probability of selecting exactly one defective product from a set of products -/
def probability_one_defective (total : ℕ) (defective : ℕ) : ℚ :=
  let qualified := total - defective
  (defective.choose 1 * qualified.choose 1 : ℚ) / total.choose 2

/-- Given 6 products with 2 defective ones, the probability of selecting exactly one defective product is 8/15 -/
theorem probability_one_defective_six_two :
  probability_one_defective 6 2 = 8 / 15 := by
  sorry

#eval probability_one_defective 6 2

end NUMINAMATH_CALUDE_probability_one_defective_six_two_l2674_267403


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l2674_267412

theorem nested_fraction_evaluation : 
  2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l2674_267412


namespace NUMINAMATH_CALUDE_second_group_size_l2674_267467

/-- The number of men in the first group -/
def first_group : ℕ := 20

/-- The number of days taken by the first group -/
def first_days : ℕ := 30

/-- The number of days taken by the second group -/
def second_days : ℕ := 24

/-- The total amount of work in man-days -/
def total_work : ℕ := first_group * first_days

/-- The number of men in the second group -/
def second_group : ℕ := total_work / second_days

theorem second_group_size : second_group = 25 := by
  sorry

end NUMINAMATH_CALUDE_second_group_size_l2674_267467


namespace NUMINAMATH_CALUDE_doctor_engineer_ratio_l2674_267438

theorem doctor_engineer_ratio (d l e : ℕ) (avg_age : ℚ) : 
  avg_age = 45 →
  (40 * d + 55 * l + 50 * e : ℚ) / (d + l + e : ℚ) = avg_age →
  d = 3 * e :=
by sorry

end NUMINAMATH_CALUDE_doctor_engineer_ratio_l2674_267438


namespace NUMINAMATH_CALUDE_marys_potatoes_l2674_267475

def potatoes_problem (initial_potatoes : ℕ) (eaten_potatoes : ℕ) : Prop :=
  initial_potatoes - eaten_potatoes = 5

theorem marys_potatoes : potatoes_problem 8 3 :=
sorry

end NUMINAMATH_CALUDE_marys_potatoes_l2674_267475


namespace NUMINAMATH_CALUDE_sequence_property_l2674_267482

theorem sequence_property (a : ℕ → ℕ) 
  (h1 : ∀ n, 1 < a n)
  (h2 : ∀ n, a n < a (n + 1))
  (h3 : ∀ n, a (n + a n) = 2 * a n) :
  ∃ c : ℕ, ∀ n, a n = n + c := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l2674_267482


namespace NUMINAMATH_CALUDE_harkamal_payment_l2674_267421

/-- The amount Harkamal paid to the shopkeeper -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Proof that Harkamal paid 1145 to the shopkeeper -/
theorem harkamal_payment : total_amount_paid 8 70 9 65 = 1145 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l2674_267421


namespace NUMINAMATH_CALUDE_tom_clothing_count_l2674_267444

/-- The total number of pieces of clothing Tom had -/
def total_clothing : ℕ := 36

/-- The number of pieces in the first load -/
def first_load : ℕ := 18

/-- The number of pieces in each of the two equal loads -/
def equal_load : ℕ := 9

/-- The number of equal loads -/
def num_equal_loads : ℕ := 2

theorem tom_clothing_count :
  total_clothing = first_load + num_equal_loads * equal_load :=
by sorry

end NUMINAMATH_CALUDE_tom_clothing_count_l2674_267444


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2674_267413

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 8) * (x - 6) = -50 + k * x) ↔ 
  (k = -10 + 2 * Real.sqrt 6 ∨ k = -10 - 2 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2674_267413


namespace NUMINAMATH_CALUDE_tank_b_circumference_l2674_267431

/-- The circumference of Tank B given the conditions of the problem -/
theorem tank_b_circumference : 
  ∀ (h_a h_b c_a c_b r_a r_b v_a v_b : ℝ),
  h_a = 7 →
  h_b = 8 →
  c_a = 8 →
  c_a = 2 * Real.pi * r_a →
  v_a = Real.pi * r_a^2 * h_a →
  v_b = Real.pi * r_b^2 * h_b →
  v_a = 0.5600000000000001 * v_b →
  c_b = 2 * Real.pi * r_b →
  c_b = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_b_circumference_l2674_267431


namespace NUMINAMATH_CALUDE_fourth_month_sales_l2674_267486

/-- Calculates the missing sales amount for a month given the sales of other months and the average --/
def calculate_missing_sales (sale1 sale2 sale3 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)

theorem fourth_month_sales :
  let sale1 : ℕ := 6400
  let sale2 : ℕ := 7000
  let sale3 : ℕ := 6800
  let sale5 : ℕ := 6500
  let sale6 : ℕ := 5100
  let average : ℕ := 6500
  calculate_missing_sales sale1 sale2 sale3 sale5 sale6 average = 7200 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l2674_267486


namespace NUMINAMATH_CALUDE_honey_production_l2674_267414

/-- The amount of honey (in grams) produced by a single bee in 60 days -/
def single_bee_honey : ℕ := 1

/-- The number of bees in the group -/
def num_bees : ℕ := 60

/-- The amount of honey (in grams) produced by the group of bees in 60 days -/
def group_honey : ℕ := num_bees * single_bee_honey

theorem honey_production :
  group_honey = 60 := by sorry

end NUMINAMATH_CALUDE_honey_production_l2674_267414


namespace NUMINAMATH_CALUDE_simple_random_for_ten_basketballs_l2674_267497

/-- Enumeration of sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | WithReplacement

/-- Definition of a sampling scenario --/
structure SamplingScenario where
  population_size : ℕ
  sample_size : ℕ
  for_quality_testing : Bool

/-- Function to determine the appropriate sampling method --/
def appropriate_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- Theorem stating that Simple Random Sampling is appropriate for the given scenario --/
theorem simple_random_for_ten_basketballs :
  let scenario : SamplingScenario := {
    population_size := 10,
    sample_size := 1,
    for_quality_testing := true
  }
  appropriate_sampling_method scenario = SamplingMethod.SimpleRandom :=
sorry

end NUMINAMATH_CALUDE_simple_random_for_ten_basketballs_l2674_267497


namespace NUMINAMATH_CALUDE_genevieve_errors_fixed_l2674_267415

/-- Represents a programmer's coding and debugging process -/
structure Programmer where
  total_lines : ℕ
  debug_interval : ℕ
  errors_per_debug : ℕ

/-- Calculates the total number of errors fixed by a programmer -/
def total_errors_fixed (p : Programmer) : ℕ :=
  (p.total_lines / p.debug_interval) * p.errors_per_debug

/-- Theorem stating that under given conditions, the programmer fixes 129 errors -/
theorem genevieve_errors_fixed :
  ∀ (p : Programmer),
    p.total_lines = 4300 →
    p.debug_interval = 100 →
    p.errors_per_debug = 3 →
    total_errors_fixed p = 129 := by
  sorry


end NUMINAMATH_CALUDE_genevieve_errors_fixed_l2674_267415


namespace NUMINAMATH_CALUDE_smallest_possible_total_l2674_267487

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- The ratios given in the problem --/
def ninth_to_tenth_ratio : Rat := 7 / 4
def ninth_to_eleventh_ratio : Rat := 5 / 3

/-- The condition that the ratios are correct --/
def ratios_correct (gc : GradeCount) : Prop :=
  (gc.ninth : Rat) / gc.tenth = ninth_to_tenth_ratio ∧
  (gc.ninth : Rat) / gc.eleventh = ninth_to_eleventh_ratio

/-- The total number of students --/
def total_students (gc : GradeCount) : ℕ :=
  gc.ninth + gc.tenth + gc.eleventh

/-- The main theorem to prove --/
theorem smallest_possible_total : 
  ∃ (gc : GradeCount), ratios_correct gc ∧ 
    (∀ (gc' : GradeCount), ratios_correct gc' → total_students gc ≤ total_students gc') ∧
    total_students gc = 76 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_total_l2674_267487


namespace NUMINAMATH_CALUDE_octagon_area_theorem_l2674_267493

/-- Represents an octagon with given width and height -/
structure Octagon where
  width : ℕ
  height : ℕ

/-- Calculates the area of the octagon -/
def octagonArea (o : Octagon) : ℕ :=
  -- The actual calculation is not provided, as it should be part of the proof
  sorry

/-- Theorem stating that an octagon with width 5 and height 8 has an area of 30 square units -/
theorem octagon_area_theorem (o : Octagon) (h1 : o.width = 5) (h2 : o.height = 8) : 
  octagonArea o = 30 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_theorem_l2674_267493


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l2674_267432

theorem parallelogram_side_sum (x y : ℝ) : 
  (12 * y - 2 = 10) → (5 * x + 15 = 20) → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l2674_267432


namespace NUMINAMATH_CALUDE_expression_evaluation_l2674_267474

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem expression_evaluation (y : ℕ) (x : ℕ) (h1 : y = 2) (h2 : x = y + 1) :
  5 * (factorial y) * (x ^ y) + 3 * (factorial x) * (y ^ x) = 234 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2674_267474


namespace NUMINAMATH_CALUDE_debate_team_group_size_l2674_267422

theorem debate_team_group_size 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (num_groups : ℕ) 
  (h1 : num_boys = 31)
  (h2 : num_girls = 32)
  (h3 : num_groups = 7) :
  (num_boys + num_girls) / num_groups = 9 := by
sorry

end NUMINAMATH_CALUDE_debate_team_group_size_l2674_267422


namespace NUMINAMATH_CALUDE_smallest_n_remainder_l2674_267400

theorem smallest_n_remainder (n : ℕ) : 
  (n > 0 ∧ ∀ m : ℕ, m > 0 → m < n → (3 * m + 45) % 1060 ≠ 16) →
  (3 * n + 45) % 1060 = 16 →
  (18 * n + 17) % 1920 = 1043 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_remainder_l2674_267400


namespace NUMINAMATH_CALUDE_unique_triple_solution_l2674_267434

theorem unique_triple_solution (a b c : ℝ) : 
  a > 2 ∧ b > 2 ∧ c > 2 ∧ 
  ((a + 3)^2) / (b + c - 3) + ((b + 5)^2) / (c + a - 5) + ((c + 7)^2) / (a + b - 7) = 45 →
  a = 13 ∧ b = 11 ∧ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l2674_267434


namespace NUMINAMATH_CALUDE_potato_sale_revenue_l2674_267406

-- Define the given constants
def total_weight : ℕ := 6500
def damaged_weight : ℕ := 150
def bag_weight : ℕ := 50
def price_per_bag : ℕ := 72

-- Define the theorem
theorem potato_sale_revenue : 
  (((total_weight - damaged_weight) / bag_weight) * price_per_bag = 9144) := by
  sorry


end NUMINAMATH_CALUDE_potato_sale_revenue_l2674_267406


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l2674_267465

theorem twenty_five_percent_less_than_80 (x : ℝ) : x + (1/3) * x = 60 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l2674_267465


namespace NUMINAMATH_CALUDE_pizza_toppings_l2674_267455

theorem pizza_toppings (total_slices cheese_slices olive_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : cheese_slices = 16)
  (h3 : olive_slices = 18)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ cheese_slices ∨ slice ≤ olive_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = 10 ∧ 
    cheese_slices + olive_slices - both_toppings = total_slices :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l2674_267455


namespace NUMINAMATH_CALUDE_x_value_and_n_bound_l2674_267490

theorem x_value_and_n_bound (x n : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + n < 4) : 
  x = 1 ∧ n < 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_and_n_bound_l2674_267490


namespace NUMINAMATH_CALUDE_floor_sum_of_squares_and_product_l2674_267479

theorem floor_sum_of_squares_and_product (p q r s : ℝ) : 
  0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s →
  p^2 + q^2 = 2500 →
  r^2 + s^2 = 2500 →
  p * q = 1152 →
  r * s = 1152 →
  ⌊p + q + r + s⌋ = 138 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_of_squares_and_product_l2674_267479


namespace NUMINAMATH_CALUDE_tangerine_tree_count_prove_tangerine_tree_count_l2674_267451

theorem tangerine_tree_count : ℕ → ℕ → ℕ → Prop :=
  fun pear_trees apple_trees tangerine_trees =>
    (pear_trees = 56) →
    (pear_trees = apple_trees + 18) →
    (tangerine_trees = apple_trees - 12) →
    (tangerine_trees = 26)

-- Proof
theorem prove_tangerine_tree_count :
  ∃ (pear_trees apple_trees tangerine_trees : ℕ),
    tangerine_tree_count pear_trees apple_trees tangerine_trees :=
by
  sorry

end NUMINAMATH_CALUDE_tangerine_tree_count_prove_tangerine_tree_count_l2674_267451


namespace NUMINAMATH_CALUDE_ellipse_properties_l2674_267428

noncomputable def ellipse_C (x y a b : ℝ) : Prop :=
  (y^2 / a^2) + (x^2 / b^2) = 1

theorem ellipse_properties (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0)
  (h3 : (a^2 - b^2) / a^2 = 6/9)
  (h4 : ellipse_C (2*Real.sqrt 2/3) (Real.sqrt 3/3) a b) :
  (∃ (x y : ℝ), ellipse_C x y 1 (Real.sqrt 3)) ∧
  (∃ (S : ℝ → ℝ → ℝ), 
    (∀ A B : ℝ × ℝ, 
      ellipse_C A.1 A.2 1 (Real.sqrt 3) → 
      ellipse_C B.1 B.2 1 (Real.sqrt 3) → 
      (∃ m : ℝ, A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2) → 
      S A.1 A.2 ≤ Real.sqrt 3 / 2) ∧
    (∃ A B : ℝ × ℝ, 
      ellipse_C A.1 A.2 1 (Real.sqrt 3) ∧ 
      ellipse_C B.1 B.2 1 (Real.sqrt 3) ∧ 
      (∃ m : ℝ, A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2) ∧ 
      S A.1 A.2 = Real.sqrt 3 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2674_267428


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l2674_267429

theorem exactly_one_greater_than_one 
  (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (prod_one : a * b * c = 1)
  (sum_greater : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l2674_267429


namespace NUMINAMATH_CALUDE_numeric_methods_count_l2674_267452

/-- The number of second-year students studying numeric methods -/
def numeric_methods_students : ℕ := 225

/-- The number of second-year students studying automatic control of airborne vehicles -/
def automatic_control_students : ℕ := 450

/-- The number of second-year students studying both subjects -/
def both_subjects_students : ℕ := 134

/-- The total number of students in the faculty -/
def total_students : ℕ := 676

/-- The approximate percentage of second-year students -/
def second_year_percentage : ℚ := 80 / 100

/-- The total number of second-year students -/
def total_second_year_students : ℕ := 541

theorem numeric_methods_count : 
  numeric_methods_students = 
    total_second_year_students + both_subjects_students - automatic_control_students :=
by sorry

end NUMINAMATH_CALUDE_numeric_methods_count_l2674_267452


namespace NUMINAMATH_CALUDE_principal_calculation_l2674_267471

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement -/
theorem principal_calculation (sum : ℝ) (rate : ℝ) (time : ℕ) 
  (h_sum : sum = 3969)
  (h_rate : rate = 0.05)
  (h_time : time = 2) :
  ∃ (principal : ℝ), 
    compound_interest principal rate time = sum ∧ 
    principal = 3600 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2674_267471


namespace NUMINAMATH_CALUDE_walter_age_2005_conditions_hold_l2674_267439

-- Define Walter's age in 2000
def walter_age_2000 : ℚ := 4 / 3

-- Define grandmother's age in 2000
def grandmother_age_2000 : ℚ := 2 * walter_age_2000

-- Define the current year
def current_year : ℕ := 2000

-- Define the target year
def target_year : ℕ := 2005

-- Define the sum of birth years
def sum_birth_years : ℕ := 4004

-- Theorem statement
theorem walter_age_2005 :
  (walter_age_2000 + (target_year - current_year : ℚ)) = 19 / 3 :=
by
  sorry

-- Verify the conditions
theorem conditions_hold :
  (walter_age_2000 = grandmother_age_2000 / 2) ∧
  (current_year - walter_age_2000 + current_year - grandmother_age_2000 = sum_birth_years) :=
by
  sorry

end NUMINAMATH_CALUDE_walter_age_2005_conditions_hold_l2674_267439


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l2674_267492

/-- Given a line passing through points (0, -4) and (4, 4), prove that the product of its slope and y-intercept equals -8. -/
theorem line_slope_intercept_product : 
  ∀ (m b : ℝ), 
  (∀ x y : ℝ, y = m * x + b → (x = 0 ∧ y = -4) ∨ (x = 4 ∧ y = 4)) → 
  m * b = -8 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l2674_267492


namespace NUMINAMATH_CALUDE_multiple_properties_l2674_267476

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a - b = 4 * p) ∧ 
  (∃ q : ℤ, a - b = 2 * q) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l2674_267476


namespace NUMINAMATH_CALUDE_expression_equality_l2674_267426

theorem expression_equality : 
  (3 / Real.sqrt 3) - (Real.sqrt 3)^2 - Real.sqrt 27 + |Real.sqrt 3 - 2| = -1 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2674_267426


namespace NUMINAMATH_CALUDE_y_minus_3x_equals_7_l2674_267411

theorem y_minus_3x_equals_7 (x y : ℝ) (h1 : x + y = 8) (h2 : y - x = 7.5) : y - 3 * x = 7 := by
  sorry

end NUMINAMATH_CALUDE_y_minus_3x_equals_7_l2674_267411


namespace NUMINAMATH_CALUDE_kingfisher_to_warbler_ratio_l2674_267499

/-- Represents the composition of bird species in the Goshawk-Eurasian Nature Reserve -/
structure BirdPopulation where
  hawks : ℝ
  paddyfieldWarblers : ℝ
  kingfishers : ℝ
  others : ℝ

/-- The conditions of the bird population in the nature reserve -/
def validBirdPopulation (bp : BirdPopulation) : Prop :=
  bp.hawks = 0.3 ∧
  bp.paddyfieldWarblers = 0.4 * (1 - bp.hawks) ∧
  bp.others = 0.35 ∧
  bp.hawks + bp.paddyfieldWarblers + bp.kingfishers + bp.others = 1

/-- The theorem stating the relationship between kingfishers and paddyfield-warblers -/
theorem kingfisher_to_warbler_ratio (bp : BirdPopulation) 
  (h : validBirdPopulation bp) : 
  bp.kingfishers / bp.paddyfieldWarblers = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_kingfisher_to_warbler_ratio_l2674_267499


namespace NUMINAMATH_CALUDE_gcd_7384_12873_l2674_267477

theorem gcd_7384_12873 : Nat.gcd 7384 12873 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7384_12873_l2674_267477


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2674_267404

-- Define the propositions
def p : Prop := (m : ℝ) → m = -1

def q (m : ℝ) : Prop := 
  let line1 : ℝ → ℝ → Prop := λ x y => x - y = 0
  let line2 : ℝ → ℝ → Prop := λ x y => x + m^2 * y = 0
  ∀ x1 y1 x2 y2, line1 x1 y1 → line2 x2 y2 → 
    (x2 - x1) * (y2 - y1) + (x2 - x1) * (x2 - x1) = 0

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, p → q m) ∧ (∃ m : ℝ, q m ∧ ¬p) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2674_267404


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2674_267435

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: For a geometric sequence {a_n}, if a_2 * a_4 = 1/2, then a_1 * a_3^2 * a_5 = 1/4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : geometric_sequence a) (h_cond : a 2 * a 4 = 1/2) : 
    a 1 * (a 3)^2 * a 5 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2674_267435


namespace NUMINAMATH_CALUDE_unique_solution_l2674_267480

/-- Represents a four-digit number as individual digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- Converts a four-digit number to its numerical value -/
def to_nat (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Converts a two-digit number to its numerical value -/
def two_digit_to_nat (a b : Nat) : Nat :=
  10 * a + b

/-- States that A̅B² = A̅CDB -/
def condition1 (n : FourDigitNumber) : Prop :=
  (two_digit_to_nat n.a n.b)^2 = to_nat n

/-- States that C̅D³ = A̅CBD -/
def condition2 (n : FourDigitNumber) : Prop :=
  (two_digit_to_nat n.c n.d)^3 = 1000 * n.a + 100 * n.c + 10 * n.b + n.d

/-- The main theorem stating that the only solution is A = 9, B = 6, C = 2, D = 1 -/
theorem unique_solution :
  ∀ n : FourDigitNumber, condition1 n ∧ condition2 n →
  n.a = 9 ∧ n.b = 6 ∧ n.c = 2 ∧ n.d = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2674_267480


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l2674_267459

/-- Given a block with houses and junk mail to distribute, calculate the number of pieces per house -/
def junk_mail_per_house (num_houses : ℕ) (total_junk_mail : ℕ) : ℕ :=
  total_junk_mail / num_houses

/-- Theorem: For a block with 6 houses and 24 pieces of junk mail, each house receives 4 pieces -/
theorem junk_mail_distribution :
  junk_mail_per_house 6 24 = 4 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l2674_267459


namespace NUMINAMATH_CALUDE_equation_solution_l2674_267423

theorem equation_solution :
  ∃ x : ℝ, (5 + 3.5 * x = 2 * x - 25) ∧ (x = -20) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2674_267423


namespace NUMINAMATH_CALUDE_gcd_1029_1437_5649_l2674_267441

theorem gcd_1029_1437_5649 : Nat.gcd 1029 (Nat.gcd 1437 5649) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1029_1437_5649_l2674_267441


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l2674_267450

theorem product_mod_seventeen :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 14 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l2674_267450


namespace NUMINAMATH_CALUDE_irreducible_fraction_l2674_267417

theorem irreducible_fraction (a b c d : ℤ) (h : a * d - b * c = 1) :
  ¬∃ (m : ℤ), m > 1 ∧ m ∣ (a^2 + b^2) ∧ m ∣ (a * c + b * d) := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l2674_267417


namespace NUMINAMATH_CALUDE_symmetry_condition_l2674_267489

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

theorem symmetry_condition (m : ℝ) :
  (∀ x, f m (2 - x) = f m x) ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_condition_l2674_267489


namespace NUMINAMATH_CALUDE_inscribed_triangle_radius_l2674_267416

theorem inscribed_triangle_radius 
  (S : ℝ) 
  (α : ℝ) 
  (h1 : S > 0) 
  (h2 : 0 < α ∧ α < 2 * Real.pi) : 
  ∃ R : ℝ, R > 0 ∧ 
    R = (Real.sqrt (S * Real.sqrt 3)) / (2 * (Real.sin (α / 4))^2) :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_radius_l2674_267416


namespace NUMINAMATH_CALUDE_cleaning_time_with_help_l2674_267491

-- Define the grove dimensions
def trees_width : ℕ := 4
def trees_height : ℕ := 5

-- Define the initial cleaning time per tree
def initial_cleaning_time : ℕ := 6

-- Define the helper effect (halves the cleaning time)
def helper_effect : ℚ := 1/2

-- Theorem to prove
theorem cleaning_time_with_help :
  let total_trees := trees_width * trees_height
  let cleaning_time_with_help := initial_cleaning_time * helper_effect
  let total_cleaning_time := (total_trees : ℚ) * cleaning_time_with_help
  total_cleaning_time / 60 = 1 := by sorry

end NUMINAMATH_CALUDE_cleaning_time_with_help_l2674_267491


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l2674_267464

/-- The perimeter of a hexagon with side length 5 inches is 30 inches. -/
theorem hexagon_perimeter (side_length : ℝ) (h : side_length = 5) : 
  6 * side_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l2674_267464


namespace NUMINAMATH_CALUDE_remaining_distance_l2674_267420

theorem remaining_distance (total_distance : ℝ) (father_fraction : ℝ) (mother_fraction : ℝ) :
  total_distance = 240 →
  father_fraction = 1/2 →
  mother_fraction = 3/8 →
  total_distance * (1 - father_fraction - mother_fraction) = 30 := by
sorry

end NUMINAMATH_CALUDE_remaining_distance_l2674_267420


namespace NUMINAMATH_CALUDE_luncheon_absence_l2674_267427

/-- The number of people who didn't show up to a luncheon --/
def people_absent (invited : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) : ℕ :=
  invited - (table_capacity * tables_needed)

/-- Proof that 50 people didn't show up to the luncheon --/
theorem luncheon_absence : people_absent 68 3 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_absence_l2674_267427


namespace NUMINAMATH_CALUDE_expression_evaluation_l2674_267440

theorem expression_evaluation : 
  (123 - (45 * (9 - 6) - 78)) + (0 / 1994) = 66 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2674_267440


namespace NUMINAMATH_CALUDE_difference_of_squares_535_465_l2674_267445

theorem difference_of_squares_535_465 : 535^2 - 465^2 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_535_465_l2674_267445


namespace NUMINAMATH_CALUDE_circles_intersect_l2674_267472

/-- The circles x^2 + y^2 + 4x - 4y - 8 = 0 and x^2 + y^2 - 2x + 4y + 1 = 0 intersect. -/
theorem circles_intersect : ∃ (x y : ℝ),
  (x^2 + y^2 + 4*x - 4*y - 8 = 0) ∧ (x^2 + y^2 - 2*x + 4*y + 1 = 0) := by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_l2674_267472


namespace NUMINAMATH_CALUDE_line_and_symmetric_point_l2674_267446

/-- A line with inclination angle 135° passing through (1, 1) -/
structure Line :=
  (equation : ℝ → ℝ → Prop)
  (passes_through : equation 1 1)
  (inclination : Real.tan (135 * π / 180) = -1)

/-- The symmetric point of A with respect to a line -/
def symmetric_point (l : Line) (A : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem line_and_symmetric_point (l : Line) :
  l.equation = fun x y ↦ x + y - 2 = 0 ∧
  symmetric_point l (3, 4) = (-2, -1) :=
sorry

end NUMINAMATH_CALUDE_line_and_symmetric_point_l2674_267446


namespace NUMINAMATH_CALUDE_arcsin_sin_eq_solution_l2674_267460

theorem arcsin_sin_eq_solution (x : ℝ) : 
  Real.arcsin (Real.sin x) = (3 * x) / 4 ∧ 
  -(π / 2) ≤ (3 * x) / 4 ∧ 
  (3 * x) / 4 ≤ π / 2 → 
  x = 0 := by
sorry

end NUMINAMATH_CALUDE_arcsin_sin_eq_solution_l2674_267460


namespace NUMINAMATH_CALUDE_certain_number_proof_l2674_267484

theorem certain_number_proof : 
  ∃ x : ℝ, 0.8 * x = (4 / 5) * 25 + 16 ∧ x = 45 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2674_267484


namespace NUMINAMATH_CALUDE_wall_width_l2674_267401

theorem wall_width (area : ℝ) (height : ℝ) (width : ℝ) :
  area = 8 ∧ height = 4 ∧ area = width * height → width = 2 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_l2674_267401


namespace NUMINAMATH_CALUDE_min_sum_squares_l2674_267488

-- Define the points A, B, C, D, E as real numbers representing their positions on a line
def A : ℝ := 0
def B : ℝ := 1
def C : ℝ := 3
def D : ℝ := 6
def E : ℝ := 10

-- Define the function to be minimized
def f (x : ℝ) : ℝ := (x - A)^2 + (x - B)^2 + (x - C)^2 + (x - D)^2 + (x - E)^2

-- State the theorem
theorem min_sum_squares :
  ∃ (min : ℝ), min = 60 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2674_267488


namespace NUMINAMATH_CALUDE_work_completion_time_B_l2674_267470

theorem work_completion_time_B (a b : ℝ) : 
  (a + b = 1/6) →  -- A and B together complete 1/6 of the work in one day
  (a = 1/11) →     -- A alone completes 1/11 of the work in one day
  (b = 5/66) →     -- B alone completes 5/66 of the work in one day
  (1/b = 66/5) :=  -- The time B takes to complete the work alone is 66/5 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_B_l2674_267470


namespace NUMINAMATH_CALUDE_average_cost_is_11_cents_l2674_267408

/-- Calculates the average cost per marker in cents, rounded to the nearest whole number -/
def average_cost_per_marker (num_markers : ℕ) (marker_price : ℚ) (shipping_price : ℚ) : ℕ :=
  let total_cost_cents := (marker_price + shipping_price) * 100
  let avg_cost_cents := total_cost_cents / num_markers
  (avg_cost_cents + 1/2).floor.toNat

/-- Proves that for 300 markers at $25.50 with $8.50 shipping, the average cost is 11 cents -/
theorem average_cost_is_11_cents :
  average_cost_per_marker 300 (25.5) (8.5) = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_cost_is_11_cents_l2674_267408


namespace NUMINAMATH_CALUDE_jellybean_probability_l2674_267449

/-- Represents the number of ways to choose k items from n items --/
def binomial (n k : ℕ) : ℕ := sorry

/-- The probability of an event given the number of favorable outcomes and total outcomes --/
def probability (favorable total : ℕ) : ℚ := sorry

theorem jellybean_probability :
  let total_jellybeans : ℕ := 15
  let green_jellybeans : ℕ := 6
  let purple_jellybeans : ℕ := 2
  let yellow_jellybeans : ℕ := 7
  let picked_jellybeans : ℕ := 4

  let total_outcomes : ℕ := binomial total_jellybeans picked_jellybeans
  let yellow_combinations : ℕ := binomial yellow_jellybeans 2
  let non_yellow_combinations : ℕ := binomial (green_jellybeans + purple_jellybeans) 2
  let favorable_outcomes : ℕ := yellow_combinations * non_yellow_combinations

  probability favorable_outcomes total_outcomes = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2674_267449


namespace NUMINAMATH_CALUDE_grandma_olga_grandchildren_l2674_267424

/-- Represents the number of grandchildren Grandma Olga has -/
def total_grandchildren (num_daughters num_sons : ℕ) (sons_per_daughter daughters_per_son : ℕ) : ℕ :=
  num_daughters * sons_per_daughter + num_sons * daughters_per_son

/-- Proves that Grandma Olga has 33 grandchildren given the specified conditions -/
theorem grandma_olga_grandchildren :
  total_grandchildren 3 3 6 5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_grandma_olga_grandchildren_l2674_267424


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2674_267498

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + 2 * x^4 + 5 * x^2 + 16) - (x^6 + 4 * x^5 - 2 * x^3 + 3 * x^2 + 18) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 + 2 * x^2 - 2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2674_267498


namespace NUMINAMATH_CALUDE_olivia_payment_l2674_267436

/-- Represents the number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- Represents the number of quarters Olivia pays for chips -/
def quarters_for_chips : ℕ := 4

/-- Represents the number of quarters Olivia pays for soda -/
def quarters_for_soda : ℕ := 12

/-- Calculates the total amount Olivia pays in dollars -/
def total_paid : ℚ :=
  (quarters_for_chips + quarters_for_soda) / quarters_per_dollar

theorem olivia_payment :
  total_paid = 4 := by sorry

end NUMINAMATH_CALUDE_olivia_payment_l2674_267436


namespace NUMINAMATH_CALUDE_prime_combinations_theorem_l2674_267483

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def all_combinations_prime (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → is_prime (10^k * 7 + (10^n - 1) / 9 - 10^k)

theorem prime_combinations_theorem :
  ∀ n : ℕ, (all_combinations_prime n ↔ n = 1 ∨ n = 2) :=
sorry

end NUMINAMATH_CALUDE_prime_combinations_theorem_l2674_267483


namespace NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l2674_267495

/-- Calculates the cost per quart of ratatouille given ingredient quantities and prices -/
theorem ratatouille_cost_per_quart :
  let eggplant_oz : Real := 88
  let eggplant_price : Real := 0.22
  let zucchini_oz : Real := 60.8
  let zucchini_price : Real := 0.15
  let tomato_oz : Real := 73.6
  let tomato_price : Real := 0.25
  let onion_oz : Real := 43.2
  let onion_price : Real := 0.07
  let basil_oz : Real := 16
  let basil_price : Real := 2.70 / 4
  let bell_pepper_oz : Real := 12
  let bell_pepper_price : Real := 0.20
  let total_yield_quarts : Real := 4.5
  let total_cost : Real := 
    eggplant_oz * eggplant_price +
    zucchini_oz * zucchini_price +
    tomato_oz * tomato_price +
    onion_oz * onion_price +
    basil_oz * basil_price +
    bell_pepper_oz * bell_pepper_price
  let cost_per_quart : Real := total_cost / total_yield_quarts
  cost_per_quart = 14.02 := by
  sorry

end NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l2674_267495


namespace NUMINAMATH_CALUDE_ceil_e_plus_pi_l2674_267433

theorem ceil_e_plus_pi : ⌈Real.exp 1 + Real.pi⌉ = 6 := by sorry

end NUMINAMATH_CALUDE_ceil_e_plus_pi_l2674_267433
