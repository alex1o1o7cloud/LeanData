import Mathlib

namespace NUMINAMATH_CALUDE_friday_13th_more_frequent_l747_74770

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year in the Gregorian calendar -/
structure GregorianYear where
  year : Nat

/-- Determines if a given year is a leap year -/
def isLeapYear (y : GregorianYear) : Bool :=
  (y.year % 4 == 0 && y.year % 100 != 0) || y.year % 400 == 0

/-- Calculates the day of the week for the 13th of a given month in a given year -/
def dayOf13th (y : GregorianYear) (month : Nat) : DayOfWeek :=
  sorry

/-- Counts the frequency of each day of the week being the 13th over a 400-year cycle -/
def countDayOf13thIn400Years : DayOfWeek → Nat :=
  sorry

/-- Theorem: The 13th is more likely to be a Friday than any other day of the week -/
theorem friday_13th_more_frequent :
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Friday →
    countDayOf13thIn400Years DayOfWeek.Friday > countDayOf13thIn400Years d :=
  sorry

end NUMINAMATH_CALUDE_friday_13th_more_frequent_l747_74770


namespace NUMINAMATH_CALUDE_cycling_jogging_swimming_rates_l747_74745

theorem cycling_jogging_swimming_rates : ∃ (b j s : ℕ), 
  (3 * b + 2 * j + 4 * s = 66) ∧ 
  (3 * j + 2 * s + 4 * b = 96) ∧ 
  (b^2 + j^2 + s^2 = 612) := by
  sorry

end NUMINAMATH_CALUDE_cycling_jogging_swimming_rates_l747_74745


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l747_74780

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (3 + I) / (2 - I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l747_74780


namespace NUMINAMATH_CALUDE_fifteen_percent_problem_l747_74718

theorem fifteen_percent_problem : ∃ x : ℝ, (15 / 100) * x = 90 ∧ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_problem_l747_74718


namespace NUMINAMATH_CALUDE_work_distribution_l747_74716

theorem work_distribution (p : ℕ) (x : ℚ) (h1 : 0 < p) (h2 : 0 ≤ x) (h3 : x < 1) :
  p * 1 = (1 - x) * p * (3/2) → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_work_distribution_l747_74716


namespace NUMINAMATH_CALUDE_ryan_reads_more_pages_l747_74721

/-- Given that Ryan read 2100 pages in 7 days and his brother read 200 pages per day for 7 days,
    prove that Ryan read 100 more pages per day on average compared to his brother. -/
theorem ryan_reads_more_pages (ryan_total_pages : ℕ) (days : ℕ) (brother_daily_pages : ℕ)
    (h1 : ryan_total_pages = 2100)
    (h2 : days = 7)
    (h3 : brother_daily_pages = 200) :
    ryan_total_pages / days - brother_daily_pages = 100 := by
  sorry

end NUMINAMATH_CALUDE_ryan_reads_more_pages_l747_74721


namespace NUMINAMATH_CALUDE_probability_theorem_l747_74784

def num_forks : ℕ := 8
def num_spoons : ℕ := 5
def num_knives : ℕ := 7
def total_pieces : ℕ := num_forks + num_spoons + num_knives
def num_selected : ℕ := 4

def probability_two_forks_one_spoon_one_knife : ℚ :=
  (Nat.choose num_forks 2 * Nat.choose num_spoons 1 * Nat.choose num_knives 1 : ℚ) /
  Nat.choose total_pieces num_selected

theorem probability_theorem :
  probability_two_forks_one_spoon_one_knife = 196 / 969 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l747_74784


namespace NUMINAMATH_CALUDE_central_angle_of_chord_l747_74759

theorem central_angle_of_chord (α : Real) (chord_length : Real) :
  (∀ R, R = 1 → chord_length = Real.sqrt 3 → 2 * Real.sin (α / 2) = chord_length) →
  α = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_of_chord_l747_74759


namespace NUMINAMATH_CALUDE_intersection_implies_m_range_l747_74724

/-- The set M defined by the equation 3x^2 + 4y^2 - 6mx + 3m^2 - 12 = 0 -/
def M (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1^2 + 4 * p.2^2 - 6 * m * p.1 + 3 * m^2 - 12 = 0}

/-- The set N defined by the equation 2y^2 - 12x + 9 = 0 -/
def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.2^2 - 12 * p.1 + 9 = 0}

/-- Theorem stating that if M and N have a non-empty intersection,
    then m is in the range [-5/4, 11/4] -/
theorem intersection_implies_m_range :
  ∀ m : ℝ, (M m ∩ N).Nonempty → -5/4 ≤ m ∧ m ≤ 11/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_range_l747_74724


namespace NUMINAMATH_CALUDE_prime_sequence_ones_digit_l747_74794

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) :
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
  p > 10 ∧
  q = p + 10 ∧ r = q + 10 ∧ s = r + 10 →
  ones_digit p = 1 :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_ones_digit_l747_74794


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l747_74766

/-- Represents the systematic sampling problem -/
def SystematicSampling (total_population : ℕ) (sample_size : ℕ) (first_number : ℕ) (threshold : ℕ) : Prop :=
  let interval := total_population / sample_size
  let selected_numbers := List.range sample_size |>.map (fun i => first_number + i * interval)
  (selected_numbers.filter (fun n => n > threshold)).length = 8

/-- Theorem stating the result of the systematic sampling problem -/
theorem systematic_sampling_result : 
  SystematicSampling 960 32 9 750 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l747_74766


namespace NUMINAMATH_CALUDE_smallest_common_multiple_10_15_gt_100_l747_74702

theorem smallest_common_multiple_10_15_gt_100 : ∃ (n : ℕ), n > 100 ∧ n.lcm 10 = n ∧ n.lcm 15 = n ∧ ∀ (m : ℕ), m > 100 ∧ m.lcm 10 = m ∧ m.lcm 15 = m → n ≤ m :=
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_10_15_gt_100_l747_74702


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_discriminants_l747_74797

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ :=
  p.b ^ 2 - 4 * p.a * p.c

/-- A quadratic polynomial has zero discriminant -/
def hasZeroDiscriminant (p : QuadraticPolynomial) : Prop :=
  discriminant p = 0

/-- The sum of two quadratic polynomials -/
def sumQuadratic (p q : QuadraticPolynomial) : QuadraticPolynomial where
  a := p.a + q.a
  b := p.b + q.b
  c := p.c + q.c

theorem quadratic_sum_zero_discriminants :
  ∀ (p : QuadraticPolynomial),
  ∃ (q r : QuadraticPolynomial),
  hasZeroDiscriminant q ∧
  hasZeroDiscriminant r ∧
  p = sumQuadratic q r :=
sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_discriminants_l747_74797


namespace NUMINAMATH_CALUDE_parking_lot_rows_parking_lot_rows_example_l747_74713

/-- Given a parking lot with the following properties:
    - A car is 5th from the right and 4th from the left in a row
    - The parking lot has 10 floors
    - There are 1600 cars in total
    The number of rows on each floor is 20. -/
theorem parking_lot_rows (car_position_right : Nat) (car_position_left : Nat)
                         (num_floors : Nat) (total_cars : Nat) : Nat :=
  let cars_in_row := car_position_right + car_position_left - 1
  let cars_per_floor := total_cars / num_floors
  cars_per_floor / cars_in_row

#check parking_lot_rows 5 4 10 1600 = 20

/-- The parking_lot_rows theorem holds for the given values. -/
theorem parking_lot_rows_example : parking_lot_rows 5 4 10 1600 = 20 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_rows_parking_lot_rows_example_l747_74713


namespace NUMINAMATH_CALUDE_initial_stops_l747_74755

/-- Represents the number of stops made by a delivery driver -/
structure DeliveryStops where
  total : Nat
  after_initial : Nat
  initial : Nat

/-- Theorem stating the number of initial stops given the total and after-initial stops -/
theorem initial_stops (d : DeliveryStops) 
  (h1 : d.total = 7) 
  (h2 : d.after_initial = 4) 
  (h3 : d.total = d.initial + d.after_initial) : 
  d.initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_stops_l747_74755


namespace NUMINAMATH_CALUDE_expected_heads_value_l747_74732

/-- The number of coins -/
def n : ℕ := 64

/-- The probability of getting heads on a single fair coin toss -/
def p : ℚ := 1/2

/-- The probability of getting heads after up to three tosses -/
def prob_heads : ℚ := p + (1 - p) * p + (1 - p) * (1 - p) * p

/-- The expected number of coins showing heads after the process -/
def expected_heads : ℚ := n * prob_heads

theorem expected_heads_value : expected_heads = 56 := by sorry

end NUMINAMATH_CALUDE_expected_heads_value_l747_74732


namespace NUMINAMATH_CALUDE_order_combinations_l747_74756

theorem order_combinations (num_drinks num_salads num_pizzas : ℕ) 
  (h1 : num_drinks = 3)
  (h2 : num_salads = 2)
  (h3 : num_pizzas = 5) :
  num_drinks * num_salads * num_pizzas = 30 := by
  sorry

end NUMINAMATH_CALUDE_order_combinations_l747_74756


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l747_74723

theorem volume_of_inscribed_sphere (edge_length : ℝ) (h : edge_length = 6) :
  let radius : ℝ := edge_length / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * radius ^ 3
  sphere_volume = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l747_74723


namespace NUMINAMATH_CALUDE_apples_left_theorem_l747_74776

def num_baskets : ℕ := 11
def num_children : ℕ := 10
def initial_apples : ℕ := 1000

def apples_picked (basket : ℕ) : ℕ := basket * num_children

def total_apples_picked : ℕ := (List.range num_baskets).map (λ i => apples_picked (i + 1)) |>.sum

theorem apples_left_theorem :
  initial_apples - total_apples_picked = 340 := by sorry

end NUMINAMATH_CALUDE_apples_left_theorem_l747_74776


namespace NUMINAMATH_CALUDE_gcd_triple_existence_l747_74765

theorem gcd_triple_existence (S : Set ℕ+) (hS_infinite : Set.Infinite S)
  (a b c d : ℕ+) (hab : a ∈ S) (hbc : b ∈ S) (hcd : c ∈ S) (hda : d ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (hgcd : Nat.gcd a.val b.val ≠ Nat.gcd c.val d.val) :
  ∃ x y z : ℕ+, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Nat.gcd x.val y.val = Nat.gcd y.val z.val ∧
    Nat.gcd y.val z.val ≠ Nat.gcd z.val x.val :=
by sorry

end NUMINAMATH_CALUDE_gcd_triple_existence_l747_74765


namespace NUMINAMATH_CALUDE_essay_introduction_length_l747_74789

theorem essay_introduction_length 
  (total_words : ℕ) 
  (body_section_words : ℕ) 
  (body_section_count : ℕ) 
  (h1 : total_words = 5000)
  (h2 : body_section_words = 800)
  (h3 : body_section_count = 4) :
  ∃ (intro_words : ℕ),
    intro_words = 450 ∧ 
    total_words = intro_words + (body_section_count * body_section_words) + (3 * intro_words) :=
by sorry

end NUMINAMATH_CALUDE_essay_introduction_length_l747_74789


namespace NUMINAMATH_CALUDE_flower_bed_total_l747_74788

theorem flower_bed_total (tulips carnations : ℕ) : 
  tulips = 3 → carnations = 4 → tulips + carnations = 7 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_total_l747_74788


namespace NUMINAMATH_CALUDE_sum_of_segments_l747_74752

/-- Given a number line with points P at 3 and V at 33, and the line between them
    divided into six equal parts, the sum of the lengths of PS and TV is 25. -/
theorem sum_of_segments (P V Q R S T U : ℝ) : 
  P = 3 → V = 33 → 
  Q - P = R - Q → R - Q = S - R → S - R = T - S → T - S = U - T → U - T = V - U →
  (S - P) + (V - T) = 25 := by sorry

end NUMINAMATH_CALUDE_sum_of_segments_l747_74752


namespace NUMINAMATH_CALUDE_slope_relationship_l747_74772

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Definition of point A₁ -/
def A₁ : ℝ × ℝ := (-2, 0)

/-- Definition of point A₂ -/
def A₂ : ℝ × ℝ := (2, 0)

/-- Definition of the line PQ -/
def line_PQ (x y : ℝ) : Prop := ∃ m : ℝ, x = m * y + 1/2

/-- Theorem stating the relationship between slopes -/
theorem slope_relationship (P Q : ℝ × ℝ) :
  ellipse_C P.1 P.2 →
  ellipse_C Q.1 Q.2 →
  line_PQ P.1 P.2 →
  line_PQ Q.1 Q.2 →
  P ≠ A₁ →
  P ≠ A₂ →
  Q ≠ A₁ →
  Q ≠ A₂ →
  (P.2 - A₁.2) / (P.1 - A₁.1) = 3/5 * (Q.2 - A₂.2) / (Q.1 - A₂.1) :=
sorry

end NUMINAMATH_CALUDE_slope_relationship_l747_74772


namespace NUMINAMATH_CALUDE_min_value_xy_l747_74760

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y + 12 = x * y) :
  x * y ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_l747_74760


namespace NUMINAMATH_CALUDE_system_solution_l747_74751

theorem system_solution (a b c : ℝ) 
  (eq1 : b + c = 15 - 2*a)
  (eq2 : a + c = -10 - 4*b)
  (eq3 : a + b = 8 - 2*c) :
  3*a + 3*b + 3*c = 39/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l747_74751


namespace NUMINAMATH_CALUDE_problem_3_l747_74762

theorem problem_3 (a : ℝ) : a = 1 / (Real.sqrt 5 - 2) → 2 * a^2 - 8 * a + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_3_l747_74762


namespace NUMINAMATH_CALUDE_unique_equidistant_cell_l747_74734

-- Define the distance function for cells on an infinite chessboard
def distance (a b : ℤ × ℤ) : ℕ :=
  max (Int.natAbs (a.1 - b.1)) (Int.natAbs (a.2 - b.2))

-- Define the theorem
theorem unique_equidistant_cell
  (A B C : ℤ × ℤ)
  (hab : distance A B = 100)
  (hac : distance A C = 100)
  (hbc : distance B C = 100) :
  ∃! X : ℤ × ℤ, distance X A = 50 ∧ distance X B = 50 ∧ distance X C = 50 :=
sorry

end NUMINAMATH_CALUDE_unique_equidistant_cell_l747_74734


namespace NUMINAMATH_CALUDE_odd_function_interval_l747_74709

/-- A function f is odd on an interval [a, b] if the interval is symmetric about the origin -/
def is_odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a = -b ∧ ∀ x ∈ Set.Icc a b, f (-x) = -f x

/-- The theorem stating that if f is an odd function on [t, t^2 - 3t - 3], then t = -1 -/
theorem odd_function_interval (f : ℝ → ℝ) (t : ℝ) :
  is_odd_on_interval f t (t^2 - 3*t - 3) → t = -1 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_interval_l747_74709


namespace NUMINAMATH_CALUDE_range_of_m_l747_74777

/-- The line x - 2y + 3 = 0 and the parabola y² = mx (m ≠ 0) have no points of intersection -/
def p (m : ℝ) : Prop :=
  ∀ x y : ℝ, x - 2*y + 3 = 0 → y^2 = m*x → m ≠ 0 → False

/-- The equation x²/(5-2m) + y²/m = 1 represents a hyperbola -/
def q (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2/(5-2*m) + y^2/m = 1 ∧ m*(5-2*m) < 0

theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) →
    m ≥ 3 ∨ m < 0 ∨ (0 < m ∧ m ≤ 5/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l747_74777


namespace NUMINAMATH_CALUDE_no_natural_square_diff_2018_l747_74744

theorem no_natural_square_diff_2018 : ¬∃ (a b : ℕ), a^2 - b^2 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_diff_2018_l747_74744


namespace NUMINAMATH_CALUDE_shaded_area_in_circle_l747_74773

/-- The area of a specific shaded region in a circle -/
theorem shaded_area_in_circle (r : ℝ) (h : r = 5) :
  let circle_area := π * r^2
  let triangle_area := r^2 / 2
  let sector_area := circle_area / 4
  2 * triangle_area + 2 * sector_area = 25 + 25 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_in_circle_l747_74773


namespace NUMINAMATH_CALUDE_square_number_plus_minus_five_is_square_l747_74726

theorem square_number_plus_minus_five_is_square : ∃ (n : ℕ), 
  (∃ (a : ℕ), n = a^2) ∧ 
  (∃ (b : ℕ), n + 5 = b^2) ∧ 
  (∃ (c : ℕ), n - 5 = c^2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_square_number_plus_minus_five_is_square_l747_74726


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_l747_74754

theorem cube_roots_of_unity :
  let z₁ : ℂ := 1
  let z₂ : ℂ := -1/2 + Complex.I * Real.sqrt 3 / 2
  let z₃ : ℂ := -1/2 - Complex.I * Real.sqrt 3 / 2
  ∀ z : ℂ, z^3 = 1 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃ := by
sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_l747_74754


namespace NUMINAMATH_CALUDE_constant_value_c_l747_74714

theorem constant_value_c (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_c_l747_74714


namespace NUMINAMATH_CALUDE_perimeter_is_20_l747_74712

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define a line passing through the right focus
def line_through_right_focus (x y : ℝ) : Prop := sorry

-- Define points A and B on the ellipse and the line
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- Assumption that A and B are on the ellipse and the line
axiom A_on_ellipse : ellipse point_A.1 point_A.2
axiom B_on_ellipse : ellipse point_B.1 point_B.2
axiom A_on_line : line_through_right_focus point_A.1 point_A.2
axiom B_on_line : line_through_right_focus point_B.1 point_B.2

-- Define the perimeter of triangle F₁AB
def perimeter_F1AB : ℝ := sorry

-- Theorem statement
theorem perimeter_is_20 : perimeter_F1AB = 20 := by sorry

end NUMINAMATH_CALUDE_perimeter_is_20_l747_74712


namespace NUMINAMATH_CALUDE_tau_phi_equality_characterization_l747_74711

/-- Number of natural numbers dividing n -/
def tau (n : ℕ) : ℕ := sorry

/-- Number of natural numbers less than n that are relatively prime to n -/
def phi (n : ℕ) : ℕ := sorry

/-- Predicate for n having exactly two different prime divisors -/
def has_two_prime_divisors (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem tau_phi_equality_characterization (n : ℕ) :
  has_two_prime_divisors n ∧ tau (phi n) = phi (tau n) ↔
  ∃ k : ℕ, n = 3 * 2^(2^k - 1) :=
sorry

end NUMINAMATH_CALUDE_tau_phi_equality_characterization_l747_74711


namespace NUMINAMATH_CALUDE_exponent_division_l747_74722

theorem exponent_division (a : ℝ) : a^7 / a^4 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l747_74722


namespace NUMINAMATH_CALUDE_f_at_2023_half_l747_74774

/-- A function that is odd and symmetric about x = 1 -/
def f (x : ℝ) : ℝ :=
  sorry

/-- The function f is odd -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- The function f is symmetric about x = 1 -/
axiom f_sym (x : ℝ) : f x = f (2 - x)

/-- The function f is defined as 2^x + b for x ∈ [0,1] -/
axiom f_def (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = 2^x + Real.pi

/-- The main theorem -/
theorem f_at_2023_half : f (2023/2) = 1 - Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_f_at_2023_half_l747_74774


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l747_74731

/-- The problem statement -/
theorem min_reciprocal_sum (x y a b : ℝ) : 
  8 * x - y - 4 ≤ 0 →
  x + y + 1 ≥ 0 →
  y - 4 * x ≤ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', 8 * x' - y' - 4 ≤ 0 → x' + y' + 1 ≥ 0 → y' - 4 * x' ≤ 0 → a * x' + b * y' ≤ 2) →
  a * x + b * y = 2 →
  (∀ a' b', a' > 0 → b' > 0 → 
    (∀ x' y', 8 * x' - y' - 4 ≤ 0 → x' + y' + 1 ≥ 0 → y' - 4 * x' ≤ 0 → a' * x' + b' * y' ≤ 2) →
    1 / a + 1 / b ≤ 1 / a' + 1 / b') →
  1 / a + 1 / b = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l747_74731


namespace NUMINAMATH_CALUDE_falcons_minimum_wins_l747_74796

/-- The minimum number of additional games the Falcons need to win -/
def min_additional_games : ℕ := 29

/-- The total number of initial games played -/
def initial_games : ℕ := 5

/-- The number of games won by the Falcons initially -/
def initial_falcons_wins : ℕ := 2

/-- The minimum winning percentage required for the Falcons -/
def min_winning_percentage : ℚ := 91 / 100

theorem falcons_minimum_wins (N : ℕ) :
  (N ≥ min_additional_games) →
  ((initial_falcons_wins + N : ℚ) / (initial_games + N)) ≥ min_winning_percentage ∧
  ∀ M : ℕ, M < min_additional_games →
    ((initial_falcons_wins + M : ℚ) / (initial_games + M)) < min_winning_percentage :=
by sorry

end NUMINAMATH_CALUDE_falcons_minimum_wins_l747_74796


namespace NUMINAMATH_CALUDE_max_smoothie_servings_l747_74735

/-- Represents the recipe for 8 servings -/
structure Recipe where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ
  milk : ℕ

/-- Represents Sarah's available ingredients -/
structure Ingredients where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ
  milk : ℕ

/-- Calculates the maximum number of servings possible for a given ingredient -/
def max_servings (recipe_amount : ℕ) (available_amount : ℕ) : ℚ :=
  (available_amount : ℚ) / (recipe_amount : ℚ) * 8

/-- Theorem stating the maximum number of servings Sarah can make -/
theorem max_smoothie_servings (recipe : Recipe) (sarah_ingredients : Ingredients) :
  recipe.bananas = 3 ∧ 
  recipe.strawberries = 2 ∧ 
  recipe.yogurt = 1 ∧ 
  recipe.milk = 4 ∧
  sarah_ingredients.bananas = 10 ∧
  sarah_ingredients.strawberries = 5 ∧
  sarah_ingredients.yogurt = 3 ∧
  sarah_ingredients.milk = 10 →
  ⌊min 
    (min (max_servings recipe.bananas sarah_ingredients.bananas) (max_servings recipe.strawberries sarah_ingredients.strawberries))
    (min (max_servings recipe.yogurt sarah_ingredients.yogurt) (max_servings recipe.milk sarah_ingredients.milk))
  ⌋ = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_smoothie_servings_l747_74735


namespace NUMINAMATH_CALUDE_certain_value_proof_l747_74750

theorem certain_value_proof (n : ℤ) (v : ℤ) : 
  (∀ m : ℤ, 101 * m^2 ≤ v → m ≤ 8) → 
  (101 * 8^2 ≤ v) →
  v = 6464 := by
sorry

end NUMINAMATH_CALUDE_certain_value_proof_l747_74750


namespace NUMINAMATH_CALUDE_book_club_member_ratio_l747_74778

theorem book_club_member_ratio :
  ∀ (r p : ℕ), 
    r > 0 → p > 0 →
    (5 * r + 12 * p : ℚ) / (r + p) = 8 →
    (r : ℚ) / p = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_book_club_member_ratio_l747_74778


namespace NUMINAMATH_CALUDE_complex_number_sum_l747_74720

theorem complex_number_sum (z : ℂ) 
  (h : 16 * Complex.abs z ^ 2 = 3 * Complex.abs (z + 1) ^ 2 + Complex.abs (z ^ 2 - 2) ^ 2 + 43) : 
  z + 8 / z = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_l747_74720


namespace NUMINAMATH_CALUDE_max_handshakes_60_men_l747_74727

/-- The maximum number of handshakes for n people without cyclic handshakes -/
def max_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 60 men, the maximum number of handshakes without cyclic handshakes is 1770 -/
theorem max_handshakes_60_men :
  max_handshakes 60 = 1770 := by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_60_men_l747_74727


namespace NUMINAMATH_CALUDE_certain_number_existence_and_value_l747_74769

theorem certain_number_existence_and_value :
  ∃ n : ℝ, 8 * n - (0.6 * 10) / 1.2 = 31.000000000000004 ∧ 
  ∃ ε > 0, |n - 4.5| < ε := by
  sorry

end NUMINAMATH_CALUDE_certain_number_existence_and_value_l747_74769


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l747_74757

-- Define the repeating decimal 0.333...
def repeating_third : ℚ := 1/3

-- State the theorem
theorem eight_divided_by_repeating_third (h : repeating_third = 1/3) : 
  8 / repeating_third = 24 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l747_74757


namespace NUMINAMATH_CALUDE_ram_price_increase_l747_74790

theorem ram_price_increase (original_price current_price : ℝ) 
  (h1 : original_price = 50)
  (h2 : current_price = 52)
  (h3 : current_price = 0.8 * (original_price * (1 + increase_percentage / 100))) :
  increase_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_ram_price_increase_l747_74790


namespace NUMINAMATH_CALUDE_special_subset_count_l747_74736

def subset_count (n : ℕ) : ℕ :=
  (Finset.range 11).sum (fun k => Nat.choose (n - k + 1) k)

theorem special_subset_count : subset_count 20 = 3164 := by
  sorry

end NUMINAMATH_CALUDE_special_subset_count_l747_74736


namespace NUMINAMATH_CALUDE_mango_ratio_l747_74787

def alexis_mangoes : ℕ := 60
def total_mangoes : ℕ := 75

def others_mangoes : ℕ := total_mangoes - alexis_mangoes

theorem mango_ratio : 
  (alexis_mangoes : ℚ) / (others_mangoes : ℚ) = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_mango_ratio_l747_74787


namespace NUMINAMATH_CALUDE_rectangle_side_length_l747_74728

/-- Given two rectangles A and B, where A has sides a and b, and B has sides c and d,
    with the ratio of corresponding sides being 3/4, prove that when a = 3 and b = 6,
    the length of side d in Rectangle B is 8. -/
theorem rectangle_side_length (a b c d : ℝ) : 
  a = 3 → b = 6 → a / c = 3 / 4 → b / d = 3 / 4 → d = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l747_74728


namespace NUMINAMATH_CALUDE_cone_radius_l747_74729

/-- Given a cone with slant height 5 cm and lateral surface area 15π cm², 
    prove that the radius of the base is 3 cm. -/
theorem cone_radius (l : ℝ) (A : ℝ) (r : ℝ) : 
  l = 5 → A = 15 * Real.pi → A = Real.pi * r * l → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_radius_l747_74729


namespace NUMINAMATH_CALUDE_quadratic_even_deductive_reasoning_l747_74753

-- Definition of an even function
def IsEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Definition of a quadratic function
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c

-- Definition of deductive reasoning process
structure DeductiveReasoning :=
  (majorPremise : Prop)
  (minorPremise : Prop)
  (conclusion : Prop)

-- Theorem stating that the reasoning process for proving x^2 is even is deductive
theorem quadratic_even_deductive_reasoning :
  ∃ (reasoning : DeductiveReasoning),
    reasoning.majorPremise = (∀ f : ℝ → ℝ, IsEvenFunction f → ∀ x : ℝ, f (-x) = f x) ∧
    reasoning.minorPremise = (∃ f : ℝ → ℝ, QuadraticFunction f ∧ ∀ x : ℝ, f (-x) = f x) ∧
    reasoning.conclusion = (∃ f : ℝ → ℝ, QuadraticFunction f ∧ IsEvenFunction f) :=
  sorry


end NUMINAMATH_CALUDE_quadratic_even_deductive_reasoning_l747_74753


namespace NUMINAMATH_CALUDE_product_digits_l747_74793

def a : ℕ := 8476235982145327
def b : ℕ := 2983674531

theorem product_digits : (String.length (toString (a * b))) = 28 := by
  sorry

end NUMINAMATH_CALUDE_product_digits_l747_74793


namespace NUMINAMATH_CALUDE_ball_painting_probability_l747_74733

def num_balls : ℕ := 8
def num_red : ℕ := 4
def num_blue : ℕ := 4

def prob_red : ℚ := 1 / 2
def prob_blue : ℚ := 1 / 2

theorem ball_painting_probability :
  (prob_red ^ num_red) * (prob_blue ^ num_blue) = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_ball_painting_probability_l747_74733


namespace NUMINAMATH_CALUDE_unique_solution_modular_equation_l747_74742

theorem unique_solution_modular_equation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 102 ∧ 99 * n % 102 = 73 % 102 ∧ n = 97 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_modular_equation_l747_74742


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l747_74748

theorem twenty_five_percent_less_than_80 : ∃ x : ℝ, x + (1/4 * x) = 0.75 * 80 ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l747_74748


namespace NUMINAMATH_CALUDE_complement_of_union_l747_74710

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3, 4}
def N : Set Nat := {4, 5}

theorem complement_of_union :
  (U \ (M ∪ N)) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l747_74710


namespace NUMINAMATH_CALUDE_blue_balls_count_l747_74781

def total_balls : ℕ := 12

def prob_two_blue : ℚ := 1/22

theorem blue_balls_count :
  ∃ b : ℕ, 
    b ≤ total_balls ∧ 
    (b : ℚ) / total_balls * ((b - 1) : ℚ) / (total_balls - 1) = prob_two_blue ∧
    b = 3 :=
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l747_74781


namespace NUMINAMATH_CALUDE_seven_possible_D_values_l747_74792

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition of two 5-digit numbers ABBCB + BCAIA = DBDDD -/
def ValidAddition (A B C D : Digit) : Prop :=
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D) ∧
  (10000 * A.val + 1000 * B.val + 100 * B.val + 10 * C.val + B.val) +
  (10000 * B.val + 1000 * C.val + 100 * A.val + 10 * 1 + A.val) =
  (10000 * D.val + 1000 * B.val + 100 * D.val + 10 * D.val + D.val)

/-- The theorem stating that there are exactly 7 possible values for D -/
theorem seven_possible_D_values :
  ∃ (S : Finset Digit), S.card = 7 ∧
  (∀ D, D ∈ S ↔ ∃ A B C, ValidAddition A B C D) :=
sorry

end NUMINAMATH_CALUDE_seven_possible_D_values_l747_74792


namespace NUMINAMATH_CALUDE_range_of_a_l747_74741

-- Define set A
def A : Set ℝ := {x | x^2 - x < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := Set.Ioo 0 a

-- Theorem statement
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : A ⊆ B a) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l747_74741


namespace NUMINAMATH_CALUDE_school_teacher_student_ratio_l747_74795

theorem school_teacher_student_ratio 
  (b c k h : ℕ) 
  (h_positive : h > 0) 
  (k_ge_two : k ≥ 2) 
  (c_ge_two : c ≥ 2) :
  (b : ℚ) / h = (c * (c - 1)) / (k * (k - 1)) := by
sorry

end NUMINAMATH_CALUDE_school_teacher_student_ratio_l747_74795


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l747_74791

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l747_74791


namespace NUMINAMATH_CALUDE_percentage_of_women_parents_l747_74746

theorem percentage_of_women_parents (P : ℝ) (M : ℝ) (F : ℝ) : 
  P > 0 →
  M + F = P →
  (1 / 8) * M + (1 / 4) * F = (17.5 / 100) * P →
  M / P = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_women_parents_l747_74746


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l747_74771

-- Problem 1
theorem problem_one : 4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_two : (4 * Real.sqrt 3 - 6 * Real.sqrt (1/3) + 3 * Real.sqrt 12) / (2 * Real.sqrt 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l747_74771


namespace NUMINAMATH_CALUDE_room_occupancy_l747_74740

theorem room_occupancy (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) :
  total_chairs = 25 →
  seated_people = (4 : ℕ) * total_chairs / 5 →
  seated_people = (3 : ℕ) * total_people / 5 →
  total_people = 33 :=
by
  sorry

#check room_occupancy

end NUMINAMATH_CALUDE_room_occupancy_l747_74740


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l747_74719

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l747_74719


namespace NUMINAMATH_CALUDE_hexagon_triangle_count_l747_74763

/-- Represents a regular hexagon -/
structure RegularHexagon :=
  (area : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (area : ℝ)

/-- Counts the number of equilateral triangles with a given area that can be formed from the vertices of a set of regular hexagons -/
def countEquilateralTriangles (hexagons : List RegularHexagon) (targetTriangle : EquilateralTriangle) : ℕ :=
  sorry

/-- The main theorem stating that 4 regular hexagons with area 6 can form 8 equilateral triangles with area 4 -/
theorem hexagon_triangle_count :
  let hexagons := List.replicate 4 { area := 6 : RegularHexagon }
  let targetTriangle := { area := 4 : EquilateralTriangle }
  countEquilateralTriangles hexagons targetTriangle = 8 := by sorry

end NUMINAMATH_CALUDE_hexagon_triangle_count_l747_74763


namespace NUMINAMATH_CALUDE_elberta_amount_l747_74743

-- Define the amounts for each person
def granny_smith : ℕ := 75
def anjou : ℕ := granny_smith / 4
def elberta : ℕ := anjou + 3

-- Theorem statement
theorem elberta_amount : elberta = 22 := by
  sorry

end NUMINAMATH_CALUDE_elberta_amount_l747_74743


namespace NUMINAMATH_CALUDE_set_operations_l747_74704

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 4 < x ∧ x < 10}

-- Define the intervals for the results
def interval_3_10 : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def open_4_7 : Set ℝ := {x | 4 < x ∧ x < 7}
def union_4_7_7_10 : Set ℝ := {x | (4 < x ∧ x < 7) ∨ (7 ≤ x ∧ x < 10)}

-- State the theorem
theorem set_operations :
  (A ∪ B = interval_3_10) ∧
  (A ∩ B = open_4_7) ∧
  ((Set.univ \ A) ∩ B = union_4_7_7_10) := by sorry

end NUMINAMATH_CALUDE_set_operations_l747_74704


namespace NUMINAMATH_CALUDE_modulus_of_2_plus_i_l747_74775

/-- The modulus of the complex number 2 + i is √5 -/
theorem modulus_of_2_plus_i : Complex.abs (2 + Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_2_plus_i_l747_74775


namespace NUMINAMATH_CALUDE_factors_of_28350_l747_74701

/-- The number of positive factors of 28350 -/
def num_factors_28350 : ℕ := sorry

/-- 28350 is the number we are analyzing -/
def n : ℕ := 28350

theorem factors_of_28350 : num_factors_28350 = 48 := by sorry

end NUMINAMATH_CALUDE_factors_of_28350_l747_74701


namespace NUMINAMATH_CALUDE_candy_bar_difference_l747_74783

theorem candy_bar_difference (lena nicole kevin : ℕ) : 
  lena = 23 →
  lena + 7 = 4 * kevin →
  nicole = kevin + 6 →
  lena - nicole = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_bar_difference_l747_74783


namespace NUMINAMATH_CALUDE_triangle_sides_simplification_l747_74707

theorem triangle_sides_simplification (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : a + c > b) 
  (h4 : a > 0) 
  (h5 : b > 0) 
  (h6 : c > 0) : 
  |c - a - b| + |c + b - a| = 2 * b := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_simplification_l747_74707


namespace NUMINAMATH_CALUDE_sum_base5_equals_1333_l747_74798

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from decimal to base 5 --/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The sum of 213₅, 324₅, and 141₅ is equal to 1333₅ in base 5 --/
theorem sum_base5_equals_1333 :
  decimalToBase5 (base5ToDecimal [3, 1, 2] + base5ToDecimal [4, 2, 3] + base5ToDecimal [1, 4, 1]) = [3, 3, 3, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_base5_equals_1333_l747_74798


namespace NUMINAMATH_CALUDE_hadley_books_added_l747_74782

theorem hadley_books_added (initial_books : ℕ) (borrowed_by_lunch : ℕ) (borrowed_by_evening : ℕ) (remaining_books : ℕ) : 
  initial_books = 100 →
  borrowed_by_lunch = 50 →
  borrowed_by_evening = 30 →
  remaining_books = 60 →
  initial_books - borrowed_by_lunch + (remaining_books + borrowed_by_evening - (initial_books - borrowed_by_lunch)) = 100 :=
by
  sorry

#check hadley_books_added

end NUMINAMATH_CALUDE_hadley_books_added_l747_74782


namespace NUMINAMATH_CALUDE_lawn_width_proof_l747_74737

/-- Proves that the width of a rectangular lawn is 60 meters given specific conditions --/
theorem lawn_width_proof (W : ℝ) : 
  W > 0 →  -- Width is positive
  (10 * W + 10 * 70 - 10 * 10) * 3 = 3600 →  -- Cost equation
  W = 60 := by
  sorry

end NUMINAMATH_CALUDE_lawn_width_proof_l747_74737


namespace NUMINAMATH_CALUDE_dining_bill_share_l747_74779

/-- Given a total bill, number of people, and tip percentage, calculate the amount each person should pay. -/
def calculate_share (total_bill : ℚ) (num_people : ℕ) (tip_percentage : ℚ) : ℚ :=
  let total_with_tip := total_bill * (1 + tip_percentage)
  total_with_tip / num_people

/-- Prove that for a bill of $139.00 split among 5 people with a 10% tip, each person should pay $30.58. -/
theorem dining_bill_share :
  calculate_share 139 5 (1/10) = 3058/100 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_share_l747_74779


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l747_74799

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- The number of herbs that react negatively with one crystal. -/
def num_incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l747_74799


namespace NUMINAMATH_CALUDE_snail_final_position_l747_74767

/-- Represents the direction the snail is facing -/
inductive Direction
  | Up
  | Right
  | Down
  | Left

/-- Represents a position on the grid -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the state of the snail -/
structure SnailState where
  pos : Position
  dir : Direction
  visited : Set Position

/-- The grid dimensions -/
def gridWidth : Nat := 300
def gridHeight : Nat := 50

/-- Check if a position is within the grid -/
def isValidPosition (p : Position) : Bool :=
  p.row >= 1 && p.row <= gridHeight && p.col >= 1 && p.col <= gridWidth

/-- Move the snail according to the rules -/
def moveSnail (state : SnailState) : SnailState :=
  sorry -- Implementation of snail movement logic

/-- The main theorem stating the final position of the snail -/
theorem snail_final_position :
  ∃ (finalState : SnailState),
    (∀ (p : Position), isValidPosition p → p ∈ finalState.visited) ∧
    finalState.pos = Position.mk 25 26 := by
  sorry


end NUMINAMATH_CALUDE_snail_final_position_l747_74767


namespace NUMINAMATH_CALUDE_angle_B_value_max_sum_sides_l747_74725

-- Define the triangle
variable (A B C a b c : ℝ)

-- Define the conditions
variable (triangle_abc : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variable (side_angle_relation : b * Real.cos C = (2 * a - c) * Real.cos B)

-- First theorem: B = π/3
theorem angle_B_value : B = π / 3 := by sorry

-- Second theorem: Maximum value of a + c when b = √3
theorem max_sum_sides (h_b : b = Real.sqrt 3) :
  ∃ (max : ℝ), max = 2 * Real.sqrt 3 ∧ 
  ∀ (a c : ℝ), a + c ≤ max := by sorry

end NUMINAMATH_CALUDE_angle_B_value_max_sum_sides_l747_74725


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_7_neg7_4_l747_74785

/-- Converts rectangular coordinates to cylindrical coordinates -/
def rect_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ := sorry

theorem rect_to_cylindrical_7_neg7_4 :
  let (r, θ, z) := rect_to_cylindrical 7 (-7) 4
  r = 7 * Real.sqrt 2 ∧
  θ = 7 * Real.pi / 4 ∧
  z = 4 ∧
  r > 0 ∧
  0 ≤ θ ∧ θ < 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_7_neg7_4_l747_74785


namespace NUMINAMATH_CALUDE_parts_count_l747_74717

/-- Represents the number of parts in pile A -/
def pile_a : ℕ := sorry

/-- Represents the number of parts in pile B -/
def pile_b : ℕ := sorry

/-- The condition that transferring 15 parts from A to B makes them equal -/
axiom equal_after_a_to_b : pile_a - 15 = pile_b + 15

/-- The condition that transferring 15 parts from B to A makes A three times B -/
axiom triple_after_b_to_a : pile_a + 15 = 3 * (pile_b - 15)

/-- The theorem stating the original number in pile A and the total number of parts -/
theorem parts_count : pile_a = 75 ∧ pile_a + pile_b = 120 := by sorry

end NUMINAMATH_CALUDE_parts_count_l747_74717


namespace NUMINAMATH_CALUDE_quadratic_roots_l747_74758

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  point_neg_two : a * (-2)^2 + b * (-2) + c = 12
  point_zero : c = -8
  point_one : a + b + c = -12
  point_three : a * 3^2 + b * 3 + c = -8

/-- The theorem statement -/
theorem quadratic_roots (f : QuadraticFunction) :
  let roots := {x : ℝ | f.a * x^2 + f.b * x + f.c + 8 = 0}
  roots = {0, 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l747_74758


namespace NUMINAMATH_CALUDE_garden_perimeter_l747_74786

/-- A rectangular garden with a diagonal of 20 meters and an area of 96 square meters has a perimeter of 49 meters. -/
theorem garden_perimeter : ∀ a b : ℝ,
  a > 0 → b > 0 →
  a^2 + b^2 = 20^2 →
  a * b = 96 →
  2 * (a + b) = 49 := by
sorry

end NUMINAMATH_CALUDE_garden_perimeter_l747_74786


namespace NUMINAMATH_CALUDE_task_assignment_count_l747_74700

def select_and_assign (n m : ℕ) : ℕ :=
  Nat.choose n m * Nat.choose m 2 * 2

theorem task_assignment_count : select_and_assign 10 4 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_task_assignment_count_l747_74700


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l747_74739

theorem yellow_marbles_count (total : ℕ) (yellow green red blue : ℕ) : 
  total = 60 →
  green = yellow / 2 →
  red = blue →
  blue = total / 4 →
  total = yellow + green + red + blue →
  yellow = 20 := by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l747_74739


namespace NUMINAMATH_CALUDE_melanies_turnips_l747_74715

/-- The number of turnips Benny grew -/
def bennys_turnips : ℕ := 113

/-- The total number of turnips grown by Melanie and Benny -/
def total_turnips : ℕ := 252

/-- Melanie's turnips are equal to the total minus Benny's -/
theorem melanies_turnips : ℕ := total_turnips - bennys_turnips

#check melanies_turnips

end NUMINAMATH_CALUDE_melanies_turnips_l747_74715


namespace NUMINAMATH_CALUDE_cafe_choices_l747_74703

/-- The number of ways two people can choose different items from a set of n items -/
def differentChoices (n : ℕ) : ℕ := n * (n - 1)

/-- The number of menu items in the café -/
def menuItems : ℕ := 12

/-- Theorem: The number of ways Alex and Jamie can choose different dishes from a menu of 12 items is 132 -/
theorem cafe_choices : differentChoices menuItems = 132 := by
  sorry

end NUMINAMATH_CALUDE_cafe_choices_l747_74703


namespace NUMINAMATH_CALUDE_alexandre_winning_strategy_l747_74730

/-- A game on an n-gon where players alternately mark vertices with 0 or 1 -/
def Game (n : ℕ) := Unit

/-- A strategy for the second player (Alexandre) -/
def Strategy (n : ℕ) := Game n → ℕ → ℕ

/-- Predicate to check if three consecutive vertices have a sum divisible by 3 -/
def HasWinningTriple (g : Game n) : Prop := sorry

/-- Predicate to check if a strategy is winning for the second player -/
def IsWinningStrategy (s : Strategy n) : Prop := sorry

theorem alexandre_winning_strategy 
  (n : ℕ) 
  (h1 : n > 3) 
  (h2 : Even n) : 
  ∃ (s : Strategy n), IsWinningStrategy s := by sorry

end NUMINAMATH_CALUDE_alexandre_winning_strategy_l747_74730


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l747_74747

theorem sum_of_distinct_prime_factors : 
  (let n := 7^6 - 2 * 7^4
   Finset.sum (Finset.filter (fun p => Nat.Prime p ∧ n % p = 0) (Finset.range (n + 1))) id) = 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l747_74747


namespace NUMINAMATH_CALUDE_probability_both_preferred_is_one_fourth_l747_74749

/-- Represents the colors of the balls -/
inductive Color
| Red
| Yellow
| Blue
| Green
| Purple

/-- Represents a person -/
structure Person where
  name : String
  preferredColors : List Color

/-- Represents the bag of balls -/
def bag : List Color := [Color.Red, Color.Yellow, Color.Blue, Color.Green, Color.Purple]

/-- Person A's preferred colors -/
def personA : Person := { name := "A", preferredColors := [Color.Red, Color.Yellow] }

/-- Person B's preferred colors -/
def personB : Person := { name := "B", preferredColors := [Color.Yellow, Color.Green, Color.Purple] }

/-- Calculates the probability of both persons drawing their preferred colors -/
def probabilityBothPreferred (bag : List Color) (personA personB : Person) : ℚ :=
  sorry

/-- Theorem stating the probability of both persons drawing their preferred colors is 1/4 -/
theorem probability_both_preferred_is_one_fourth :
  probabilityBothPreferred bag personA personB = 1/4 :=
sorry

end NUMINAMATH_CALUDE_probability_both_preferred_is_one_fourth_l747_74749


namespace NUMINAMATH_CALUDE_dislike_both_count_l747_74764

/-- The number of people who don't like both radio and music in a poll -/
def people_dislike_both (total : ℕ) (radio_dislike_percent : ℚ) (music_dislike_percent : ℚ) : ℕ :=
  ⌊(radio_dislike_percent * music_dislike_percent * total : ℚ)⌋₊

/-- Theorem about the number of people who don't like both radio and music -/
theorem dislike_both_count :
  people_dislike_both 1500 (35/100) (15/100) = 79 := by
  sorry

#eval people_dislike_both 1500 (35/100) (15/100)

end NUMINAMATH_CALUDE_dislike_both_count_l747_74764


namespace NUMINAMATH_CALUDE_pencil_order_cost_l747_74705

/-- Calculates the cost of pencils with a potential discount -/
def pencilCost (boxSize : ℕ) (boxPrice : ℚ) (discountThreshold : ℕ) (discountRate : ℚ) (quantity : ℕ) : ℚ :=
  let basePrice := (quantity : ℚ) * boxPrice / (boxSize : ℚ)
  if quantity > discountThreshold then
    basePrice * (1 - discountRate)
  else
    basePrice

theorem pencil_order_cost :
  pencilCost 200 40 1000 (1/10) 2400 = 432 :=
by sorry

end NUMINAMATH_CALUDE_pencil_order_cost_l747_74705


namespace NUMINAMATH_CALUDE_sector_central_angle_l747_74708

theorem sector_central_angle (area : ℝ) (perimeter : ℝ) (h1 : area = 5) (h2 : perimeter = 9) :
  ∃ (r : ℝ) (l : ℝ),
    2 * r + l = perimeter ∧
    1/2 * l * r = area ∧
    (l / r = 5/2 ∨ l / r = 8/5) :=
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l747_74708


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l747_74761

def a : ℝ × ℝ := (2, 3)
def b (t : ℝ) : ℝ × ℝ := (t, -1)

theorem perpendicular_vectors_magnitude (t : ℝ) :
  (a.1 * (b t).1 + a.2 * (b t).2 = 0) →
  Real.sqrt ((a.1 - 2 * (b t).1)^2 + (a.2 - 2 * (b t).2)^2) = Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l747_74761


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l747_74738

theorem negative_sixty_four_to_seven_thirds :
  (-64 : ℝ) ^ (7/3) = -1024 := by sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l747_74738


namespace NUMINAMATH_CALUDE_rectangular_equation_chord_length_l747_74706

-- Define the polar equation of curve C
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ)^2 = 8 * Real.cos θ

-- Define the parametric equations of line l
def line_equation (t x y : ℝ) : Prop :=
  x = 2 + (1/2) * t ∧ y = (Real.sqrt 3 / 2) * t

-- Theorem for the rectangular equation of curve C
theorem rectangular_equation (x y : ℝ) :
  (∃ ρ θ, polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔
  y^2 = 8 * x :=
sorry

-- Theorem for the length of chord AB
theorem chord_length :
  ∃ t₁ t₂ x₁ y₁ x₂ y₂,
    line_equation t₁ x₁ y₁ ∧ line_equation t₂ x₂ y₂ ∧
    y₁^2 = 8 * x₁ ∧ y₂^2 = 8 * x₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 32/3 :=
sorry

end NUMINAMATH_CALUDE_rectangular_equation_chord_length_l747_74706


namespace NUMINAMATH_CALUDE_robins_hair_length_l747_74768

theorem robins_hair_length :
  ∀ (initial_length : ℝ),
    initial_length + 8 - 20 = 2 →
    initial_length = 14 :=
by sorry

end NUMINAMATH_CALUDE_robins_hair_length_l747_74768
