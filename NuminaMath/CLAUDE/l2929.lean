import Mathlib

namespace NUMINAMATH_CALUDE_leo_has_more_leo_excess_marbles_l2929_292986

/-- The number of marbles Ben has -/
def ben_marbles : ℕ := 56

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := 132

/-- Leo's marbles are the difference between the total and Ben's marbles -/
def leo_marbles : ℕ := total_marbles - ben_marbles

/-- The statement that Leo has more marbles than Ben -/
theorem leo_has_more : leo_marbles > ben_marbles := by sorry

/-- The main theorem: Leo has 20 more marbles than Ben -/
theorem leo_excess_marbles : leo_marbles - ben_marbles = 20 := by sorry

end NUMINAMATH_CALUDE_leo_has_more_leo_excess_marbles_l2929_292986


namespace NUMINAMATH_CALUDE_toms_average_speed_l2929_292985

/-- Prove that Tom's average speed is 45 mph given the race conditions -/
theorem toms_average_speed (karen_speed : ℝ) (karen_delay : ℝ) (karen_win_margin : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 →
  karen_delay = 1 / 15 →
  karen_win_margin = 4 →
  tom_distance = 24 →
  (tom_distance / ((tom_distance + karen_win_margin) / karen_speed + karen_delay)) = 45 :=
by sorry

end NUMINAMATH_CALUDE_toms_average_speed_l2929_292985


namespace NUMINAMATH_CALUDE_correct_dial_probability_l2929_292921

/-- The probability of correctly dialing a phone number with a missing last digit -/
def dial_probability : ℚ := 3 / 10

/-- The number of possible digits for a phone number -/
def num_digits : ℕ := 10

/-- The maximum number of attempts allowed -/
def max_attempts : ℕ := 3

theorem correct_dial_probability :
  (∀ n : ℕ, n ≤ max_attempts → (1 : ℚ) / num_digits = 1 / 10) →
  (∀ n : ℕ, n < max_attempts → (num_digits - n : ℚ) / num_digits * (1 : ℚ) / (num_digits - n) = 1 / 10) →
  dial_probability = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_dial_probability_l2929_292921


namespace NUMINAMATH_CALUDE_sqrt_23_bound_l2929_292996

theorem sqrt_23_bound : 4.5 < Real.sqrt 23 ∧ Real.sqrt 23 < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_23_bound_l2929_292996


namespace NUMINAMATH_CALUDE_potato_slab_length_difference_l2929_292973

theorem potato_slab_length_difference (total_length first_piece_length : ℕ) 
  (h1 : total_length = 600)
  (h2 : first_piece_length = 275) :
  total_length - first_piece_length - first_piece_length = 50 :=
by sorry

end NUMINAMATH_CALUDE_potato_slab_length_difference_l2929_292973


namespace NUMINAMATH_CALUDE_angle_D_measure_l2929_292998

-- Define the hexagon and its angles
structure Hexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the properties of the hexagon
def is_convex_hexagon_with_properties (h : Hexagon) : Prop :=
  -- Angles A, B, and C are congruent
  h.A = h.B ∧ h.B = h.C
  -- Angles D and E are congruent
  ∧ h.D = h.E
  -- Angle A is 50 degrees less than angle D
  ∧ h.A + 50 = h.D
  -- Sum of angles in a hexagon is 720 degrees
  ∧ h.A + h.B + h.C + h.D + h.E + h.F = 720

-- Theorem statement
theorem angle_D_measure (h : Hexagon) 
  (h_props : is_convex_hexagon_with_properties h) : 
  h.D = 153.33 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l2929_292998


namespace NUMINAMATH_CALUDE_election_votes_calculation_l2929_292910

theorem election_votes_calculation (total_votes : ℕ) : 
  (∃ (candidate_votes rival_votes : ℕ),
    candidate_votes = (40 * total_votes) / 100 ∧
    rival_votes = candidate_votes + 5000 ∧
    rival_votes + candidate_votes = total_votes) →
  total_votes = 25000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l2929_292910


namespace NUMINAMATH_CALUDE_non_pine_trees_l2929_292916

theorem non_pine_trees (total : ℕ) (pine_percentage : ℚ) : 
  total = 350 → pine_percentage = 70 / 100 → 
  total - (total * pine_percentage).floor = 105 := by
  sorry

end NUMINAMATH_CALUDE_non_pine_trees_l2929_292916


namespace NUMINAMATH_CALUDE_approximation_theorem_l2929_292983

theorem approximation_theorem (a b : ℝ) (ε : ℝ) (hε : ε > 0) :
  ∃ (k m : ℤ) (n : ℕ), |n • a - k| < ε ∧ |n • b - m| < ε := by
  sorry

end NUMINAMATH_CALUDE_approximation_theorem_l2929_292983


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l2929_292982

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : IsEven f) : IsEven (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l2929_292982


namespace NUMINAMATH_CALUDE_minimize_sum_distances_l2929_292991

/-- A structure representing a point on a line --/
structure Point where
  x : ℝ

/-- The distance between two points on a line --/
def distance (p q : Point) : ℝ := abs (p.x - q.x)

/-- The theorem stating that Q₅ minimizes the sum of distances --/
theorem minimize_sum_distances 
  (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : Point)
  (h_order : Q₁.x < Q₂.x ∧ Q₂.x < Q₃.x ∧ Q₃.x < Q₄.x ∧ Q₄.x < Q₅.x ∧ 
             Q₅.x < Q₆.x ∧ Q₆.x < Q₇.x ∧ Q₇.x < Q₈.x ∧ Q₈.x < Q₉.x)
  (h_fixed : Q₁.x ≠ Q₉.x)
  (h_not_midpoint : Q₅.x ≠ (Q₁.x + Q₉.x) / 2) :
  ∀ Q : Point, 
    distance Q Q₁ + distance Q Q₂ + distance Q Q₃ + distance Q Q₄ + 
    distance Q Q₅ + distance Q Q₆ + distance Q Q₇ + distance Q Q₈ + 
    distance Q Q₉ 
    ≥ 
    distance Q₅ Q₁ + distance Q₅ Q₂ + distance Q₅ Q₃ + distance Q₅ Q₄ + 
    distance Q₅ Q₅ + distance Q₅ Q₆ + distance Q₅ Q₇ + distance Q₅ Q₈ + 
    distance Q₅ Q₉ :=
by sorry

end NUMINAMATH_CALUDE_minimize_sum_distances_l2929_292991


namespace NUMINAMATH_CALUDE_balanced_polynomial_existence_balanced_polynomial_equality_l2929_292908

-- Define what it means for an integer to be balanced
def IsBalanced (n : ℤ) : Prop :=
  n = 1 ∨ ∃ (k : ℕ) (p : List ℤ), k % 2 = 0 ∧ n = p.prod ∧ ∀ x ∈ p, Nat.Prime x.natAbs

-- Define the polynomial P(x) = (x+a)(x+b)
def P (a b : ℤ) (x : ℤ) : ℤ := (x + a) * (x + b)

theorem balanced_polynomial_existence :
  ∃ (a b : ℤ), a ≠ b ∧ a > 0 ∧ b > 0 ∧ ∀ n : ℤ, 1 ≤ n ∧ n ≤ 50 → IsBalanced (P a b n) :=
sorry

theorem balanced_polynomial_equality (a b : ℤ) (h : ∀ n : ℤ, IsBalanced (P a b n)) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_balanced_polynomial_existence_balanced_polynomial_equality_l2929_292908


namespace NUMINAMATH_CALUDE_purely_imaginary_z_reciprocal_l2929_292974

theorem purely_imaginary_z_reciprocal (m : ℝ) :
  let z : ℂ := m^2 - 1 + (m + 1) * I
  (∃ (y : ℝ), z = y * I) → 2 / z = -I :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_reciprocal_l2929_292974


namespace NUMINAMATH_CALUDE_can_cross_all_rivers_l2929_292941

def river1_width : ℕ := 487
def river2_width : ℕ := 621
def river3_width : ℕ := 376
def existing_bridge : ℕ := 295
def additional_material : ℕ := 1020

def extra_needed (river_width : ℕ) : ℕ :=
  if river_width > existing_bridge then river_width - existing_bridge else 0

theorem can_cross_all_rivers :
  extra_needed river1_width + extra_needed river2_width + extra_needed river3_width ≤ additional_material :=
by sorry

end NUMINAMATH_CALUDE_can_cross_all_rivers_l2929_292941


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_circle_segment_area_l2929_292917

theorem right_isosceles_triangle_circle_segment_area :
  let hypotenuse : ℝ := 10
  let radius : ℝ := hypotenuse / 2
  let sector_angle : ℝ := 45 -- in degrees
  let sector_area : ℝ := (sector_angle / 360) * π * radius^2
  let triangle_area : ℝ := (1 / 2) * radius^2
  let shaded_area : ℝ := sector_area - triangle_area
  let a : ℝ := 25
  let b : ℝ := 50
  let c : ℝ := 1
  (shaded_area = a * π - b * Real.sqrt c) ∧ (a + b + c = 76) := by
  sorry

end NUMINAMATH_CALUDE_right_isosceles_triangle_circle_segment_area_l2929_292917


namespace NUMINAMATH_CALUDE_teds_chocolates_l2929_292915

theorem teds_chocolates : ∃ (x : ℚ), 
  x > 0 ∧ 
  (3/16 * x - 3/4 - 5 = 10) ∧ 
  x = 84 := by
  sorry

end NUMINAMATH_CALUDE_teds_chocolates_l2929_292915


namespace NUMINAMATH_CALUDE_book_sale_price_l2929_292929

-- Define the total number of books
def total_books : ℕ := 150

-- Define the number of unsold books
def unsold_books : ℕ := 50

-- Define the total amount received
def total_amount : ℕ := 500

-- Define the fraction of books sold
def fraction_sold : ℚ := 2/3

-- Theorem to prove
theorem book_sale_price :
  let sold_books := total_books - unsold_books
  let price_per_book := total_amount / sold_books
  fraction_sold * total_books = sold_books ∧
  price_per_book = 5 := by
sorry

end NUMINAMATH_CALUDE_book_sale_price_l2929_292929


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_l2929_292920

theorem greatest_integer_fraction (x : ℤ) : 
  x ≠ 3 → 
  (∀ y : ℤ, y > 28 → ¬(∃ k : ℤ, (y^2 + 2*y + 10) = k * (y - 3))) → 
  (∃ k : ℤ, (28^2 + 2*28 + 10) = k * (28 - 3)) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_l2929_292920


namespace NUMINAMATH_CALUDE_problem_solution_l2929_292976

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 3|
def g (m x : ℝ) : ℝ := m - 2*|x - 11|

-- State the theorem
theorem problem_solution :
  (∀ x m : ℝ, 2 * f x ≥ g m (x + 4)) →
  (∃ t : ℝ, t = 20 ∧ ∀ m : ℝ, (∀ x : ℝ, 2 * f x ≥ g m (x + 4)) → m ≤ t) ∧
  (∀ a : ℝ, a > 0 →
    (∃ x y z : ℝ, 2*x^2 + 3*y^2 + 6*z^2 = a ∧
      ∀ x' y' z' : ℝ, 2*x'^2 + 3*y'^2 + 6*z'^2 = a → x' + y' + z' ≤ 1) →
    a = 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2929_292976


namespace NUMINAMATH_CALUDE_fibonacci_rectangle_division_l2929_292967

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- A rectangle that can be divided into n squares -/
structure DivisibleRectangle (n : ℕ) :=
  (width : ℕ)
  (height : ℕ)
  (divides_into_squares : ∃ (squares : Finset (ℕ × ℕ)), 
    squares.card = n ∧ 
    (∀ (s : ℕ × ℕ), s ∈ squares → s.1 * s.2 ≤ width * height) ∧
    (∀ (s1 s2 s3 : ℕ × ℕ), s1 ∈ squares → s2 ∈ squares → s3 ∈ squares → 
      s1 = s2 ∧ s2 = s3 → s1 = s2))

/-- Theorem: For each natural number n, there exists a rectangle that can be 
    divided into n squares with no more than two squares being the same size -/
theorem fibonacci_rectangle_division (n : ℕ) : 
  ∃ (rect : DivisibleRectangle n), rect.width = fib n ∧ rect.height = fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_rectangle_division_l2929_292967


namespace NUMINAMATH_CALUDE_tv_cash_savings_l2929_292987

/-- Calculates the savings when buying a television by cash instead of installments -/
theorem tv_cash_savings 
  (cash_price : ℕ) 
  (down_payment : ℕ) 
  (monthly_payment : ℕ) 
  (num_months : ℕ) : 
  cash_price = 400 →
  down_payment = 120 →
  monthly_payment = 30 →
  num_months = 12 →
  down_payment + monthly_payment * num_months - cash_price = 80 := by
sorry

end NUMINAMATH_CALUDE_tv_cash_savings_l2929_292987


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2929_292970

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| ≤ 1} = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2929_292970


namespace NUMINAMATH_CALUDE_carol_college_distance_l2929_292914

/-- The distance between Carol's college and home -/
def college_distance (fuel_efficiency : ℝ) (tank_capacity : ℝ) (remaining_distance : ℝ) : ℝ :=
  fuel_efficiency * tank_capacity + remaining_distance

/-- Theorem stating the distance between Carol's college and home -/
theorem carol_college_distance :
  college_distance 20 16 100 = 420 := by
  sorry

end NUMINAMATH_CALUDE_carol_college_distance_l2929_292914


namespace NUMINAMATH_CALUDE_shell_collection_l2929_292901

theorem shell_collection (laurie_shells : ℕ) (h1 : laurie_shells = 36) :
  ∃ (ben_shells alan_shells : ℕ),
    ben_shells = laurie_shells / 3 ∧
    alan_shells = ben_shells * 4 ∧
    alan_shells = 48 := by
  sorry

end NUMINAMATH_CALUDE_shell_collection_l2929_292901


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_triple_exterior_l2929_292965

theorem polygon_sides_when_interior_triple_exterior : ∃ n : ℕ,
  (n ≥ 3) ∧
  ((n - 2) * 180 = 3 * 360) ∧
  (∀ m : ℕ, m ≥ 3 → (m - 2) * 180 = 3 * 360 → m = n) :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_triple_exterior_l2929_292965


namespace NUMINAMATH_CALUDE_fraction_transformation_l2929_292957

theorem fraction_transformation (x : ℚ) : 
  x = 437 → (537 - x) / (463 + x) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2929_292957


namespace NUMINAMATH_CALUDE_amy_garden_space_l2929_292978

/-- Calculates the total square footage of garden beds -/
def total_sq_ft (num_beds1 num_beds2 : ℕ) (length1 width1 length2 width2 : ℝ) : ℝ :=
  (num_beds1 * length1 * width1) + (num_beds2 * length2 * width2)

/-- Proves that Amy's garden beds have a total of 42 sq ft of growing space -/
theorem amy_garden_space : total_sq_ft 2 2 3 3 4 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_amy_garden_space_l2929_292978


namespace NUMINAMATH_CALUDE_smallest_n_for_zoe_play_l2929_292995

def can_zoe_play (n : ℕ) : Prop :=
  ∀ (yvan_first : ℕ) (h_yvan_first : yvan_first ≤ n),
    ∃ (zoe_first : ℕ) (zoe_last : ℕ),
      zoe_first < zoe_last ∧
      zoe_last ≤ n ∧
      zoe_first ≠ yvan_first ∧
      zoe_last ≠ yvan_first ∧
      ∀ (yvan_second : ℕ) (yvan_second_last : ℕ),
        yvan_second < yvan_second_last ∧
        yvan_second_last ≤ n ∧
        yvan_second ≠ yvan_first ∧
        yvan_second_last ≠ yvan_first ∧
        yvan_second ∉ Set.Icc zoe_first zoe_last ∧
        yvan_second_last ∉ Set.Icc zoe_first zoe_last →
        ∃ (zoe_second : ℕ) (zoe_second_last : ℕ),
          zoe_second < zoe_second_last ∧
          zoe_second_last ≤ n ∧
          zoe_second ∉ Set.Icc zoe_first zoe_last ∪ Set.Icc yvan_second yvan_second_last ∧
          zoe_second_last ∉ Set.Icc zoe_first zoe_last ∪ Set.Icc yvan_second yvan_second_last ∧
          zoe_second_last - zoe_second = 3

theorem smallest_n_for_zoe_play :
  (∀ k < 14, ¬ can_zoe_play k) ∧ can_zoe_play 14 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_zoe_play_l2929_292995


namespace NUMINAMATH_CALUDE_x_less_than_y_l2929_292951

theorem x_less_than_y (a : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_l2929_292951


namespace NUMINAMATH_CALUDE_f_properties_l2929_292952

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (h2 : ∀ x : ℝ, x > 0 → f x < 0)

-- Theorem statement
theorem f_properties : (∀ x : ℝ, f (-x) = -f x) ∧
                       (∀ x y : ℝ, x < y → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2929_292952


namespace NUMINAMATH_CALUDE_divisibility_by_thirteen_l2929_292939

theorem divisibility_by_thirteen (a b c : ℤ) (h : 13 ∣ (a + b + c)) :
  13 ∣ (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_thirteen_l2929_292939


namespace NUMINAMATH_CALUDE_bottle_cap_probability_l2929_292953

theorem bottle_cap_probability (p_convex : ℝ) (h1 : p_convex = 0.44) :
  1 - p_convex = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_probability_l2929_292953


namespace NUMINAMATH_CALUDE_chickens_and_rabbits_l2929_292937

theorem chickens_and_rabbits (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : total_legs = 108) : 
  ∃ (chickens rabbits : ℕ), 
    chickens + rabbits = total_animals ∧ 
    2 * chickens + 4 * rabbits = total_legs ∧ 
    chickens = 26 ∧ 
    rabbits = 14 := by
  sorry

end NUMINAMATH_CALUDE_chickens_and_rabbits_l2929_292937


namespace NUMINAMATH_CALUDE_log_inequality_condition_l2929_292992

theorem log_inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, Real.log a > Real.log b → 2*a > 2*b) ∧
  ¬(∀ a b, 2*a > 2*b → Real.log a > Real.log b) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_condition_l2929_292992


namespace NUMINAMATH_CALUDE_sector_area_l2929_292963

theorem sector_area (α : Real) (r : Real) (h1 : α = 150 * π / 180) (h2 : r = Real.sqrt 3) :
  (α * r^2) / 2 = 5 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2929_292963


namespace NUMINAMATH_CALUDE_at_least_one_half_l2929_292909

theorem at_least_one_half (x y z : ℝ) 
  (h : x + y + z - 2*(x*y + y*z + x*z) + 4*x*y*z = 1/2) :
  x = 1/2 ∨ y = 1/2 ∨ z = 1/2 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_half_l2929_292909


namespace NUMINAMATH_CALUDE_sector_area_l2929_292912

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 16) (h2 : central_angle = 2) : 
  let radius := perimeter / (2 + central_angle)
  (1/2) * central_angle * radius^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2929_292912


namespace NUMINAMATH_CALUDE_eight_people_seating_theorem_l2929_292905

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (total_people : ℕ) (restricted_people : ℕ) : ℕ :=
  factorial total_people - factorial (total_people - restricted_people + 1) * factorial restricted_people

theorem eight_people_seating_theorem :
  seating_arrangements 8 3 = 36000 :=
by sorry

end NUMINAMATH_CALUDE_eight_people_seating_theorem_l2929_292905


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2929_292969

/-- Given vectors a and b in R^2, prove that their difference has magnitude 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2929_292969


namespace NUMINAMATH_CALUDE_digit_for_divisibility_by_6_l2929_292902

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem digit_for_divisibility_by_6 :
  ∃ B : ℕ, B < 10 ∧ is_divisible_by_6 (5170 + B) ∧ (B = 2 ∨ B = 8) :=
by sorry

end NUMINAMATH_CALUDE_digit_for_divisibility_by_6_l2929_292902


namespace NUMINAMATH_CALUDE_staff_pizza_fraction_l2929_292962

theorem staff_pizza_fraction (teachers : ℕ) (staff : ℕ) (teacher_pizza_fraction : ℚ) (non_pizza_eaters : ℕ) :
  teachers = 30 →
  staff = 45 →
  teacher_pizza_fraction = 2/3 →
  non_pizza_eaters = 19 →
  (staff - (non_pizza_eaters - (teachers - teacher_pizza_fraction * teachers))) / staff = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_staff_pizza_fraction_l2929_292962


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l2929_292947

theorem ufo_convention_attendees (total : ℕ) (difference : ℕ) : 
  total = 120 → difference = 4 → 
  ∃ (male female : ℕ), 
    male + female = total ∧ 
    male = female + difference ∧ 
    male = 62 := by
  sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l2929_292947


namespace NUMINAMATH_CALUDE_square_base_exponent_l2929_292931

theorem square_base_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (a^2)^(2*b) = a^b * y^b → y = a^3 := by sorry

end NUMINAMATH_CALUDE_square_base_exponent_l2929_292931


namespace NUMINAMATH_CALUDE_last_number_is_2802_l2929_292933

/-- Represents a piece of paper with a given width and height in characters. -/
structure Paper where
  width : Nat
  height : Nat

/-- Represents the space required to write a number, including the following space. -/
def spaceRequired (n : Nat) : Nat :=
  if n < 10 then 2
  else if n < 100 then 3
  else if n < 1000 then 4
  else 5

/-- The last number that can be fully written on the paper. -/
def lastNumberWritten (p : Paper) : Nat :=
  2802

/-- Theorem stating that 2802 is the last number that can be fully written on a 100x100 character paper. -/
theorem last_number_is_2802 (p : Paper) (h1 : p.width = 100) (h2 : p.height = 100) :
  lastNumberWritten p = 2802 := by
  sorry

end NUMINAMATH_CALUDE_last_number_is_2802_l2929_292933


namespace NUMINAMATH_CALUDE_right_triangle_area_l2929_292944

/-- A right-angled triangle with an altitude from the right angle -/
structure RightTriangleWithAltitude where
  /-- The length of one leg of the triangle -/
  a : ℝ
  /-- The length of the other leg of the triangle -/
  b : ℝ
  /-- The radius of the inscribed circle in one of the smaller triangles -/
  r₁ : ℝ
  /-- The radius of the inscribed circle in the other smaller triangle -/
  r₂ : ℝ
  /-- Ensure the radii are positive -/
  h_positive_r₁ : r₁ > 0
  h_positive_r₂ : r₂ > 0
  /-- The ratio of the legs is equal to the ratio of the radii -/
  h_ratio : a / b = r₁ / r₂

/-- The theorem stating the area of the right-angled triangle -/
theorem right_triangle_area (t : RightTriangleWithAltitude) (h_r₁ : t.r₁ = 3) (h_r₂ : t.r₂ = 4) : 
  (1/2) * t.a * t.b = 150 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_area_l2929_292944


namespace NUMINAMATH_CALUDE_slope_dividing_area_l2929_292903

-- Define the vertices of the L-shaped region
def vertices : List (ℝ × ℝ) := [(0, 0), (0, 4), (4, 4), (4, 2), (7, 2), (7, 0)]

-- Define the L-shaped region
def l_shape (x y : ℝ) : Prop :=
  (0 ≤ x ∧ x ≤ 7 ∧ 0 ≤ y ∧ y ≤ 4) ∧
  (x ≤ 4 ∨ y ≤ 2)

-- Define the area of the L-shaped region
def area_l_shape : ℝ := 22

-- Define a line through the origin
def line_through_origin (m : ℝ) (x y : ℝ) : Prop :=
  y = m * x

-- Define the area above the line
def area_above_line (m : ℝ) : ℝ := 11

-- Theorem: The slope of the line that divides the area in half is -0.375
theorem slope_dividing_area :
  ∃ (m : ℝ), m = -0.375 ∧
    area_above_line m = area_l_shape / 2 ∧
    ∀ (x y : ℝ), l_shape x y → line_through_origin m x y →
      (y ≥ m * x → area_above_line m ≥ area_l_shape / 2) ∧
      (y ≤ m * x → area_above_line m ≤ area_l_shape / 2) :=
by sorry

end NUMINAMATH_CALUDE_slope_dividing_area_l2929_292903


namespace NUMINAMATH_CALUDE_day_284_is_saturday_l2929_292934

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week for a given day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_284_is_saturday (h : dayOfWeek 25 = DayOfWeek.Saturday) :
  dayOfWeek 284 = DayOfWeek.Saturday := by
  sorry

end NUMINAMATH_CALUDE_day_284_is_saturday_l2929_292934


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2929_292997

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 64 * π) :
  2 * π * r^2 + π * r^2 = 192 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2929_292997


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2929_292990

-- Define the sets P and Q
def P : Set ℝ := {x | |x| < 2}
def Q : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2) + 1}

-- State the theorem
theorem intersection_P_Q :
  P ∩ Q = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2929_292990


namespace NUMINAMATH_CALUDE_intersection_line_of_given_circles_l2929_292940

/-- Circle with center and radius --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line equation of the form ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The intersection line of two circles --/
def intersection_line (c1 c2 : Circle) : Line :=
  sorry

theorem intersection_line_of_given_circles :
  let c1 : Circle := { center := (1, 5), radius := 7 }
  let c2 : Circle := { center := (-2, -1), radius := 5 * Real.sqrt 2 }
  let l : Line := intersection_line c1 c2
  l.a = 1 ∧ l.b = 1 ∧ l.c = 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_of_given_circles_l2929_292940


namespace NUMINAMATH_CALUDE_imaginary_part_of_1_plus_2i_l2929_292932

theorem imaginary_part_of_1_plus_2i :
  Complex.im (1 + 2*I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_1_plus_2i_l2929_292932


namespace NUMINAMATH_CALUDE_ratio_transformation_l2929_292972

/-- Given an original ratio of 2:3, prove that adding 2 to each term results in a ratio of 4:5 -/
theorem ratio_transformation (x : ℚ) : x = 2 → (2 + x) / (3 + x) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transformation_l2929_292972


namespace NUMINAMATH_CALUDE_polynomial_symmetry_representation_l2929_292943

theorem polynomial_symmetry_representation
  (p : ℝ → ℝ) (a : ℝ)
  (h_symmetry : ∀ x, p x = p (a - x)) :
  ∃ h : ℝ → ℝ, ∀ x, p x = h ((x - a / 2) ^ 2) :=
sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_representation_l2929_292943


namespace NUMINAMATH_CALUDE_max_value_d_l2929_292918

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10) 
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_d_l2929_292918


namespace NUMINAMATH_CALUDE_xiao_wang_total_score_l2929_292954

/-- Xiao Wang's jump rope scores -/
def score1 : ℕ := 23
def score2 : ℕ := 34
def score3 : ℕ := 29

/-- Theorem: The sum of Xiao Wang's three jump rope scores equals 86 -/
theorem xiao_wang_total_score : score1 + score2 + score3 = 86 := by
  sorry

end NUMINAMATH_CALUDE_xiao_wang_total_score_l2929_292954


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2929_292971

theorem rationalize_denominator :
  ∃ (A B C D : ℕ),
    (A = 25 ∧ B = 2 ∧ C = 20 ∧ D = 17) ∧
    D > 0 ∧
    (∀ p : ℕ, Prime p → ¬(p^2 ∣ B)) ∧
    (Real.sqrt 50 / (Real.sqrt 25 - 2 * Real.sqrt 2) = (A * Real.sqrt B + C) / D) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2929_292971


namespace NUMINAMATH_CALUDE_min_sum_given_log_condition_l2929_292926

theorem min_sum_given_log_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : Real.log a / Real.log 4 + Real.log b / Real.log 4 ≥ 5) : 
  a + b ≥ 64 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    Real.log a₀ / Real.log 4 + Real.log b₀ / Real.log 4 ≥ 5 ∧ a₀ + b₀ = 64 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_log_condition_l2929_292926


namespace NUMINAMATH_CALUDE_minimum_h_22_l2929_292966

def IsTenuous (h : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, h x + h y > (y : ℤ)^2

def SumUpTo30 (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 30).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem minimum_h_22 (h : ℕ+ → ℤ) (h_tenuous : IsTenuous h) 
    (h_min : ∀ g : ℕ+ → ℤ, IsTenuous g → SumUpTo30 h ≤ SumUpTo30 g) :
    h ⟨22, by norm_num⟩ ≥ 357 := by
  sorry

end NUMINAMATH_CALUDE_minimum_h_22_l2929_292966


namespace NUMINAMATH_CALUDE_matrix_power_result_l2929_292981

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec ![4, -1] = ![12, -3]) :
  (B ^ 4).mulVec ![4, -1] = ![324, -81] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_result_l2929_292981


namespace NUMINAMATH_CALUDE_initial_boarders_count_l2929_292977

theorem initial_boarders_count (initial_boarders day_students new_boarders : ℕ) : 
  initial_boarders > 0 ∧ 
  day_students > 0 ∧
  new_boarders = 66 ∧
  initial_boarders * 12 = day_students * 5 ∧
  (initial_boarders + new_boarders) * 2 = day_students * 1 →
  initial_boarders = 330 := by
sorry

end NUMINAMATH_CALUDE_initial_boarders_count_l2929_292977


namespace NUMINAMATH_CALUDE_cos_2017pi_minus_2alpha_l2929_292904

theorem cos_2017pi_minus_2alpha (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 2) :
  Real.cos (2017 * π - 2 * α) = 1/2 := by sorry

end NUMINAMATH_CALUDE_cos_2017pi_minus_2alpha_l2929_292904


namespace NUMINAMATH_CALUDE_intersection_difference_l2929_292961

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

theorem intersection_difference :
  ∃ (a b c d : ℝ),
    (parabola1 a = parabola2 a) ∧
    (parabola1 c = parabola2 c) ∧
    (c ≥ a) ∧
    (c - a = 2/5) :=
by sorry

end NUMINAMATH_CALUDE_intersection_difference_l2929_292961


namespace NUMINAMATH_CALUDE_imaginary_part_i_2015_l2929_292922

theorem imaginary_part_i_2015 : Complex.im (Complex.I ^ 2015) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_i_2015_l2929_292922


namespace NUMINAMATH_CALUDE_percentage_loss_calculation_l2929_292942

def cost_price : ℝ := 800
def selling_price : ℝ := 680

theorem percentage_loss_calculation : 
  (cost_price - selling_price) / cost_price * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_percentage_loss_calculation_l2929_292942


namespace NUMINAMATH_CALUDE_variance_transformation_l2929_292930

-- Define a type for our dataset
def Dataset := Fin 10 → ℝ

-- Define the variance of a dataset
noncomputable def variance (data : Dataset) : ℝ := sorry

-- State the theorem
theorem variance_transformation (data : Dataset) :
  variance data = 3 →
  variance (fun i => 2 * (data i) + 3) = 12 := by sorry

end NUMINAMATH_CALUDE_variance_transformation_l2929_292930


namespace NUMINAMATH_CALUDE_radical_simplification_l2929_292906

theorem radical_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (45 * y) * Real.sqrt (18 * y) * Real.sqrt (22 * y) = 18 * y * Real.sqrt (55 * y) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l2929_292906


namespace NUMINAMATH_CALUDE_remainder_11_power_4001_mod_13_l2929_292979

theorem remainder_11_power_4001_mod_13 : 11^4001 % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_power_4001_mod_13_l2929_292979


namespace NUMINAMATH_CALUDE_todd_total_gum_l2929_292945

/-- The number of gum pieces Todd has now, given his initial amount and the amount he received. -/
def total_gum (initial : ℕ) (received : ℕ) : ℕ := initial + received

/-- Todd's initial number of gum pieces -/
def todd_initial : ℕ := 38

/-- Number of gum pieces Todd received from Steve -/
def steve_gave : ℕ := 16

/-- Theorem stating that Todd's total gum pieces is 54 -/
theorem todd_total_gum : total_gum todd_initial steve_gave = 54 := by
  sorry

end NUMINAMATH_CALUDE_todd_total_gum_l2929_292945


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2929_292938

/-- The line passing through a fixed point for all real values of a -/
def line (a x y : ℝ) : Prop := (a - 1) * x + a * y + 3 = 0

/-- The fixed point through which the line passes -/
def fixed_point : ℝ × ℝ := (3, -3)

/-- Theorem stating that the fixed point lies on the line for all real a -/
theorem fixed_point_on_line :
  ∀ a : ℝ, line a (fixed_point.1) (fixed_point.2) := by
sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2929_292938


namespace NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l2929_292964

theorem chocolate_bars_in_large_box :
  let small_boxes : ℕ := 16
  let bars_per_small_box : ℕ := 25
  let total_bars : ℕ := small_boxes * bars_per_small_box
  total_bars = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l2929_292964


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2929_292913

theorem rationalize_denominator (x : ℝ) :
  x > 0 → (45 * Real.sqrt 3) / Real.sqrt x = 3 * Real.sqrt 15 ↔ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2929_292913


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2929_292949

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (9 * z) / (3 * x + 2 * y) + (9 * x) / (2 * y + 3 * z) + (4 * y) / (2 * x + z) ≥ 9 / 2 :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    (9 * z) / (3 * x + 2 * y) + (9 * x) / (2 * y + 3 * z) + (4 * y) / (2 * x + z) = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2929_292949


namespace NUMINAMATH_CALUDE_investment_interest_l2929_292968

/-- Calculate simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_interest (y : ℝ) : 
  simple_interest 3000 (y / 100) 2 = 60 * y := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_l2929_292968


namespace NUMINAMATH_CALUDE_fundraising_shortfall_l2929_292955

def goal : ℕ := 10000

def ken_raised : ℕ := 800

theorem fundraising_shortfall (mary_raised scott_raised amy_raised : ℕ) 
  (h1 : mary_raised = 5 * ken_raised)
  (h2 : mary_raised = 3 * scott_raised)
  (h3 : amy_raised = 2 * ken_raised)
  (h4 : amy_raised = scott_raised / 2)
  : ken_raised + mary_raised + scott_raised + amy_raised = goal - 400 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_shortfall_l2929_292955


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l2929_292925

theorem quadratic_integer_roots (p q : ℤ) :
  ∀ n : ℕ, n ≤ 9 →
  ∃ x y : ℤ, x^2 + (p + n) * x + (q + n) = 0 ∧
             y^2 + (p + n) * y + (q + n) = 0 ∧
             x ≠ y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l2929_292925


namespace NUMINAMATH_CALUDE_one_greater_than_digit_squares_l2929_292988

def digit_squares_sum (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d => d^2) |>.sum

theorem one_greater_than_digit_squares : {n : ℕ | n > 0 ∧ n = digit_squares_sum n + 1} = {35, 75} := by
  sorry

end NUMINAMATH_CALUDE_one_greater_than_digit_squares_l2929_292988


namespace NUMINAMATH_CALUDE_factorizable_polynomial_l2929_292919

theorem factorizable_polynomial (x y a b : ℝ) : 
  ∃ (p q : ℝ), x^2 - x + (1/4) = (p - q)^2 ∧ 
  (∀ (r s : ℝ), 4*x^2 + 1 ≠ (r - s)^2) ∧
  (∀ (r s : ℝ), 9*a^2*b^2 - 3*a*b + 1 ≠ (r - s)^2) ∧
  (∀ (r s : ℝ), -x^2 - y^2 ≠ (r - s)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_factorizable_polynomial_l2929_292919


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l2929_292980

/-- Given a train of length 1200 meters that crosses a tree in 120 seconds,
    calculate the time required to pass a platform of length 900 meters. -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (tree_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200) 
  (h2 : tree_passing_time = 120) 
  (h3 : platform_length = 900) :
  (train_length + platform_length) / (train_length / tree_passing_time) = 210 := by
  sorry

#check train_platform_passing_time

end NUMINAMATH_CALUDE_train_platform_passing_time_l2929_292980


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l2929_292924

def f (x : ℝ) : ℝ := |x - 4| - 2 + 5

theorem minimum_point_of_translated_graph :
  ∀ x : ℝ, f x ≥ f 4 ∧ f 4 = 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l2929_292924


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2929_292928

theorem quadratic_equation_properties (m : ℝ) (hm : m ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - (m^2 + 2)*x + m^2 + 1
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - 2*x₁ - 1 = m^2 - 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2929_292928


namespace NUMINAMATH_CALUDE_max_gold_coins_l2929_292956

theorem max_gold_coins (n : ℕ) (h1 : n < 150) 
  (h2 : ∃ (k : ℕ), n = 13 * k + 3) : n ≤ 143 :=
by
  sorry

end NUMINAMATH_CALUDE_max_gold_coins_l2929_292956


namespace NUMINAMATH_CALUDE_ninas_age_l2929_292911

theorem ninas_age (lisa mike nina : ℝ) 
  (h1 : (lisa + mike + nina) / 3 = 12)
  (h2 : nina - 5 = 2 * lisa)
  (h3 : mike + 2 = (lisa + 2) / 2) :
  nina = 34.6 := by
  sorry

end NUMINAMATH_CALUDE_ninas_age_l2929_292911


namespace NUMINAMATH_CALUDE_marble_draw_probability_l2929_292984

/-- The probability of drawing one red marble followed by one blue marble without replacement -/
theorem marble_draw_probability (red blue yellow : ℕ) (h_red : red = 4) (h_blue : blue = 3) (h_yellow : yellow = 6) :
  (red : ℚ) / (red + blue + yellow) * blue / (red + blue + yellow - 1) = 1 / 13 := by sorry

end NUMINAMATH_CALUDE_marble_draw_probability_l2929_292984


namespace NUMINAMATH_CALUDE_sin_shift_l2929_292946

theorem sin_shift (x : ℝ) : Real.sin x = Real.sin (2 * (x - π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l2929_292946


namespace NUMINAMATH_CALUDE_solve_equation_l2929_292958

theorem solve_equation (x : ℝ) : 0.3 * x = 45 → (10 / 3) * (0.3 * x) = 150 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2929_292958


namespace NUMINAMATH_CALUDE_cookie_count_l2929_292993

/-- The number of cookies Paul and Paula bought together -/
def total_cookies (paul_cookies paula_cookies : ℕ) : ℕ :=
  paul_cookies + paula_cookies

/-- Theorem: Paul and Paula bought 87 cookies in total -/
theorem cookie_count : ∃ (paula_cookies : ℕ),
  (paula_cookies = 45 - 3) ∧ (total_cookies 45 paula_cookies = 87) :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l2929_292993


namespace NUMINAMATH_CALUDE_triangle_count_is_68_l2929_292927

/-- Represents a grid-divided rectangle with diagonals -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  vertical_divisions : ℕ
  horizontal_divisions : ℕ
  has_corner_diagonals : Bool
  has_midpoint_diagonals : Bool
  has_full_diagonal : Bool

/-- Counts the number of triangles in a GridRectangle -/
def count_triangles (rect : GridRectangle) : ℕ :=
  sorry

/-- The specific rectangle from the problem -/
def problem_rectangle : GridRectangle :=
  { width := 40
  , height := 30
  , vertical_divisions := 3
  , horizontal_divisions := 2
  , has_corner_diagonals := true
  , has_midpoint_diagonals := true
  , has_full_diagonal := true }

theorem triangle_count_is_68 : count_triangles problem_rectangle = 68 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_is_68_l2929_292927


namespace NUMINAMATH_CALUDE_certain_number_problem_l2929_292999

theorem certain_number_problem : ∃ x : ℕ, 3*15 + 3*16 + 3*19 + x = 161 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2929_292999


namespace NUMINAMATH_CALUDE_unfair_die_theorem_l2929_292936

def unfair_die_expected_value (p1to6 p7 p8 : ℚ) : ℚ :=
  (1 * p1to6 + 2 * p1to6 + 3 * p1to6 + 4 * p1to6 + 5 * p1to6 + 6 * p1to6) +
  (7 * p7) + (8 * p8)

theorem unfair_die_theorem :
  let p1to6 : ℚ := 1 / 15
  let p7 : ℚ := 1 / 6
  let p8 : ℚ := 1 / 3
  unfair_die_expected_value p1to6 p7 p8 = 157 / 30 :=
by
  sorry

#eval unfair_die_expected_value (1/15) (1/6) (1/3)

end NUMINAMATH_CALUDE_unfair_die_theorem_l2929_292936


namespace NUMINAMATH_CALUDE_range_equivalence_l2929_292960

/-- The set of real numbers satisfying the given conditions -/
def A : Set ℝ :=
  {a | ∀ x, (x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 2*x - 8 > 0) ∧
    ∃ y, (y^2 - 4*a*y + 3*a^2 ≥ 0 ∧ y^2 + 2*y - 8 > 0)}

/-- The theorem stating the equivalence of set A and the expected range -/
theorem range_equivalence : A = {a : ℝ | a ≤ -4 ∨ a ≥ 2 ∨ a = 0} := by
  sorry

end NUMINAMATH_CALUDE_range_equivalence_l2929_292960


namespace NUMINAMATH_CALUDE_opposite_value_implies_ab_zero_l2929_292975

/-- Given that for all x, a(-x) + b(-x)^2 = -(ax + bx^2), prove that ab = 0 -/
theorem opposite_value_implies_ab_zero (a b : ℝ) 
  (h : ∀ x : ℝ, a * (-x) + b * (-x)^2 = -(a * x + b * x^2)) : 
  a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_value_implies_ab_zero_l2929_292975


namespace NUMINAMATH_CALUDE_sqrt_90000_equals_300_l2929_292994

theorem sqrt_90000_equals_300 : Real.sqrt 90000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_90000_equals_300_l2929_292994


namespace NUMINAMATH_CALUDE_mashas_dolls_l2929_292935

theorem mashas_dolls (n : ℕ) : 
  (n / 2 : ℚ) * 1 + (n / 4 : ℚ) * 2 + (n / 4 : ℚ) * 4 = 24 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_mashas_dolls_l2929_292935


namespace NUMINAMATH_CALUDE_jade_handled_83_transactions_l2929_292907

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + mabel_transactions / 10
def cal_transactions : ℕ := anthony_transactions * 2 / 3
def jade_transactions : ℕ := cal_transactions + 17

-- Theorem to prove
theorem jade_handled_83_transactions : jade_transactions = 83 := by
  sorry

end NUMINAMATH_CALUDE_jade_handled_83_transactions_l2929_292907


namespace NUMINAMATH_CALUDE_special_sequence_sum_l2929_292900

/-- A sequence with specific initial conditions -/
def special_sequence : ℕ → ℚ := sorry

/-- The sum of the first n terms of the special sequence -/
def sum_n (n : ℕ) : ℚ := sorry

theorem special_sequence_sum :
  (special_sequence 1 = 2) →
  (sum_n 2 = 8) →
  (sum_n 3 = 20) →
  ∀ n : ℕ, sum_n n = n * (n + 1) * (2 * n + 4) / 3 := by sorry

end NUMINAMATH_CALUDE_special_sequence_sum_l2929_292900


namespace NUMINAMATH_CALUDE_square_side_length_l2929_292950

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) :
  rectangle_length = 9 →
  rectangle_width = 16 →
  rectangle_length * rectangle_width = square_side * square_side →
  square_side = 12 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l2929_292950


namespace NUMINAMATH_CALUDE_faye_coloring_books_l2929_292923

/-- The number of coloring books Faye gave away -/
def books_given_away : ℕ := sorry

/-- The initial number of coloring books Faye had -/
def initial_books : ℕ := 34

/-- The number of coloring books Faye bought -/
def books_bought : ℕ := 48

/-- The final number of coloring books Faye has -/
def final_books : ℕ := 79

theorem faye_coloring_books : 
  initial_books - books_given_away + books_bought = final_books ∧ 
  books_given_away = 3 := by sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l2929_292923


namespace NUMINAMATH_CALUDE_min_pressure_cyclic_process_l2929_292948

-- Define the constants and variables
variable (V₀ T₀ a b c R : ℝ)
variable (V T P : ℝ → ℝ)

-- Define the cyclic process equation
def cyclic_process (t : ℝ) : Prop :=
  ((V t) / V₀ - a)^2 + ((T t) / T₀ - b)^2 = c^2

-- Define the ideal gas law
def ideal_gas_law (t : ℝ) : Prop :=
  (P t) * (V t) = R * (T t)

-- State the theorem
theorem min_pressure_cyclic_process
  (h1 : ∀ t, cyclic_process V₀ T₀ a b c V T t)
  (h2 : ∀ t, ideal_gas_law R V T P t)
  (h3 : c^2 < a^2 + b^2) :
  ∃ P_min : ℝ, ∀ t, P t ≥ P_min ∧ 
    P_min = (R * T₀ / V₀) * (a * Real.sqrt (a^2 + b^2 - c^2) - b * c) / 
      (b * Real.sqrt (a^2 + b^2 - c^2) + a * c) :=
sorry

end NUMINAMATH_CALUDE_min_pressure_cyclic_process_l2929_292948


namespace NUMINAMATH_CALUDE_max_value_x_plus_inverse_l2929_292989

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (x + 1/x) ≤ Real.sqrt 15 ∧ ∃ y : ℝ, 13 = y^2 + 1/y^2 ∧ y + 1/y = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_inverse_l2929_292989


namespace NUMINAMATH_CALUDE_vacation_duration_l2929_292959

-- Define the parameters
def miles_per_day : ℕ := 250
def total_miles : ℕ := 1250

-- Theorem statement
theorem vacation_duration :
  total_miles / miles_per_day = 5 :=
sorry

end NUMINAMATH_CALUDE_vacation_duration_l2929_292959
