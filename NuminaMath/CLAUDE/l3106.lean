import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l3106_310610

theorem problem_solution (x : ℝ) (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 24) :
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 685 / 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3106_310610


namespace NUMINAMATH_CALUDE_triangle_special_case_l3106_310681

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_special_case (t : Triangle) 
  (h1 : t.A - t.C = π / 2)  -- A - C = 90°
  (h2 : t.a + t.c = Real.sqrt 2 * t.b)  -- a + c = √2 * b
  : t.C = π / 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_case_l3106_310681


namespace NUMINAMATH_CALUDE_candy_store_revenue_l3106_310601

/-- Calculates the revenue of a candy store given specific sales conditions --/
theorem candy_store_revenue :
  let fudge_pounds : ℕ := 37
  let fudge_price : ℚ := 5/2
  let truffle_count : ℕ := 82
  let truffle_price : ℚ := 3/2
  let pretzel_count : ℕ := 48
  let pretzel_price : ℚ := 2
  let fudge_discount : ℚ := 1/10
  let sales_tax : ℚ := 1/20
  let truffle_promo : ℕ := 3  -- buy 3, get 1 free

  let fudge_revenue := (1 - fudge_discount) * (fudge_pounds : ℚ) * fudge_price
  let truffle_revenue := (truffle_count - truffle_count / (truffle_promo + 1)) * truffle_price
  let pretzel_revenue := (pretzel_count : ℚ) * pretzel_price
  
  let total_before_tax := fudge_revenue + truffle_revenue + pretzel_revenue
  let total_after_tax := total_before_tax * (1 + sales_tax)

  total_after_tax = 28586 / 100
  := by sorry

end NUMINAMATH_CALUDE_candy_store_revenue_l3106_310601


namespace NUMINAMATH_CALUDE_total_highlighters_l3106_310631

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : pink = 10) (h2 : yellow = 15) (h3 : blue = 8) : 
  pink + yellow + blue = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l3106_310631


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3106_310641

def total_balls (a : ℕ) : ℕ := 2 + 3 + a

def probability_red (a : ℕ) : ℚ := 2 / total_balls a

theorem yellow_balls_count : ∃ a : ℕ, probability_red a = 1/3 ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3106_310641


namespace NUMINAMATH_CALUDE_solution_system_equations_l3106_310639

theorem solution_system_equations :
  ∃! (x y : ℝ), 
    x + Real.sqrt (x + 2*y) - 2*y = 7/2 ∧
    x^2 + x + 2*y - 4*y^2 = 27/2 ∧
    x = 19/4 ∧ y = 17/8 := by
  sorry

end NUMINAMATH_CALUDE_solution_system_equations_l3106_310639


namespace NUMINAMATH_CALUDE_hot_dog_eating_contest_l3106_310663

theorem hot_dog_eating_contest (first_competitor second_competitor third_competitor : ℕ) :
  first_competitor = 12 →
  third_competitor = 18 →
  third_competitor = (3 * second_competitor) / 4 →
  second_competitor / first_competitor = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_eating_contest_l3106_310663


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3106_310646

/-- The sum of the infinite series ∑_{n=1}^∞ (2n^2 + n + 1) / (n(n+1)(n+2)) is equal to 5/6. -/
theorem infinite_series_sum : 
  ∑' n : ℕ+, (2 * n.val^2 + n.val + 1) / (n.val * (n.val + 1) * (n.val + 2)) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3106_310646


namespace NUMINAMATH_CALUDE_sum_of_distances_eq_three_halves_side_length_l3106_310685

/-- An equilateral triangle with a point inside it -/
structure EquilateralTriangleWithPoint where
  /-- The side length of the equilateral triangle -/
  a : ℝ
  /-- The point inside the triangle -/
  M : ℝ × ℝ
  /-- Assertion that the triangle is equilateral with side length a -/
  is_equilateral : a > 0
  /-- Assertion that M is inside the triangle -/
  M_inside : True  -- This is a simplification; in a real proof, we'd need to define this properly

/-- The sum of distances from a point to the sides of an equilateral triangle -/
def sum_of_distances (t : EquilateralTriangleWithPoint) : ℝ :=
  sorry  -- The actual calculation would go here

/-- Theorem: The sum of distances from any point inside an equilateral triangle
    to its sides is equal to 3/2 times the side length -/
theorem sum_of_distances_eq_three_halves_side_length (t : EquilateralTriangleWithPoint) :
  sum_of_distances t = 3/2 * t.a := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_eq_three_halves_side_length_l3106_310685


namespace NUMINAMATH_CALUDE_longest_chord_in_circle_l3106_310636

theorem longest_chord_in_circle (r : ℝ) (h : r = 3) : 
  ∃ (c : ℝ), c = 6 ∧ ∀ (chord : ℝ), chord ≤ c :=
sorry

end NUMINAMATH_CALUDE_longest_chord_in_circle_l3106_310636


namespace NUMINAMATH_CALUDE_total_students_count_l3106_310678

/-- The number of students who wish to go on a scavenger hunting trip -/
def scavenger_students : ℕ := 4000

/-- The number of students who wish to go on a skiing trip -/
def skiing_students : ℕ := 2 * scavenger_students

/-- The total number of students who wish to go on either trip -/
def total_students : ℕ := scavenger_students + skiing_students

theorem total_students_count : total_students = 12000 := by
  sorry

end NUMINAMATH_CALUDE_total_students_count_l3106_310678


namespace NUMINAMATH_CALUDE_flute_ratio_is_two_to_one_l3106_310650

/-- Represents the number of musical instruments owned by a person -/
structure Instruments where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- The total number of instruments owned by a person -/
def total_instruments (i : Instruments) : ℕ :=
  i.flutes + i.horns + i.harps

/-- Charlie's instruments -/
def charlie : Instruments :=
  { flutes := 1, horns := 2, harps := 1 }

/-- Carli's instruments in terms of F (number of flutes) -/
def carli (F : ℕ) : Instruments :=
  { flutes := F, horns := charlie.horns / 2, harps := 0 }

/-- The theorem to be proved -/
theorem flute_ratio_is_two_to_one :
  ∃ F : ℕ, 
    (total_instruments charlie + total_instruments (carli F) = 7) ∧ 
    ((carli F).flutes : ℚ) / charlie.flutes = 2 := by
  sorry

end NUMINAMATH_CALUDE_flute_ratio_is_two_to_one_l3106_310650


namespace NUMINAMATH_CALUDE_potato_fries_price_l3106_310633

/-- The price of a pack of potato fries given Einstein's fundraising scenario -/
theorem potato_fries_price (total_goal : ℚ) (pizza_price : ℚ) (soda_price : ℚ)
  (pizzas_sold : ℕ) (fries_sold : ℕ) (sodas_sold : ℕ) (remaining : ℚ)
  (h1 : total_goal = 500)
  (h2 : pizza_price = 12)
  (h3 : soda_price = 2)
  (h4 : pizzas_sold = 15)
  (h5 : fries_sold = 40)
  (h6 : sodas_sold = 25)
  (h7 : remaining = 258)
  : (total_goal - remaining - (pizza_price * pizzas_sold + soda_price * sodas_sold)) / fries_sold = (3 / 10) :=
sorry

end NUMINAMATH_CALUDE_potato_fries_price_l3106_310633


namespace NUMINAMATH_CALUDE_towel_area_decrease_l3106_310642

theorem towel_area_decrease (length width : ℝ) (h1 : length > 0) (h2 : width > 0) :
  let new_length := 0.9 * length
  let new_width := 0.8 * width
  let original_area := length * width
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.28 := by
sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l3106_310642


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_and_prime_l3106_310644

theorem largest_four_digit_divisible_by_88_and_prime : ∃ (p : ℕ), 
  p.Prime ∧ 
  p > 100 ∧ 
  9944 % 88 = 0 ∧ 
  9944 % p = 0 ∧ 
  ∀ (n : ℕ), n > 9944 → n < 10000 → ¬(n % 88 = 0 ∧ ∃ (q : ℕ), q.Prime ∧ q > 100 ∧ n % q = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_and_prime_l3106_310644


namespace NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l3106_310657

theorem triangle_area_with_cosine_root (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → b = 5 → (5 * cos_theta^2 - 7 * cos_theta - 6 = 0) → 
  (1/2 : ℝ) * a * b * cos_theta = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l3106_310657


namespace NUMINAMATH_CALUDE_proportional_segments_l3106_310602

theorem proportional_segments (a b c d : ℝ) :
  b = 3 → c = 4 → d = 6 → (a / b = c / d) → a = 2 := by sorry

end NUMINAMATH_CALUDE_proportional_segments_l3106_310602


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3106_310628

theorem min_value_of_expression (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z) ≥ 36 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 1 ∧ (1 / a + 4 / b + 9 / c = 36) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3106_310628


namespace NUMINAMATH_CALUDE_circle_center_l3106_310656

/-- The center of the circle x^2 + y^2 - 2x + 4y + 3 = 0 is at the point (1, -2). -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y + 3 = 0) → 
  ∃ (h : ℝ), (x - 1)^2 + (y + 2)^2 = h^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l3106_310656


namespace NUMINAMATH_CALUDE_unique_prime_permutation_residue_system_l3106_310659

theorem unique_prime_permutation_residue_system : ∃! (p : ℕ), 
  Nat.Prime p ∧ 
  p % 2 = 1 ∧
  ∃ (b : Fin (p - 1) → Fin (p - 1)), Function.Bijective b ∧
    (∀ (x : Fin (p - 1)), 
      ∃ (y : Fin (p - 1)), (x.val + 1) ^ (b y).val ≡ (y.val + 1) [ZMOD p]) ∧
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_permutation_residue_system_l3106_310659


namespace NUMINAMATH_CALUDE_mr_callen_loss_l3106_310676

def paintings_count : ℕ := 15
def paintings_price : ℚ := 60
def wooden_toys_count : ℕ := 12
def wooden_toys_price : ℚ := 25
def hats_count : ℕ := 20
def hats_price : ℚ := 15

def paintings_loss_percentage : ℚ := 18 / 100
def wooden_toys_loss_percentage : ℚ := 25 / 100
def hats_loss_percentage : ℚ := 10 / 100

def total_cost : ℚ := 
  paintings_count * paintings_price + 
  wooden_toys_count * wooden_toys_price + 
  hats_count * hats_price

def total_selling_price : ℚ := 
  paintings_count * paintings_price * (1 - paintings_loss_percentage) +
  wooden_toys_count * wooden_toys_price * (1 - wooden_toys_loss_percentage) +
  hats_count * hats_price * (1 - hats_loss_percentage)

def total_loss : ℚ := total_cost - total_selling_price

theorem mr_callen_loss : total_loss = 267 := by
  sorry

end NUMINAMATH_CALUDE_mr_callen_loss_l3106_310676


namespace NUMINAMATH_CALUDE_divisible_by_30_implies_x_is_0_l3106_310699

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

def four_digit_number (x : ℕ) : ℕ := x * 1000 + 240 + x

theorem divisible_by_30_implies_x_is_0 (x : ℕ) (h : x < 10) :
  is_divisible_by (four_digit_number x) 30 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_30_implies_x_is_0_l3106_310699


namespace NUMINAMATH_CALUDE_tan_45_degrees_l3106_310603

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l3106_310603


namespace NUMINAMATH_CALUDE_f_properties_l3106_310698

-- Define the function f
def f (x : ℝ) : ℝ := x * (2 * x^2 - 3 * x - 12) + 5

-- Define the interval
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem f_properties :
  -- 1. Tangent line at x = 1
  (∃ (m c : ℝ), ∀ x y, y = f x → (x - 1) * (f 1 - y) = m * (x - 1)^2 + c * (x - 1) 
                     ∧ m = -12 ∧ c = 0) ∧
  -- 2. Maximum value
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 5) ∧
  -- 3. Minimum value
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = -15) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3106_310698


namespace NUMINAMATH_CALUDE_root_cubic_expression_l3106_310632

theorem root_cubic_expression (m : ℝ) : 
  m^2 + 3*m - 2023 = 0 → m^3 + 2*m^2 - 2026*m - 2023 = -4046 := by
  sorry

end NUMINAMATH_CALUDE_root_cubic_expression_l3106_310632


namespace NUMINAMATH_CALUDE_expensive_module_cost_l3106_310672

/-- Proves the cost of the more expensive module in a Bluetooth device assembly factory -/
theorem expensive_module_cost :
  let cheaper_cost : ℝ := 2.5
  let total_stock_value : ℝ := 62.5
  let total_modules : ℕ := 22
  let cheaper_modules : ℕ := 21
  let expensive_modules : ℕ := total_modules - cheaper_modules
  expensive_modules * expensive_cost + cheaper_modules * cheaper_cost = total_stock_value →
  expensive_cost = 10 := by
  sorry

#check expensive_module_cost

end NUMINAMATH_CALUDE_expensive_module_cost_l3106_310672


namespace NUMINAMATH_CALUDE_price_ratio_theorem_l3106_310679

theorem price_ratio_theorem (CP : ℝ) (SP1 SP2 : ℝ) 
  (h1 : SP1 = CP * (1 + 0.2))
  (h2 : SP2 = CP * (1 - 0.2)) :
  SP2 / SP1 = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_price_ratio_theorem_l3106_310679


namespace NUMINAMATH_CALUDE_correct_number_of_vans_l3106_310620

/-- The number of vans taken on a field trip -/
def number_of_vans : ℕ := 2

/-- The total number of people on the field trip -/
def total_people : ℕ := 76

/-- The number of buses taken on the field trip -/
def number_of_buses : ℕ := 3

/-- The number of people each bus can hold -/
def people_per_bus : ℕ := 20

/-- The number of people each van can hold -/
def people_per_van : ℕ := 8

/-- Theorem stating that the number of vans is correct given the conditions -/
theorem correct_number_of_vans : 
  number_of_vans * people_per_van + number_of_buses * people_per_bus = total_people :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_vans_l3106_310620


namespace NUMINAMATH_CALUDE_melanie_caught_ten_l3106_310615

/-- The number of trout Sara caught -/
def sara_trout : ℕ := 5

/-- The factor by which Melanie's catch exceeds Sara's -/
def melanie_factor : ℕ := 2

/-- The number of trout Melanie caught -/
def melanie_trout : ℕ := melanie_factor * sara_trout

theorem melanie_caught_ten : melanie_trout = 10 := by
  sorry

end NUMINAMATH_CALUDE_melanie_caught_ten_l3106_310615


namespace NUMINAMATH_CALUDE_mark_bananas_equal_mike_matt_fruits_l3106_310680

/-- Represents the number of fruits each child received -/
structure FruitDistribution where
  mike_oranges : ℕ
  matt_apples : ℕ
  mark_bananas : ℕ

/-- The fruit distribution problem -/
def annie_fruit_problem (fd : FruitDistribution) : Prop :=
  fd.mike_oranges = 3 ∧
  fd.matt_apples = 2 * fd.mike_oranges ∧
  fd.mike_oranges + fd.matt_apples + fd.mark_bananas = 18

/-- The theorem stating the relationship between Mark's bananas and the total fruits of Mike and Matt -/
theorem mark_bananas_equal_mike_matt_fruits (fd : FruitDistribution) 
  (h : annie_fruit_problem fd) : 
  fd.mark_bananas = fd.mike_oranges + fd.matt_apples := by
  sorry

#check mark_bananas_equal_mike_matt_fruits

end NUMINAMATH_CALUDE_mark_bananas_equal_mike_matt_fruits_l3106_310680


namespace NUMINAMATH_CALUDE_total_subjects_l3106_310693

theorem total_subjects (average_all : ℝ) (average_five : ℝ) (last_subject : ℝ) :
  average_all = 80 →
  average_five = 74 →
  last_subject = 110 →
  ∃ n : ℕ, n = 6 ∧ n * average_all = 5 * average_five + last_subject :=
by sorry

end NUMINAMATH_CALUDE_total_subjects_l3106_310693


namespace NUMINAMATH_CALUDE_zero_has_square_and_cube_root_l3106_310652

/-- A number x is a square root of y if x * x = y -/
def is_square_root (x y : ℝ) : Prop := x * x = y

/-- A number x is a cube root of y if x * x * x = y -/
def is_cube_root (x y : ℝ) : Prop := x * x * x = y

/-- 0 has both a square root and a cube root -/
theorem zero_has_square_and_cube_root :
  ∃ (x y : ℝ), is_square_root x 0 ∧ is_cube_root y 0 :=
sorry

end NUMINAMATH_CALUDE_zero_has_square_and_cube_root_l3106_310652


namespace NUMINAMATH_CALUDE_system_solution_range_l3106_310634

/-- Given a system of equations 2x + y = 1 + 4a and x + 2y = 2 - a,
    if x + y > 0, then a > -1 -/
theorem system_solution_range (x y a : ℝ) 
  (eq1 : 2 * x + y = 1 + 4 * a)
  (eq2 : x + 2 * y = 2 - a)
  (h : x + y > 0) : 
  a > -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_range_l3106_310634


namespace NUMINAMATH_CALUDE_range_of_a_l3106_310671

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3106_310671


namespace NUMINAMATH_CALUDE_expansion_properties_l3106_310622

def polynomial_expansion (x : ℝ) (a : Fin 8 → ℝ) : Prop :=
  (1 - 2*x)^7 = a 0 + a 1*x + a 2*x^2 + a 3*x^3 + a 4*x^4 + a 5*x^5 + a 6*x^6 + a 7*x^7

theorem expansion_properties (a : Fin 8 → ℝ) 
  (h : ∀ x, polynomial_expansion x a) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = -2) ∧
  (a 1 + a 3 + a 5 + a 7 = -1094) ∧
  (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| = 2187) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l3106_310622


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_one_l3106_310653

theorem no_solution_implies_a_leq_one :
  (∀ x : ℝ, ¬(x + 2 > 3 ∧ x < a)) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_one_l3106_310653


namespace NUMINAMATH_CALUDE_breaking_sequences_count_l3106_310684

/-- Represents the number of targets in each column -/
def targetDistribution : List Nat := [4, 3, 3]

/-- The total number of targets -/
def totalTargets : Nat := targetDistribution.sum

/-- Calculates the number of different sequences to break all targets -/
def breakingSequences (dist : List Nat) : Nat :=
  Nat.factorial totalTargets / (dist.map Nat.factorial).prod

theorem breaking_sequences_count : breakingSequences targetDistribution = 4200 := by
  sorry

end NUMINAMATH_CALUDE_breaking_sequences_count_l3106_310684


namespace NUMINAMATH_CALUDE_square_plate_nails_l3106_310637

/-- Calculates the total number of unique nails on a square plate -/
def total_nails (nails_per_side : ℕ) : ℕ :=
  nails_per_side * 4 - 4

/-- Theorem stating that a square plate with 25 nails per side has 96 unique nails -/
theorem square_plate_nails :
  total_nails 25 = 96 := by
  sorry

#eval total_nails 25  -- This should output 96

end NUMINAMATH_CALUDE_square_plate_nails_l3106_310637


namespace NUMINAMATH_CALUDE_reading_assignment_solution_l3106_310618

/-- Represents the reading assignment for Mrs. Reed's English class -/
structure ReadingAssignment where
  total_pages : ℕ
  alice_speed : ℕ  -- seconds per page
  bob_speed : ℕ    -- seconds per page
  chandra_speed : ℕ -- seconds per page
  x : ℕ  -- last page Alice reads
  y : ℕ  -- last page Chandra reads

/-- Checks if the reading assignment satisfies the given conditions -/
def is_valid_assignment (r : ReadingAssignment) : Prop :=
  r.total_pages = 910 ∧
  r.alice_speed = 30 ∧
  r.bob_speed = 60 ∧
  r.chandra_speed = 45 ∧
  r.x < r.y ∧
  r.y < r.total_pages ∧
  r.alice_speed * r.x = r.chandra_speed * (r.y - r.x) ∧
  r.chandra_speed * (r.y - r.x) = r.bob_speed * (r.total_pages - r.y)

/-- Theorem stating the unique solution for the reading assignment -/
theorem reading_assignment_solution (r : ReadingAssignment) :
  is_valid_assignment r → r.x = 420 ∧ r.y = 700 := by
  sorry


end NUMINAMATH_CALUDE_reading_assignment_solution_l3106_310618


namespace NUMINAMATH_CALUDE_expression_value_l3106_310675

theorem expression_value : 
  1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3106_310675


namespace NUMINAMATH_CALUDE_add_multiply_round_problem_l3106_310606

theorem add_multiply_round_problem : 
  let a := 73.5891
  let b := 24.376
  let sum := a + b
  let product := sum * 2
  (product * 100).round / 100 = 195.93 := by sorry

end NUMINAMATH_CALUDE_add_multiply_round_problem_l3106_310606


namespace NUMINAMATH_CALUDE_valentino_farm_ratio_l3106_310651

/-- Given the conditions on Mr. Valentino's farm, prove the ratio of turkeys to ducks -/
theorem valentino_farm_ratio :
  let total_birds : ℕ := 1800
  let chickens : ℕ := 200
  let ducks : ℕ := 2 * chickens
  let turkeys : ℕ := total_birds - (chickens + ducks)
  (turkeys : ℚ) / (ducks : ℚ) = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_valentino_farm_ratio_l3106_310651


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_36_l3106_310611

/-- The number of ways to assign 4 intern teachers to 3 classes, with at least 1 teacher in each class -/
def allocation_schemes : ℕ :=
  -- We define the number of allocation schemes here
  -- The actual calculation is not provided, as we're only writing the statement
  36

/-- Theorem stating that the number of allocation schemes is 36 -/
theorem allocation_schemes_eq_36 : allocation_schemes = 36 := by
  sorry


end NUMINAMATH_CALUDE_allocation_schemes_eq_36_l3106_310611


namespace NUMINAMATH_CALUDE_monthly_donation_proof_l3106_310617

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total annual donation in dollars -/
def annual_donation : ℕ := 17436

/-- The monthly donation in dollars -/
def monthly_donation : ℕ := annual_donation / months_in_year

theorem monthly_donation_proof : monthly_donation = 1453 := by
  sorry

end NUMINAMATH_CALUDE_monthly_donation_proof_l3106_310617


namespace NUMINAMATH_CALUDE_gcd_111_1850_l3106_310648

theorem gcd_111_1850 : Nat.gcd 111 1850 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_111_1850_l3106_310648


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3106_310683

theorem reciprocal_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n)⁻¹ = -n := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3106_310683


namespace NUMINAMATH_CALUDE_student_rabbit_difference_l3106_310689

/-- The number of third-grade classrooms -/
def num_classrooms : ℕ := 4

/-- The number of students in each third-grade classroom -/
def students_per_classroom : ℕ := 18

/-- The number of pet rabbits in each third-grade classroom -/
def rabbits_per_classroom : ℕ := 2

/-- The difference between the total number of students and the total number of rabbits -/
theorem student_rabbit_difference : 
  num_classrooms * students_per_classroom - num_classrooms * rabbits_per_classroom = 64 := by
  sorry

end NUMINAMATH_CALUDE_student_rabbit_difference_l3106_310689


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l3106_310669

def scores : List ℕ := [87, 90, 85, 93, 89, 92]

theorem arithmetic_mean_of_scores :
  (scores.sum : ℚ) / scores.length = 268 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l3106_310669


namespace NUMINAMATH_CALUDE_sum_of_specific_primes_l3106_310665

def smallest_odd_prime : ℕ := 3

def largest_prime_less_than_50 : ℕ := 47

def smallest_prime_greater_than_60 : ℕ := 61

theorem sum_of_specific_primes :
  smallest_odd_prime + largest_prime_less_than_50 + smallest_prime_greater_than_60 = 111 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_specific_primes_l3106_310665


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3106_310600

theorem fraction_inequality_solution_set :
  {x : ℝ | (x - 1) / x < 0} = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3106_310600


namespace NUMINAMATH_CALUDE_semesters_per_year_two_semesters_per_year_l3106_310627

/-- The number of semesters in a year for a school, given the cost per semester and total cost for 13 years -/
theorem semesters_per_year (cost_per_semester : ℕ) (total_cost : ℕ) (years : ℕ) : ℕ :=
  let total_semesters := total_cost / cost_per_semester
  total_semesters / years

/-- Proof that there are 2 semesters in a year for the given school costs -/
theorem two_semesters_per_year : 
  semesters_per_year 20000 520000 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_semesters_per_year_two_semesters_per_year_l3106_310627


namespace NUMINAMATH_CALUDE_roots_sum_powers_l3106_310674

theorem roots_sum_powers (c d : ℝ) : 
  c^2 - 6*c + 10 = 0 → 
  d^2 - 6*d + 10 = 0 → 
  c^3 + c^5 * d^3 + c^3 * d^5 + d^3 = 16036 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l3106_310674


namespace NUMINAMATH_CALUDE_sqrt_two_sufficient_not_necessary_l3106_310694

/-- The line x + y = 0 is tangent to the circle x^2 + (y - a)^2 = 1 -/
def is_tangent (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x + y = 0 ∧ x^2 + (y - a)^2 = 1 ∧
  ∀ (x' y' : ℝ), x' + y' = 0 → x'^2 + (y' - a)^2 ≥ 1

/-- a = √2 is a sufficient but not necessary condition for the line to be tangent to the circle -/
theorem sqrt_two_sufficient_not_necessary :
  (∀ a : ℝ, a = Real.sqrt 2 → is_tangent a) ∧
  ¬(∀ a : ℝ, is_tangent a → a = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_sufficient_not_necessary_l3106_310694


namespace NUMINAMATH_CALUDE_ball_probabilities_l3106_310687

theorem ball_probabilities (total_balls : ℕ) (p_red p_black p_yellow : ℚ) :
  total_balls = 12 →
  p_red + p_black + p_yellow = 1 →
  p_red = 1/3 →
  p_black = p_yellow + 1/6 →
  p_black = 5/12 ∧ p_yellow = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l3106_310687


namespace NUMINAMATH_CALUDE_quadratic_max_min_difference_l3106_310629

/-- Given a quadratic function f(x) = -x^2 + 10x + 9 defined on the interval [2, a/9],
    where a/9 ≥ 8, the difference between its maximum and minimum values is 9. -/
theorem quadratic_max_min_difference (a : ℝ) (h : a / 9 ≥ 8) :
  let f : ℝ → ℝ := λ x ↦ -x^2 + 10*x + 9
  let max_val := (⨆ x ∈ Set.Icc 2 (a / 9), f x)
  let min_val := (⨅ x ∈ Set.Icc 2 (a / 9), f x)
  max_val - min_val = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_min_difference_l3106_310629


namespace NUMINAMATH_CALUDE_truck_toll_calculation_l3106_310630

/-- Calculates the number of axles for a truck given the total number of wheels,
    number of wheels on the front axle, and number of wheels on other axles -/
def calculateAxles (totalWheels : Nat) (frontAxleWheels : Nat) (otherAxleWheels : Nat) : Nat :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

/-- Calculates the toll for a truck given the number of axles -/
def calculateToll (axles : Nat) : Real :=
  1.50 + 1.50 * (axles - 2)

theorem truck_toll_calculation :
  let totalWheels : Nat := 18
  let frontAxleWheels : Nat := 2
  let otherAxleWheels : Nat := 4
  let axles := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  calculateToll axles = 6.00 := by
  sorry

end NUMINAMATH_CALUDE_truck_toll_calculation_l3106_310630


namespace NUMINAMATH_CALUDE_stating_remaining_slices_is_four_l3106_310638

/-- Represents the number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- Represents the number of slices in an extra-large pizza -/
def extra_large_pizza_slices : ℕ := 12

/-- Represents the number of slices Mary eats from the large pizza -/
def mary_large_slices : ℕ := 7

/-- Represents the number of slices Mary eats from the extra-large pizza -/
def mary_extra_large_slices : ℕ := 3

/-- Represents the number of slices John eats from the large pizza -/
def john_large_slices : ℕ := 2

/-- Represents the number of slices John eats from the extra-large pizza -/
def john_extra_large_slices : ℕ := 5

/-- 
Theorem stating that the total number of remaining slices is 4,
given the conditions of the problem.
-/
theorem remaining_slices_is_four :
  (large_pizza_slices - min mary_large_slices large_pizza_slices - min john_large_slices (large_pizza_slices - min mary_large_slices large_pizza_slices)) +
  (extra_large_pizza_slices - mary_extra_large_slices - john_extra_large_slices) = 4 := by
  sorry

end NUMINAMATH_CALUDE_stating_remaining_slices_is_four_l3106_310638


namespace NUMINAMATH_CALUDE_trapezoid_segment_property_l3106_310604

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midpoint_segment : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 150 = longer_base
  midpoint_area_ratio : (shorter_base + midpoint_segment) / (longer_base + midpoint_segment) = 3 / 4
  equal_area_condition : (shorter_base + equal_area_segment) * (height / 2) = 
                         (shorter_base + longer_base) * height / 2

/-- The theorem statement -/
theorem trapezoid_segment_property (t : Trapezoid) : 
  ⌊t.equal_area_segment^2 / 150⌋ = 304 := by sorry

end NUMINAMATH_CALUDE_trapezoid_segment_property_l3106_310604


namespace NUMINAMATH_CALUDE_gwi_seed_count_l3106_310690

/-- The number of watermelon seeds Bom has -/
def bom_seeds : ℕ := 300

/-- The total number of watermelon seeds they have together -/
def total_seeds : ℕ := 1660

/-- The number of watermelon seeds Gwi has -/
def gwi_seeds : ℕ := 340

/-- The number of watermelon seeds Yeon has -/
def yeon_seeds : ℕ := 3 * gwi_seeds

theorem gwi_seed_count :
  bom_seeds < gwi_seeds ∧
  yeon_seeds = 3 * gwi_seeds ∧
  bom_seeds + gwi_seeds + yeon_seeds = total_seeds :=
by sorry

end NUMINAMATH_CALUDE_gwi_seed_count_l3106_310690


namespace NUMINAMATH_CALUDE_potato_bundle_size_l3106_310662

theorem potato_bundle_size (total_potatoes : ℕ) (potato_bundle_price : ℚ)
  (total_carrots : ℕ) (carrots_per_bundle : ℕ) (carrot_bundle_price : ℚ)
  (total_revenue : ℚ) :
  total_potatoes = 250 →
  potato_bundle_price = 19/10 →
  total_carrots = 320 →
  carrots_per_bundle = 20 →
  carrot_bundle_price = 2 →
  total_revenue = 51 →
  ∃ (potatoes_per_bundle : ℕ),
    potatoes_per_bundle = 25 ∧
    (potato_bundle_price * (total_potatoes / potatoes_per_bundle : ℚ) +
     carrot_bundle_price * (total_carrots / carrots_per_bundle : ℚ) = total_revenue) :=
by sorry

end NUMINAMATH_CALUDE_potato_bundle_size_l3106_310662


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3106_310626

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (160 * π / 180) ^ 2) = Real.cos (20 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3106_310626


namespace NUMINAMATH_CALUDE_traveler_water_consumption_l3106_310688

/-- The amount of water drunk by the traveler and camel -/
theorem traveler_water_consumption (traveler_ounces : ℝ) : 
  traveler_ounces > 0 →  -- Assume the traveler drinks a positive amount
  (∃ (camel_ounces : ℝ), 
    camel_ounces = 7 * traveler_ounces ∧  -- Camel drinks 7 times as much
    128 * 2 = traveler_ounces + camel_ounces) →  -- Total consumption is 2 gallons
  traveler_ounces = 32 := by
sorry

end NUMINAMATH_CALUDE_traveler_water_consumption_l3106_310688


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l3106_310686

theorem complex_modulus_sqrt_5 (a b : ℝ) (z : ℂ) : 
  (a + Complex.I)^2 = b * Complex.I → z = a + b * Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l3106_310686


namespace NUMINAMATH_CALUDE_not_adjacent_probability_l3106_310625

theorem not_adjacent_probability (n : ℕ) (h : n = 10) : 
  (n.choose 2 - (n - 1)) / n.choose 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_not_adjacent_probability_l3106_310625


namespace NUMINAMATH_CALUDE_number_of_bowls_l3106_310645

theorem number_of_bowls (n : ℕ) 
  (h1 : n > 0)  -- There is at least one bowl
  (h2 : 12 ≤ n)  -- There are at least 12 bowls to add grapes to
  (h3 : (96 : ℝ) / n = 6)  -- The average increase is 6
  : n = 16 := by
sorry

end NUMINAMATH_CALUDE_number_of_bowls_l3106_310645


namespace NUMINAMATH_CALUDE_parallel_tangents_intersection_l3106_310697

theorem parallel_tangents_intersection (x₀ : ℝ) : 
  (2 * x₀ = -3 * x₀^2) ↔ (x₀ = 0 ∨ x₀ = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_parallel_tangents_intersection_l3106_310697


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l3106_310609

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than_60 (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < 60 → ¬(p ∣ n)

theorem smallest_number_with_conditions : 
  (∀ n : ℕ, n < 4087 → 
    is_prime n ∨ 
    is_square n ∨ 
    ¬(has_no_prime_factor_less_than_60 n)) ∧ 
  ¬(is_prime 4087) ∧ 
  ¬(is_square 4087) ∧ 
  has_no_prime_factor_less_than_60 4087 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l3106_310609


namespace NUMINAMATH_CALUDE_triple_overlap_area_is_six_l3106_310695

/-- Represents a rectangular carpet with width and length -/
structure Carpet where
  width : ℝ
  length : ℝ

/-- Represents the auditorium floor -/
structure Auditorium where
  width : ℝ
  length : ℝ

/-- Calculates the area of triple overlap given three carpets and an auditorium -/
def tripleOverlapArea (c1 c2 c3 : Carpet) (a : Auditorium) : ℝ :=
  sorry

/-- Theorem stating that the area of triple overlap is 6 square meters -/
theorem triple_overlap_area_is_six 
  (c1 : Carpet) 
  (c2 : Carpet) 
  (c3 : Carpet) 
  (a : Auditorium) 
  (h1 : c1.width = 6 ∧ c1.length = 8)
  (h2 : c2.width = 6 ∧ c2.length = 6)
  (h3 : c3.width = 5 ∧ c3.length = 7)
  (h4 : a.width = 10 ∧ a.length = 10) :
  tripleOverlapArea c1 c2 c3 a = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_overlap_area_is_six_l3106_310695


namespace NUMINAMATH_CALUDE_secant_triangle_area_l3106_310655

theorem secant_triangle_area (r : ℝ) (d : ℝ) (θ : ℝ) (S_ABC : ℝ) :
  r = 3 →
  d = 5 →
  θ = 30 * π / 180 →
  S_ABC = 10 →
  ∃ (S_AKL : ℝ), S_AKL = 8 / 5 :=
by sorry

end NUMINAMATH_CALUDE_secant_triangle_area_l3106_310655


namespace NUMINAMATH_CALUDE_min_rotation_angle_is_72_l3106_310673

/-- A regular five-pointed star -/
structure RegularFivePointedStar where
  -- Add necessary properties here

/-- The minimum rotation angle for a regular five-pointed star to coincide with its original position -/
def min_rotation_angle (star : RegularFivePointedStar) : ℝ :=
  72

/-- Theorem stating that the minimum rotation angle for a regular five-pointed star 
    to coincide with its original position is 72 degrees -/
theorem min_rotation_angle_is_72 (star : RegularFivePointedStar) :
  min_rotation_angle star = 72 := by
  sorry

end NUMINAMATH_CALUDE_min_rotation_angle_is_72_l3106_310673


namespace NUMINAMATH_CALUDE_min_area_rectangle_l3106_310623

/-- Given a rectangle with integer length and width, and a perimeter of 120 units,
    the minimum possible area is 59 square units. -/
theorem min_area_rectangle (l w : ℕ) : 
  (2 * l + 2 * w = 120) → (l * w ≥ 59) := by
  sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l3106_310623


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_is_13_l3106_310647

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- Represents the minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFourSameColor (machine : GumballMachine) : Nat :=
  13

/-- Theorem stating that for the given gumball machine configuration,
    the minimum number of gumballs needed to guarantee four of the same color is 13 -/
theorem min_gumballs_for_four_same_color_is_13 (machine : GumballMachine)
    (h1 : machine.red = 10)
    (h2 : machine.white = 8)
    (h3 : machine.blue = 9)
    (h4 : machine.green = 6) :
    minGumballsForFourSameColor machine = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_is_13_l3106_310647


namespace NUMINAMATH_CALUDE_min_value_of_roots_l3106_310692

theorem min_value_of_roots (a x y : ℝ) : 
  x^2 - 2*a*x + a + 6 = 0 →
  y^2 - 2*a*y + a + 6 = 0 →
  x ≠ y →
  ∃ (z : ℝ), ∀ (b : ℝ), (z^2 - 2*b*z + b + 6 = 0) → (x - 1)^2 + (y - 1)^2 ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_roots_l3106_310692


namespace NUMINAMATH_CALUDE_min_values_proof_l3106_310607

theorem min_values_proof (a b m x : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) (hx : x > 2) :
  (a + b ≥ 2 * Real.sqrt (a * b)) ∧
  (a + b = 2 * Real.sqrt (a * b) ↔ a = b) →
  ((m + 1 / m ≥ 2) ∧ (∃ m₀ > 0, m₀ + 1 / m₀ = 2)) ∧
  ((x^2 + x - 5) / (x - 2) ≥ 7 ∧ (∃ x₀ > 2, (x₀^2 + x₀ - 5) / (x₀ - 2) = 7)) := by
  sorry

end NUMINAMATH_CALUDE_min_values_proof_l3106_310607


namespace NUMINAMATH_CALUDE_a_plus_b_values_l3106_310624

/-- A strictly increasing sequence of positive integers -/
def StrictlyIncreasingPositiveSeq (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

/-- The theorem statement -/
theorem a_plus_b_values
  (a b : ℕ → ℕ)
  (h_a_incr : StrictlyIncreasingPositiveSeq a)
  (h_b_incr : StrictlyIncreasingPositiveSeq b)
  (h_eq : a 10 = b 10)
  (h_lt_2017 : a 10 < 2017)
  (h_a_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h_b_rec : ∀ n : ℕ, b (n + 1) = 2 * b n) :
  (a 1 + b 1 = 13) ∨ (a 1 + b 1 = 20) :=
sorry

end NUMINAMATH_CALUDE_a_plus_b_values_l3106_310624


namespace NUMINAMATH_CALUDE_complex_magnitude_l3106_310613

theorem complex_magnitude (z : ℂ) (h : Complex.I - z = 1 + 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3106_310613


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3106_310666

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define set A
def A : Set ℂ := {i, i^2, i^3, i^4}

-- Define set B
def B : Set ℂ := {1, -1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {1, -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3106_310666


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l3106_310696

theorem sufficient_condition_range (m : ℝ) : m > 0 →
  (∀ x : ℝ, x^2 - 8*x - 20 ≤ 0 → (1 - m ≤ x ∧ x ≤ 1 + m)) ∧ 
  (∃ x : ℝ, 1 - m ≤ x ∧ x ≤ 1 + m ∧ x^2 - 8*x - 20 > 0) ↔ 
  m ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l3106_310696


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l3106_310682

/-- Reflects a point over the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- Calculates the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let x1 := p1.1
  let y1 := p1.2
  let x2 := p2.1
  let y2 := p2.2
  let x3 := p3.1
  let y3 := p3.2
  0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC : 
  let A : ℝ × ℝ := (5, 3)
  let B : ℝ × ℝ := reflect_y_axis A
  let C : ℝ × ℝ := reflect_y_eq_x B
  triangle_area A B C = 40 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l3106_310682


namespace NUMINAMATH_CALUDE_spinner_prob_C_or_D_l3106_310677

/-- Represents a circular spinner with four parts -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)

/-- The probability of landing on either C or D -/
def probCorD (s : Spinner) : ℚ := s.probC + s.probD

theorem spinner_prob_C_or_D (s : Spinner) 
  (h1 : s.probA = 1/4)
  (h2 : s.probB = 1/3)
  (h3 : s.probA + s.probB + s.probC + s.probD = 1) :
  probCorD s = 5/12 := by
    sorry

end NUMINAMATH_CALUDE_spinner_prob_C_or_D_l3106_310677


namespace NUMINAMATH_CALUDE_lcm_of_8_and_15_l3106_310643

theorem lcm_of_8_and_15 : Nat.lcm 8 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_8_and_15_l3106_310643


namespace NUMINAMATH_CALUDE_car_speeds_satisfy_conditions_l3106_310614

/-- Represents the scenario of two cars meeting on a road --/
structure CarMeetingScenario where
  distance : ℝ
  speed1 : ℝ
  speed2 : ℝ
  speed_increase1 : ℝ
  speed_increase2 : ℝ
  time_difference : ℝ

/-- Checks if the given car speeds satisfy the meeting conditions --/
def satisfies_conditions (s : CarMeetingScenario) : Prop :=
  s.distance / (s.speed1 - s.speed2) - s.distance / ((s.speed1 + s.speed_increase1) - (s.speed2 + s.speed_increase2)) = s.time_difference

/-- The theorem stating that the given speeds satisfy the conditions --/
theorem car_speeds_satisfy_conditions : ∃ (s : CarMeetingScenario),
  s.distance = 60 ∧
  s.speed_increase1 = 10 ∧
  s.speed_increase2 = 8 ∧
  s.time_difference = 1 ∧
  s.speed1 = 50 ∧
  s.speed2 = 40 ∧
  satisfies_conditions s := by
  sorry


end NUMINAMATH_CALUDE_car_speeds_satisfy_conditions_l3106_310614


namespace NUMINAMATH_CALUDE_water_tank_capacity_l3106_310640

theorem water_tank_capacity (c : ℝ) (h1 : c > 0) : 
  (c / 3 : ℝ) / c = 1 / 3 ∧ 
  (c / 3 + 5 : ℝ) / c = 1 / 2 → 
  c = 30 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l3106_310640


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3106_310667

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℤ),
  X^4 + X^2 = (X^2 + 3*X + 2) * q + r ∧ 
  r.degree < (X^2 + 3*X + 2).degree ∧ 
  r = -18*X - 16 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3106_310667


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3106_310658

theorem complex_equation_solution : 
  ∃ z : ℂ, z * (1 - Complex.I) = 2 * Complex.I ∧ z = 1 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3106_310658


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l3106_310605

theorem dormitory_to_city_distance : ∃ (d : ℝ), 
  (1/5 : ℝ) * d + (2/3 : ℝ) * d + 14 = d ∧ d = 105 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l3106_310605


namespace NUMINAMATH_CALUDE_points_A_B_D_collinear_l3106_310612

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def vector_AB (a b : V) : V := a + 2 • b
def vector_BC (a b : V) : V := -5 • a + 6 • b
def vector_CD (a b : V) : V := 7 • a - 2 • b

theorem points_A_B_D_collinear (a b : V) :
  ∃ (k : ℝ), vector_AB a b = k • (vector_AB a b + vector_BC a b + vector_CD a b) :=
sorry

end NUMINAMATH_CALUDE_points_A_B_D_collinear_l3106_310612


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3106_310670

theorem simplify_and_evaluate : 
  let f (x : ℝ) := (2*x + 4) / (x^2 - 6*x + 9) / ((2*x - 1) / (x - 3) - 1)
  f 0 = -2/3 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3106_310670


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3106_310619

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 10 players where each player plays every other player once,
    the total number of games played is 45. --/
theorem chess_tournament_games :
  num_games 10 = 45 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3106_310619


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3106_310691

theorem cube_volume_problem (cube_A cube_B : Real → Real → Real → Real) :
  (∀ x y z, cube_A x y z = 8) →
  (∀ x y z, (6 * (cube_B x y z)^(2/3)) = 3 * (6 * 2^2)) →
  (∀ x y z, cube_B x y z = 24 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3106_310691


namespace NUMINAMATH_CALUDE_matrix_power_2023_l3106_310649

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l3106_310649


namespace NUMINAMATH_CALUDE_exam_score_distribution_l3106_310608

/-- Represents the normal distribution of exam scores -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ

/-- Represents the class and exam information -/
structure ExamInfo where
  totalStudents : ℕ
  scoreDistribution : NormalDistribution
  middleProbability : ℝ

/-- Calculates the number of students with scores above a given threshold -/
def studentsAboveThreshold (info : ExamInfo) (threshold : ℝ) : ℕ :=
  sorry

theorem exam_score_distribution (info : ExamInfo) :
  info.totalStudents = 50 ∧
  info.scoreDistribution = { μ := 110, σ := 10 } ∧
  info.middleProbability = 0.34 →
  studentsAboveThreshold info 120 = 8 :=
sorry

end NUMINAMATH_CALUDE_exam_score_distribution_l3106_310608


namespace NUMINAMATH_CALUDE_smallest_N_proof_l3106_310621

def f (n : ℕ+) : ℕ := sorry

def g (n : ℕ+) : ℕ := sorry

def N : ℕ+ := sorry

theorem smallest_N_proof : N = 44 ∧ (∀ m : ℕ+, m < N → g m < 11) ∧ g N ≥ 11 := by sorry

end NUMINAMATH_CALUDE_smallest_N_proof_l3106_310621


namespace NUMINAMATH_CALUDE_soup_tasting_equivalent_to_sample_estimation_l3106_310664

/-- Represents the entire soup -/
def Soup : Type := Unit

/-- Represents a small portion of the soup -/
def SoupSample : Type := Unit

/-- Represents the action of tasting a small portion of soup -/
def TasteSoup : SoupSample → Bool := fun _ => true

/-- Represents a population in a statistical survey -/
def Population : Type := Unit

/-- Represents a sample from a population -/
def PopulationSample : Type := Unit

/-- Represents the process of sample estimation in statistics -/
def SampleEstimation : PopulationSample → Population → Prop := fun _ _ => true

/-- Theorem stating that tasting a small portion of soup is mathematically equivalent
    to using sample estimation in statistical surveys -/
theorem soup_tasting_equivalent_to_sample_estimation :
  ∀ (soup : Soup) (sample : SoupSample) (pop : Population) (pop_sample : PopulationSample),
  TasteSoup sample ↔ SampleEstimation pop_sample pop :=
sorry

end NUMINAMATH_CALUDE_soup_tasting_equivalent_to_sample_estimation_l3106_310664


namespace NUMINAMATH_CALUDE_exam_failure_percentage_l3106_310668

theorem exam_failure_percentage 
  (pass_english : ℝ) 
  (pass_math : ℝ) 
  (pass_either : ℝ) 
  (h1 : pass_english = 0.63) 
  (h2 : pass_math = 0.65) 
  (h3 : pass_either = 0.55) : 
  1 - pass_either = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_exam_failure_percentage_l3106_310668


namespace NUMINAMATH_CALUDE_min_slope_tangent_line_l3106_310635

variable (x y : ℝ)

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 3

-- Theorem statement
theorem min_slope_tangent_line :
  ∃ (x₀ : ℝ), 
    (∀ x, f' x ≥ f' x₀) ∧ 
    (3*x - y + 1 = 0 ↔ y = f' x₀ * (x - x₀) + f x₀) :=
sorry

end NUMINAMATH_CALUDE_min_slope_tangent_line_l3106_310635


namespace NUMINAMATH_CALUDE_factorial_equation_sum_l3106_310660

theorem factorial_equation_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ k l : ℕ, n.factorial / 2 = k.factorial + l.factorial) ∧
  (∀ n : ℕ, n ∉ S → ¬∃ k l : ℕ, n.factorial / 2 = k.factorial + l.factorial) ∧
  S.sum id = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_sum_l3106_310660


namespace NUMINAMATH_CALUDE_same_color_prob_six_green_seven_white_l3106_310661

/-- The probability of drawing two balls of the same color from a bag containing 
    6 green balls and 7 white balls. -/
def same_color_probability (green : ℕ) (white : ℕ) : ℚ :=
  let total := green + white
  let p_green := (green / total) * ((green - 1) / (total - 1))
  let p_white := (white / total) * ((white - 1) / (total - 1))
  p_green + p_white

/-- Theorem stating that the probability of drawing two balls of the same color 
    from a bag with 6 green and 7 white balls is 6/13. -/
theorem same_color_prob_six_green_seven_white : 
  same_color_probability 6 7 = 6 / 13 := by
  sorry

end NUMINAMATH_CALUDE_same_color_prob_six_green_seven_white_l3106_310661


namespace NUMINAMATH_CALUDE_pure_imaginary_value_l3106_310616

def complex_number (a : ℝ) : ℂ := (a^2 + 2*a - 3 : ℝ) + (a^2 - 4*a + 3 : ℝ) * Complex.I

theorem pure_imaginary_value (a : ℝ) :
  (complex_number a).re = 0 ∧ (complex_number a).im ≠ 0 → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_value_l3106_310616


namespace NUMINAMATH_CALUDE_pauls_money_duration_l3106_310654

/-- Represents the duration (in weeks) that money lasts given earnings and weekly spending. -/
def money_duration (lawn_earnings weed_eating_earnings weekly_spending : ℚ) : ℚ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Paul's money lasts for 2 weeks given his earnings and spending. -/
theorem pauls_money_duration :
  money_duration 3 3 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pauls_money_duration_l3106_310654
