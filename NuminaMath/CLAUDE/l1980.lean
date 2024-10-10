import Mathlib

namespace milk_dilution_l1980_198014

theorem milk_dilution (whole_milk : ℝ) (added_skimmed_milk : ℝ) 
  (h1 : whole_milk = 1) 
  (h2 : added_skimmed_milk = 1/4) : 
  let initial_cream := 0.05 * whole_milk
  let initial_skimmed := 0.95 * whole_milk
  let total_volume := whole_milk + added_skimmed_milk
  let final_cream_percentage := initial_cream / total_volume
  final_cream_percentage = 0.04 := by
  sorry

end milk_dilution_l1980_198014


namespace parallelogram_intersection_theorem_l1980_198054

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (J K L M : Point)

/-- Checks if a point is on the extension of a line segment -/
def isOnExtension (A B P : Point) : Prop := sorry

/-- Checks if two line segments intersect at a point -/
def intersectsAt (A B C D Q : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (A B : Point) : ℝ := sorry

theorem parallelogram_intersection_theorem (JKLM : Parallelogram) (P Q R : Point) :
  isOnExtension JKLM.L JKLM.M P →
  intersectsAt JKLM.K P JKLM.L JKLM.J Q →
  intersectsAt JKLM.K P JKLM.J JKLM.M R →
  distance Q R = 40 →
  distance R P = 30 →
  distance JKLM.K Q = 20 := by
  sorry

end parallelogram_intersection_theorem_l1980_198054


namespace specific_gold_cube_profit_l1980_198086

/-- Calculates the profit from selling a gold cube -/
def gold_cube_profit (side_length : ℝ) (density : ℝ) (purchase_price : ℝ) (markup : ℝ) : ℝ :=
  let volume := side_length ^ 3
  let mass := density * volume
  let cost := mass * purchase_price
  let selling_price := cost * markup
  selling_price - cost

/-- Theorem stating the profit for a specific gold cube -/
theorem specific_gold_cube_profit :
  gold_cube_profit 6 19 60 1.5 = 123120 := by
  sorry

end specific_gold_cube_profit_l1980_198086


namespace house_area_l1980_198072

theorem house_area (living_dining_kitchen_area master_bedroom_area : ℝ)
  (h1 : living_dining_kitchen_area = 1000)
  (h2 : master_bedroom_area = 1040)
  (guest_bedroom_area : ℝ)
  (h3 : guest_bedroom_area = (1 / 4) * master_bedroom_area) :
  living_dining_kitchen_area + master_bedroom_area + guest_bedroom_area = 2300 :=
by sorry

end house_area_l1980_198072


namespace problem_solution_l1980_198006

-- Define the function f
def f (x : ℝ) : ℝ := |2 * |x| - 1|

-- Define the solution set A
def A : Set ℝ := {x | f x ≤ 1}

-- Theorem statement
theorem problem_solution :
  (A = {x : ℝ | -1 ≤ x ∧ x ≤ 1}) ∧
  (∀ m n : ℝ, m ∈ A → n ∈ A → |m + n| ≤ m * n + 1) := by
  sorry

end problem_solution_l1980_198006


namespace f_2_nonneg_necessary_not_sufficient_l1980_198003

/-- A quadratic function f(x) = ax^2 + bx -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

/-- f(x) is monotonically increasing on (1, +∞) -/
def monotonically_increasing_on_interval (a b : ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f a b x < f a b y

/-- f(2) ≥ 0 is a necessary but not sufficient condition for
    f(x) to be monotonically increasing on (1, +∞) -/
theorem f_2_nonneg_necessary_not_sufficient (a b : ℝ) :
  (∀ a b, monotonically_increasing_on_interval a b → f a b 2 ≥ 0) ∧
  ¬(∀ a b, f a b 2 ≥ 0 → monotonically_increasing_on_interval a b) :=
by sorry

end f_2_nonneg_necessary_not_sufficient_l1980_198003


namespace student_marks_calculation_l1980_198017

theorem student_marks_calculation 
  (max_marks : ℕ) 
  (passing_percentage : ℚ) 
  (fail_margin : ℕ) 
  (h1 : max_marks = 400)
  (h2 : passing_percentage = 36 / 100)
  (h3 : fail_margin = 14) :
  ∃ (student_marks : ℕ), 
    student_marks = max_marks * passing_percentage - fail_margin ∧
    student_marks = 130 :=
by sorry

end student_marks_calculation_l1980_198017


namespace cookout_2006_l1980_198061

/-- The number of kids at the cookout in 2004 -/
def kids_2004 : ℕ := 60

/-- The number of kids at the cookout in 2005 -/
def kids_2005 : ℕ := kids_2004 / 2

/-- The number of kids at the cookout in 2006 -/
def kids_2006 : ℕ := kids_2005 * 2 / 3

/-- Theorem stating that the number of kids at the cookout in 2006 is 20 -/
theorem cookout_2006 : kids_2006 = 20 := by
  sorry

end cookout_2006_l1980_198061


namespace calculation_proof_inequality_system_solution_l1980_198055

-- Problem 1
theorem calculation_proof (a b : ℝ) (h : a ≠ b ∧ a ≠ -b ∧ a ≠ 0) :
  (a - b) / (a + b) - (a^2 - 2*a*b + b^2) / (a^2 - b^2) / ((a - b) / a) = -b / (a + b) := by
  sorry

-- Problem 2
theorem inequality_system_solution (x : ℝ) :
  (x - 3*(x - 2) ≥ 4 ∧ (2*x - 1) / 5 > (x + 1) / 2) ↔ x < -7 := by
  sorry

end calculation_proof_inequality_system_solution_l1980_198055


namespace parallel_vectors_x_value_l1980_198026

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (4, -2)
def b : ℝ → ℝ × ℝ := λ x ↦ (x, 5)

/-- Parallel vectors in ℝ² -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  parallel a (b x) → x = -10 := by sorry

end parallel_vectors_x_value_l1980_198026


namespace car_speed_proof_l1980_198080

/-- The speed of the first car in miles per hour -/
def speed1 : ℝ := 52

/-- The time traveled in hours -/
def time : ℝ := 3.5

/-- The total distance between the cars after the given time in miles -/
def total_distance : ℝ := 385

/-- The speed of the second car in miles per hour -/
def speed2 : ℝ := 58

theorem car_speed_proof :
  speed1 * time + speed2 * time = total_distance :=
by sorry

end car_speed_proof_l1980_198080


namespace possible_student_counts_l1980_198059

def is_valid_student_count (n : ℕ) : Prop :=
  n > 1 ∧ (120 - 2) % (n - 1) = 0

theorem possible_student_counts :
  ∀ n : ℕ, is_valid_student_count n ↔ n = 2 ∨ n = 3 ∨ n = 60 ∨ n = 119 := by
  sorry

end possible_student_counts_l1980_198059


namespace towel_shrinkage_l1980_198011

theorem towel_shrinkage (original_length original_breadth : ℝ) 
  (h_positive : original_length > 0 ∧ original_breadth > 0) :
  let new_length := 0.7 * original_length
  let new_area := 0.525 * (original_length * original_breadth)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.75 * original_breadth ∧
    new_area = new_length * new_breadth :=
by sorry

end towel_shrinkage_l1980_198011


namespace stating_inscribed_triangle_area_bound_l1980_198030

/-- A parallelogram in a 2D plane. -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ

/-- A triangle in a 2D plane. -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Checks if a point is inside or on the perimeter of a parallelogram. -/
def isInOrOnParallelogram (p : ℝ × ℝ) (pgram : Parallelogram) : Prop :=
  sorry

/-- Checks if a triangle is inscribed in a parallelogram. -/
def isInscribed (t : Triangle) (pgram : Parallelogram) : Prop :=
  ∀ i, isInOrOnParallelogram (t.vertices i) pgram

/-- Calculates the area of a parallelogram. -/
noncomputable def areaParallelogram (pgram : Parallelogram) : ℝ :=
  sorry

/-- Calculates the area of a triangle. -/
noncomputable def areaTriangle (t : Triangle) : ℝ :=
  sorry

/-- 
Theorem stating that the area of any triangle inscribed in a parallelogram
is less than or equal to half the area of the parallelogram.
-/
theorem inscribed_triangle_area_bound
  (pgram : Parallelogram) (t : Triangle) (h : isInscribed t pgram) :
  areaTriangle t ≤ (1/2) * areaParallelogram pgram :=
by sorry

end stating_inscribed_triangle_area_bound_l1980_198030


namespace primitive_pythagorean_triple_ab_div_12_l1980_198010

/-- A primitive Pythagorean triple is a tuple of positive integers (a, b, c) where a² + b² = c² and gcd(a, b, c) = 1 -/
def isPrimitivePythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧ Nat.gcd a (Nat.gcd b c) = 1

/-- For any primitive Pythagorean triple (a, b, c), ab is divisible by 12 -/
theorem primitive_pythagorean_triple_ab_div_12 (a b c : ℕ) 
  (h : isPrimitivePythagoreanTriple a b c) : 
  12 ∣ (a * b) := by
  sorry


end primitive_pythagorean_triple_ab_div_12_l1980_198010


namespace greatest_sum_consecutive_odd_integers_product_less_500_l1980_198013

theorem greatest_sum_consecutive_odd_integers_product_less_500 : 
  (∃ (n : ℤ), 
    Odd n ∧ 
    n * (n + 2) < 500 ∧ 
    n + (n + 2) = 44 ∧ 
    (∀ (m : ℤ), Odd m → m * (m + 2) < 500 → m + (m + 2) ≤ 44)) :=
by sorry

end greatest_sum_consecutive_odd_integers_product_less_500_l1980_198013


namespace betty_oranges_l1980_198044

/-- Given 3 boxes and 8 oranges per box, the total number of oranges is 24. -/
theorem betty_oranges (num_boxes : ℕ) (oranges_per_box : ℕ) 
  (h1 : num_boxes = 3) 
  (h2 : oranges_per_box = 8) : 
  num_boxes * oranges_per_box = 24 := by
  sorry

end betty_oranges_l1980_198044


namespace music_listening_time_l1980_198064

/-- Given a music tempo and total beats heard per week, calculate the hours of music listened to per day. -/
theorem music_listening_time (tempo : ℕ) (total_beats_per_week : ℕ) : 
  tempo = 200 → total_beats_per_week = 168000 → 
  (total_beats_per_week / 7 / tempo * 60) / 60 = 2 := by
  sorry

#check music_listening_time

end music_listening_time_l1980_198064


namespace student_percentage_theorem_l1980_198039

theorem student_percentage_theorem (total : ℝ) (third_year_percent : ℝ) (second_year_fraction : ℝ)
  (h1 : third_year_percent = 50)
  (h2 : second_year_fraction = 2/3)
  (h3 : total > 0) :
  let non_third_year := total - (third_year_percent / 100) * total
  let second_year := second_year_fraction * non_third_year
  (total - second_year) / total * 100 = 66.66666666666667 :=
sorry

end student_percentage_theorem_l1980_198039


namespace negative_number_identification_l1980_198075

theorem negative_number_identification :
  let a := -3^2
  let b := (-3)^2
  let c := |-3|
  let d := -(-3)
  (a < 0) ∧ (b ≥ 0) ∧ (c ≥ 0) ∧ (d ≥ 0) := by
  sorry

end negative_number_identification_l1980_198075


namespace new_speed_calculation_l1980_198037

/-- Theorem: Given a distance of 630 km and an original time of 6 hours,
    if the new time is 3/2 times the original time,
    then the new speed required to cover the same distance is 70 km/h. -/
theorem new_speed_calculation (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 630 →
  original_time = 6 →
  new_time_factor = 3 / 2 →
  let new_time := original_time * new_time_factor
  let new_speed := distance / new_time
  new_speed = 70 := by
  sorry

#check new_speed_calculation

end new_speed_calculation_l1980_198037


namespace pen_cost_ratio_l1980_198025

theorem pen_cost_ratio (blue_pens : ℕ) (red_pens : ℕ) (blue_cost : ℚ) (total_cost : ℚ) : 
  blue_pens = 10 →
  red_pens = 15 →
  blue_cost = 1/10 →
  total_cost = 4 →
  (total_cost - blue_pens * blue_cost) / red_pens / blue_cost = 2 := by
sorry

end pen_cost_ratio_l1980_198025


namespace complex_sum_and_reciprocal_l1980_198035

theorem complex_sum_and_reciprocal : 
  let z : ℂ := 1 - I
  (z⁻¹ + z) = (3/2 : ℂ) - (1/2 : ℂ) * I := by sorry

end complex_sum_and_reciprocal_l1980_198035


namespace shaded_area_rectangle_l1980_198092

theorem shaded_area_rectangle (total_width total_height : ℝ)
  (small_rect_width small_rect_height : ℝ)
  (triangle1_base triangle1_height : ℝ)
  (triangle2_base triangle2_height : ℝ) :
  total_width = 8 ∧ total_height = 5 ∧
  small_rect_width = 4 ∧ small_rect_height = 2 ∧
  triangle1_base = 5 ∧ triangle1_height = 2 ∧
  triangle2_base = 3 ∧ triangle2_height = 2 →
  total_width * total_height -
  (2 * small_rect_width * small_rect_height +
   2 * (1/2 * triangle1_base * triangle1_height) +
   2 * (1/2 * triangle2_base * triangle2_height)) = 6.5 := by
sorry

end shaded_area_rectangle_l1980_198092


namespace green_or_yellow_marble_probability_l1980_198058

/-- The probability of drawing a green or yellow marble from a bag -/
theorem green_or_yellow_marble_probability
  (green : ℕ) (yellow : ℕ) (red : ℕ) (blue : ℕ)
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_red : red = 4)
  (h_blue : blue = 2) :
  (green + yellow : ℚ) / (green + yellow + red + blue) = 7 / 13 :=
by sorry

end green_or_yellow_marble_probability_l1980_198058


namespace eight_b_equals_sixteen_l1980_198060

theorem eight_b_equals_sixteen
  (h1 : 6 * a + 3 * b = 0)
  (h2 : b - 3 = a)
  (h3 : b + c = 5)
  : 8 * b = 16 := by
  sorry

end eight_b_equals_sixteen_l1980_198060


namespace work_increase_percentage_l1980_198095

/-- Proves that when 1/5 of the members in an office are absent, 
    the percentage increase in work for each remaining person is 25% -/
theorem work_increase_percentage (p : ℝ) (W : ℝ) (h1 : p > 0) (h2 : W > 0) : 
  let original_work_per_person := W / p
  let remaining_persons := p * (4/5)
  let new_work_per_person := W / remaining_persons
  let increase_percentage := (new_work_per_person - original_work_per_person) / original_work_per_person * 100
  increase_percentage = 25 := by sorry

end work_increase_percentage_l1980_198095


namespace selection_theorem_l1980_198024

/-- The number of volunteers --/
def n : ℕ := 5

/-- The number of days of service --/
def days : ℕ := 2

/-- The number of people selected each day --/
def selected_per_day : ℕ := 2

/-- The number of ways to select exactly one person to serve for both days --/
def selection_ways : ℕ := n * (n - 1) * (n - 2)

theorem selection_theorem : selection_ways = 60 := by
  sorry

end selection_theorem_l1980_198024


namespace xy_plus_y_squared_l1980_198005

theorem xy_plus_y_squared (x y : ℝ) (h : x * (x + y) = x^2 + y + 12) : 
  x * y + y^2 = y^2 + y + 12 := by
  sorry

end xy_plus_y_squared_l1980_198005


namespace grade_swap_possible_l1980_198078

/-- Represents a grade scaling system -/
structure GradeScale where
  upper_limit : ℕ
  round_up_half : Set ℕ

/-- Represents a grade within a scaling system -/
def Grade := { g : ℚ // 0 < g ∧ g < 1 }

/-- Function to rescale a grade -/
def rescale (g : Grade) (old_scale new_scale : GradeScale) : Grade :=
  sorry

/-- Theorem stating that any two grades can be swapped through a series of rescalings -/
theorem grade_swap_possible (a b : Grade) :
  ∃ (scales : List GradeScale), 
    let final_scale := scales.foldl (λ acc s => s) { upper_limit := 100, round_up_half := ∅ }
    let new_a := scales.foldl (λ acc s => rescale acc s final_scale) a
    let new_b := scales.foldl (λ acc s => rescale acc s final_scale) b
    new_a = b ∧ new_b = a :=
  sorry

end grade_swap_possible_l1980_198078


namespace complex_calculation_l1980_198071

theorem complex_calculation : 550 - (104 / (Real.sqrt 20.8)^2)^3 = 425 := by
  sorry

end complex_calculation_l1980_198071


namespace fraction_equality_l1980_198036

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - 4*b ≠ 0) (h4 : 4*a - b ≠ 0)
  (h5 : (4*a + b) / (a - 4*b) = 3) : (a + 4*b) / (4*a - b) = 9/53 := by
  sorry

end fraction_equality_l1980_198036


namespace system_solution_l1980_198062

theorem system_solution (x y z : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  (x^2 + y^2 = -x + 3*y + z ∧
   y^2 + z^2 = x + 3*y - z ∧
   x^2 + z^2 = 2*x + 2*y - z) →
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ 
   (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
by sorry

end system_solution_l1980_198062


namespace octagon_cannot_cover_floor_l1980_198027

/-- Calculate the interior angle of a regular polygon with n sides -/
def interiorAngle (n : ℕ) : ℚ :=
  (n - 2 : ℚ) * 180 / n

/-- Check if a given angle divides 360° evenly -/
def divides360 (angle : ℚ) : Prop :=
  ∃ k : ℕ, k * angle = 360

/-- Theorem: Among equilateral triangles, squares, hexagons, and octagons,
    only the octagon's interior angle does not divide 360° evenly -/
theorem octagon_cannot_cover_floor :
  divides360 (interiorAngle 3) ∧
  divides360 (interiorAngle 4) ∧
  divides360 (interiorAngle 6) ∧
  ¬divides360 (interiorAngle 8) :=
sorry

end octagon_cannot_cover_floor_l1980_198027


namespace water_usage_difference_l1980_198040

/-- Proves the difference in daily water usage before and after installing a water recycling device -/
theorem water_usage_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (b / a) - (b / (a + 4)) = (4 * b) / (a * (a + 4)) := by
  sorry

end water_usage_difference_l1980_198040


namespace pencil_length_l1980_198085

/-- The total length of a pencil with given colored sections -/
theorem pencil_length (purple_length black_length blue_length : ℝ) 
  (h_purple : purple_length = 1.5)
  (h_black : black_length = 0.5)
  (h_blue : blue_length = 2) :
  purple_length + black_length + blue_length = 4 := by
  sorry

end pencil_length_l1980_198085


namespace third_place_prize_l1980_198066

def prize_distribution (total_people : ℕ) (contribution : ℕ) (first_place_percentage : ℚ) : ℚ :=
  let total_pot : ℚ := (total_people * contribution : ℚ)
  let first_place_prize : ℚ := total_pot * first_place_percentage
  let remaining : ℚ := total_pot - first_place_prize
  remaining / 2

theorem third_place_prize :
  prize_distribution 8 5 (4/5) = 4 := by
  sorry

end third_place_prize_l1980_198066


namespace polynomial_remainder_theorem_l1980_198038

/-- Given a polynomial p(x) satisfying specific conditions, 
    prove that its remainder when divided by (x-1)(x+1)(x-3) is -x^2 + 4x + 2 -/
theorem polynomial_remainder_theorem (p : ℝ → ℝ) 
  (h1 : p 1 = 5) (h2 : p 3 = 7) (h3 : p (-1) = 9) :
  ∃ q : ℝ → ℝ, ∀ x, p x = q x * (x - 1) * (x + 1) * (x - 3) + (-x^2 + 4*x + 2) :=
by sorry

end polynomial_remainder_theorem_l1980_198038


namespace count_special_numbers_l1980_198020

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def leftmost_digit (n : ℕ) : ℕ := n / 1000

def second_digit (n : ℕ) : ℕ := (n / 100) % 10

def third_digit (n : ℕ) : ℕ := (n / 10) % 10

def last_digit (n : ℕ) : ℕ := n % 10

def all_digits_different (n : ℕ) : Prop :=
  leftmost_digit n ≠ second_digit n ∧
  leftmost_digit n ≠ third_digit n ∧
  leftmost_digit n ≠ last_digit n ∧
  second_digit n ≠ third_digit n ∧
  second_digit n ≠ last_digit n ∧
  third_digit n ≠ last_digit n

theorem count_special_numbers :
  ∃ (S : Finset ℕ),
    (∀ n ∈ S,
      is_four_digit n ∧
      leftmost_digit n % 2 = 1 ∧
      leftmost_digit n < 5 ∧
      second_digit n % 2 = 0 ∧
      second_digit n < 6 ∧
      all_digits_different n ∧
      n % 5 = 0) ∧
    S.card = 48 :=
by sorry

end count_special_numbers_l1980_198020


namespace partnership_investment_ratio_l1980_198091

theorem partnership_investment_ratio 
  (x : ℝ) 
  (m : ℝ) 
  (total_gain : ℝ) 
  (a_share : ℝ) 
  (h1 : total_gain = 27000) 
  (h2 : a_share = 9000) 
  (h3 : a_share = (1/3) * total_gain) 
  (h4 : (12*x) / (12*x + 12*x + 4*m*x) = 1/3) : 
  m = 3 := by
sorry

end partnership_investment_ratio_l1980_198091


namespace quadratic_coefficient_sum_l1980_198097

theorem quadratic_coefficient_sum : ∃ (coeff_sum : ℤ),
  (∀ a : ℤ, 
    (∃ r s : ℤ, r < 0 ∧ s < 0 ∧ r ≠ s ∧ r * s = 24 ∧ r + s = a) →
    (∃ x y : ℤ, x^2 + a*x + 24 = 0 ∧ y^2 + a*y + 24 = 0 ∧ x ≠ y ∧ x < 0 ∧ y < 0)) ∧
  (∀ a : ℤ,
    (∃ x y : ℤ, x^2 + a*x + 24 = 0 ∧ y^2 + a*y + 24 = 0 ∧ x ≠ y ∧ x < 0 ∧ y < 0) →
    (∃ r s : ℤ, r < 0 ∧ s < 0 ∧ r ≠ s ∧ r * s = 24 ∧ r + s = a)) ∧
  coeff_sum = -60 :=
by sorry

end quadratic_coefficient_sum_l1980_198097


namespace tiles_difference_7_6_l1980_198083

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := n ^ 2

/-- The theorem stating the difference in tiles between the 7th and 6th squares -/
theorem tiles_difference_7_6 : tiles_in_square 7 - tiles_in_square 6 = 13 := by
  sorry

end tiles_difference_7_6_l1980_198083


namespace g_composition_of_3_l1980_198043

def g (n : ℕ) : ℕ :=
  if n ≤ 5 then n^2 + 2*n + 1 else 2*n + 4

theorem g_composition_of_3 : g (g (g 3)) = 76 := by
  sorry

end g_composition_of_3_l1980_198043


namespace isosceles_triangle_perimeter_l1980_198079

/-- An isosceles triangle with sides 12cm and 24cm has a perimeter of 60cm -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 12 →
  b = 24 →
  c = 24 →
  a + b > c →
  a + c > b →
  b + c > a →
  a + b + c = 60 :=
by sorry

end isosceles_triangle_perimeter_l1980_198079


namespace sand_pile_removal_l1980_198068

theorem sand_pile_removal (initial_weight : ℚ) (first_removal : ℚ) (second_removal : ℚ)
  (h1 : initial_weight = 8 / 3)
  (h2 : first_removal = 1 / 4)
  (h3 : second_removal = 5 / 6) :
  first_removal + second_removal = 13 / 12 := by
sorry

end sand_pile_removal_l1980_198068


namespace article_cost_l1980_198077

/-- The cost of an article given specific selling conditions --/
theorem article_cost : ∃ (C : ℝ), 
  (450 - C = 1.1 * (380 - C)) ∧ 
  (C > 0) ∧ 
  (C = 320) := by
  sorry

end article_cost_l1980_198077


namespace hyperbola_specific_equation_l1980_198021

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The general equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola a b) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The focus of a hyperbola -/
def focus (x y : ℝ) : Prop := x = 2 ∧ y = 0

/-- The asymptotes of a hyperbola -/
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- The theorem stating the specific equation of the hyperbola given the conditions -/
theorem hyperbola_specific_equation (a b : ℝ) (h : Hyperbola a b) 
  (focus_cond : focus 2 0)
  (asymp_cond : ∀ x y, asymptotes x y ↔ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) :
  ∀ x y, hyperbola_equation h x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

end hyperbola_specific_equation_l1980_198021


namespace ellipse_sum_specific_l1980_198007

/-- The sum of the center coordinates and axis lengths of an ellipse -/
def ellipse_sum (h k a b : ℝ) : ℝ := h + k + a + b

/-- Theorem: The sum of center coordinates and axis lengths for a specific ellipse -/
theorem ellipse_sum_specific : ∃ (h k a b : ℝ), 
  (∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ∧ 
  h = 3 ∧ 
  k = -5 ∧ 
  a = 7 ∧ 
  b = 4 ∧ 
  ellipse_sum h k a b = 9 := by
  sorry

end ellipse_sum_specific_l1980_198007


namespace at_least_one_not_less_than_six_l1980_198084

theorem at_least_one_not_less_than_six (a b : ℝ) (h : a + b = 12) : max a b ≥ 6 := by
  sorry

end at_least_one_not_less_than_six_l1980_198084


namespace largest_equal_cost_integer_l1980_198049

/-- Calculates the sum of digits for a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the cost of binary representation -/
def binaryCost (n : ℕ) : ℕ := sorry

/-- Theorem stating that 311 is the largest integer less than 500 with equal costs -/
theorem largest_equal_cost_integer :
  ∀ n : ℕ, n < 500 → n > 311 → sumOfDigits n ≠ binaryCost n ∧
  sumOfDigits 311 = binaryCost 311 := by sorry

end largest_equal_cost_integer_l1980_198049


namespace checker_moves_10_l1980_198012

/-- Represents the number of ways a checker can move n cells -/
def checkerMoves : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => checkerMoves (n + 1) + checkerMoves n

/-- Theorem stating that the number of ways a checker can move 10 cells is 89 -/
theorem checker_moves_10 : checkerMoves 10 = 89 := by
  sorry

#eval checkerMoves 10

end checker_moves_10_l1980_198012


namespace sufficient_not_necessary_l1980_198009

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 1 / a^2 → a^2 > 1 / a) ∧ 
  (∃ a, a^2 > 1 / a ∧ ¬(a > 1 / a^2)) := by
sorry

end sufficient_not_necessary_l1980_198009


namespace line_symmetry_l1980_198093

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Returns true if two lines are symmetric about the y-axis -/
def symmetricAboutYAxis (l1 l2 : Line) : Prop :=
  l1.slope = -l2.slope ∧ l1.intercept = l2.intercept

theorem line_symmetry (l1 l2 : Line) :
  l1.slope = 2 ∧ l1.intercept = 3 →
  symmetricAboutYAxis l1 l2 →
  l2.slope = -2 ∧ l2.intercept = 3 := by
  sorry

end line_symmetry_l1980_198093


namespace football_game_attendance_difference_l1980_198032

theorem football_game_attendance_difference :
  let saturday : ℕ := 80
  let wednesday (monday : ℕ) : ℕ := monday + 50
  let friday (monday : ℕ) : ℕ := saturday + monday
  let total : ℕ := 390
  ∀ monday : ℕ,
    monday < saturday →
    saturday + monday + wednesday monday + friday monday = total →
    saturday - monday = 20 :=
by
  sorry

end football_game_attendance_difference_l1980_198032


namespace point_on_x_axis_l1980_198004

/-- A point P(x, y) lies on the x-axis if and only if its y-coordinate is 0 -/
def lies_on_x_axis (x y : ℝ) : Prop := y = 0

/-- The theorem states that if the point P(a-4, a+3) lies on the x-axis, then a = -3 -/
theorem point_on_x_axis (a : ℝ) :
  lies_on_x_axis (a - 4) (a + 3) → a = -3 := by
  sorry

end point_on_x_axis_l1980_198004


namespace survey_result_l1980_198076

/-- The number of households that used neither brand E nor brand B soap -/
def neither : ℕ := 80

/-- The number of households that used only brand E soap -/
def only_E : ℕ := 60

/-- The number of households that used both brands of soap -/
def both : ℕ := 40

/-- The ratio of households that used only brand B soap to those that used both brands -/
def B_to_both_ratio : ℕ := 3

/-- The total number of households surveyed -/
def total_households : ℕ := neither + only_E + both + B_to_both_ratio * both

theorem survey_result : total_households = 300 := by
  sorry

end survey_result_l1980_198076


namespace equal_distances_l1980_198070

/-- The number of people seated at the round table. -/
def n : ℕ := 41

/-- The distance between two positions in a circular arrangement. -/
def circularDistance (a b : ℕ) : ℕ :=
  min ((a - b + n) % n) ((b - a + n) % n)

/-- Theorem stating that the distance from 31 to 7 equals the distance from 31 to 14
    in a circular arrangement of 41 people. -/
theorem equal_distances : circularDistance 31 7 = circularDistance 31 14 := by
  sorry


end equal_distances_l1980_198070


namespace expand_and_simplify_1_simplify_division_2_l1980_198042

-- Define variables
variable (a b : ℝ)

-- Theorem 1
theorem expand_and_simplify_1 : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b := by
  sorry

-- Theorem 2
theorem simplify_division_2 : (12 * a^3 - 6 * a^2 + 3 * a) / (3 * a) = 4 * a^2 - 2 * a + 1 := by
  sorry

end expand_and_simplify_1_simplify_division_2_l1980_198042


namespace sara_quarters_count_l1980_198029

theorem sara_quarters_count (initial_quarters final_quarters dad_quarters : ℕ) : 
  initial_quarters = 21 → dad_quarters = 49 → final_quarters = initial_quarters + dad_quarters → 
  final_quarters = 70 := by
sorry

end sara_quarters_count_l1980_198029


namespace partial_fraction_decomposition_l1980_198088

theorem partial_fraction_decomposition (x A B C : ℚ) :
  x ≠ 2 → x ≠ 4 → x ≠ 5 →
  ((x^2 - 9) / ((x - 2) * (x - 4) * (x - 5)) = A / (x - 2) + B / (x - 4) + C / (x - 5)) ↔
  (A = 5/3 ∧ B = -7/2 ∧ C = 8/3) :=
by sorry

end partial_fraction_decomposition_l1980_198088


namespace magnitude_of_z_l1980_198053

open Complex

theorem magnitude_of_z (z : ℂ) (h : i * (1 - z) = 1) : abs z = Real.sqrt 2 := by
  sorry

end magnitude_of_z_l1980_198053


namespace same_terminal_side_angle_with_same_terminal_side_l1980_198074

theorem same_terminal_side (α β : Real) : 
  ∃ k : Int, α = β + 2 * π * (k : Real) → 
  α.cos = β.cos ∧ α.sin = β.sin :=
by sorry

theorem angle_with_same_terminal_side : 
  ∃ k : Int, (11 * π / 8 : Real) = (-5 * π / 8 : Real) + 2 * π * (k : Real) :=
by sorry

end same_terminal_side_angle_with_same_terminal_side_l1980_198074


namespace set_A_is_empty_l1980_198002

theorem set_A_is_empty (a : ℝ) : {x : ℝ | |x - 1| ≤ 2*a - a^2 - 2} = ∅ := by
  sorry

end set_A_is_empty_l1980_198002


namespace s_square_minus_product_abs_eq_eight_l1980_198099

/-- The sequence s_n defined for three real numbers a, b, c -/
def s (a b c : ℝ) : ℕ → ℝ
  | 0 => 3  -- s_0 = a^0 + b^0 + c^0 = 3
  | n + 1 => a^(n + 1) + b^(n + 1) + c^(n + 1)

/-- The theorem statement -/
theorem s_square_minus_product_abs_eq_eight
  (a b c : ℝ)
  (h1 : s a b c 1 = 2)
  (h2 : s a b c 2 = 6)
  (h3 : s a b c 3 = 14) :
  ∀ n : ℕ, n > 1 → |(s a b c n)^2 - (s a b c (n-1)) * (s a b c (n+1))| = 8 := by
  sorry

end s_square_minus_product_abs_eq_eight_l1980_198099


namespace salt_mixture_proof_l1980_198008

theorem salt_mixture_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_volume : ℝ) (added_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 40 ∧ 
  initial_concentration = 0.2 ∧ 
  added_volume = 40 ∧ 
  added_concentration = 0.6 ∧ 
  final_concentration = 0.4 →
  (initial_volume * initial_concentration + added_volume * added_concentration) / 
  (initial_volume + added_volume) = final_concentration :=
by sorry

end salt_mixture_proof_l1980_198008


namespace stating_number_of_passed_candidates_l1980_198048

/-- Represents the number of candidates who passed the examination. -/
def passed_candidates : ℕ := 346

/-- Represents the total number of candidates. -/
def total_candidates : ℕ := 500

/-- Represents the average marks of all candidates. -/
def average_marks : ℚ := 60

/-- Represents the average marks of passed candidates. -/
def average_marks_passed : ℚ := 80

/-- Represents the average marks of failed candidates. -/
def average_marks_failed : ℚ := 15

/-- 
Theorem stating that the number of candidates who passed the examination is 346,
given the total number of candidates, average marks of all candidates,
average marks of passed candidates, and average marks of failed candidates.
-/
theorem number_of_passed_candidates : 
  passed_candidates = 346 ∧
  passed_candidates + (total_candidates - passed_candidates) = total_candidates ∧
  (passed_candidates * average_marks_passed + 
   (total_candidates - passed_candidates) * average_marks_failed) / total_candidates = average_marks :=
by sorry

end stating_number_of_passed_candidates_l1980_198048


namespace smallest_valid_N_exists_l1980_198087

def is_valid_configuration (N : ℕ) (c₁ c₂ c₃ c₄ c₅ c₆ : ℕ) : Prop :=
  c₁ ≤ N ∧ c₂ ≤ N ∧ c₃ ≤ N ∧ c₄ ≤ N ∧ c₅ ≤ N ∧ c₆ ≤ N ∧
  c₁ = 6 * c₂ - 1 ∧
  N + c₂ = 6 * c₃ - 2 ∧
  2 * N + c₃ = 6 * c₄ - 3 ∧
  3 * N + c₄ = 6 * c₅ - 4 ∧
  4 * N + c₅ = 6 * c₆ - 5 ∧
  5 * N + c₆ = 6 * c₁

theorem smallest_valid_N_exists :
  ∃ N : ℕ, N > 0 ∧ 
  (∃ c₁ c₂ c₃ c₄ c₅ c₆ : ℕ, is_valid_configuration N c₁ c₂ c₃ c₄ c₅ c₆) ∧
  (∀ M : ℕ, M < N → ¬∃ c₁ c₂ c₃ c₄ c₅ c₆ : ℕ, is_valid_configuration M c₁ c₂ c₃ c₄ c₅ c₆) :=
by sorry

end smallest_valid_N_exists_l1980_198087


namespace gcd_187_253_l1980_198046

theorem gcd_187_253 : Nat.gcd 187 253 = 11 := by sorry

end gcd_187_253_l1980_198046


namespace number_ratio_l1980_198023

theorem number_ratio (x : ℚ) (h : 3 * (2 * x + 9) = 81) : x / (2 * x) = 1 / 2 := by
  sorry

end number_ratio_l1980_198023


namespace geometric_sequence_product_l1980_198022

theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n) :
  a 2 * a 5 = 32 → a 4 * a 7 = 512 := by
  sorry

end geometric_sequence_product_l1980_198022


namespace division_scaling_l1980_198089

theorem division_scaling (a b c : ℝ) (h : a / b = c) :
  (a / 10) / (b / 10) = c := by
  sorry

end division_scaling_l1980_198089


namespace distinct_c_values_l1980_198090

theorem distinct_c_values (r s t u : ℂ) (h_distinct : r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ s ≠ t ∧ s ≠ u ∧ t ≠ u) 
  (h_eq : ∀ z : ℂ, (z - r) * (z - s) * (z - t) * (z - u) = 
    (z - r * c) * (z - s * c) * (z - t * c) * (z - u * c)) : 
  ∃! (values : Finset ℂ), values.card = 4 ∧ ∀ c : ℂ, c ∈ values ↔ 
    (∀ z : ℂ, (z - r) * (z - s) * (z - t) * (z - u) = 
      (z - r * c) * (z - s * c) * (z - t * c) * (z - u * c)) :=
by sorry

end distinct_c_values_l1980_198090


namespace jerry_spent_difference_l1980_198051

/-- Jerry's initial amount of money in dollars -/
def initial_amount : ℕ := 18

/-- Jerry's remaining amount of money in dollars -/
def remaining_amount : ℕ := 12

/-- The amount Jerry spent on video games -/
def amount_spent : ℕ := initial_amount - remaining_amount

theorem jerry_spent_difference :
  amount_spent = initial_amount - remaining_amount :=
by sorry

end jerry_spent_difference_l1980_198051


namespace power_of_product_l1980_198056

theorem power_of_product (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := by
  sorry

end power_of_product_l1980_198056


namespace min_blocks_for_wall_l1980_198050

/-- Represents the dimensions of a block -/
structure Block where
  length : Nat
  height : Nat

/-- Represents the dimensions of the wall -/
structure Wall where
  length : Nat
  height : Nat

/-- Calculates the minimum number of blocks needed to build the wall -/
def minBlocksNeeded (wall : Wall) (blocks : List Block) : Nat :=
  sorry

/-- The theorem to be proven -/
theorem min_blocks_for_wall :
  let wall : Wall := { length := 120, height := 9 }
  let blocks : List Block := [
    { length := 3, height := 1 },
    { length := 2, height := 1 },
    { length := 1, height := 1 }
  ]
  minBlocksNeeded wall blocks = 365 := by sorry

end min_blocks_for_wall_l1980_198050


namespace shaded_area_between_circles_l1980_198041

/-- The area of the region between two concentric circles, where the radius of the larger circle
    is three times the radius of the smaller circle, and the diameter of the smaller circle is 6 units. -/
theorem shaded_area_between_circles (π : ℝ) : ℝ := by
  -- Define the diameter of the smaller circle
  let small_diameter : ℝ := 6
  -- Define the radius of the smaller circle
  let small_radius : ℝ := small_diameter / 2
  -- Define the radius of the larger circle
  let large_radius : ℝ := 3 * small_radius
  -- Define the area of the shaded region
  let shaded_area : ℝ := π * large_radius^2 - π * small_radius^2
  -- Prove that the shaded area equals 72π
  have : shaded_area = 72 * π := by sorry
  -- Return the result
  exact 72 * π

end shaded_area_between_circles_l1980_198041


namespace polynomial_roots_l1980_198045

def polynomial (x : ℝ) : ℝ := x^3 - 5*x^2 + 3*x + 9

theorem polynomial_roots : 
  (polynomial (-1) = 0) ∧ 
  (polynomial 3 = 0) ∧ 
  (∃ (f : ℝ → ℝ), ∀ x, polynomial x = (x + 1) * (x - 3)^2 * f x) :=
by sorry

end polynomial_roots_l1980_198045


namespace prop_2_prop_3_l1980_198067

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- State that m and n are distinct
variable (h_distinct_lines : m ≠ n)

-- State that α and β are different
variable (h_different_planes : α ≠ β)

-- Proposition ②
theorem prop_2 : 
  (parallel_planes α β ∧ subset m α) → parallel_lines m β :=
sorry

-- Proposition ③
theorem prop_3 : 
  (perpendicular n α ∧ perpendicular n β ∧ perpendicular m α) → perpendicular m β :=
sorry

end prop_2_prop_3_l1980_198067


namespace sum_of_solutions_abs_equation_l1980_198031

theorem sum_of_solutions_abs_equation : 
  ∃ (x₁ x₂ : ℝ), 
    (|3 * x₁ - 5| = 8) ∧ 
    (|3 * x₂ - 5| = 8) ∧ 
    (x₁ + x₂ = 10 / 3) :=
by sorry

end sum_of_solutions_abs_equation_l1980_198031


namespace cosine_sum_inequality_l1980_198096

theorem cosine_sum_inequality (n : ℕ) (x : ℝ) :
  (Finset.range (n + 1)).sum (fun i => |Real.cos (2^i * x)|) ≥ n / 2 := by
  sorry

end cosine_sum_inequality_l1980_198096


namespace reflection_x_axis_l1980_198018

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The reflection of (-2, -3) across the x-axis is (-2, 3) -/
theorem reflection_x_axis : reflect_x (-2, -3) = (-2, 3) := by sorry

end reflection_x_axis_l1980_198018


namespace highest_temperature_correct_l1980_198073

/-- The highest temperature reached during candy making --/
def highest_temperature (initial_temp final_temp : ℝ) (heating_rate cooling_rate : ℝ) (total_time : ℝ) : ℝ :=
  let T : ℝ := 240
  T

/-- Theorem stating that the highest temperature is correct --/
theorem highest_temperature_correct 
  (initial_temp : ℝ) (final_temp : ℝ) (heating_rate : ℝ) (cooling_rate : ℝ) (total_time : ℝ)
  (h1 : initial_temp = 60)
  (h2 : final_temp = 170)
  (h3 : heating_rate = 5)
  (h4 : cooling_rate = 7)
  (h5 : total_time = 46) :
  let T := highest_temperature initial_temp final_temp heating_rate cooling_rate total_time
  (T - initial_temp) / heating_rate + (T - final_temp) / cooling_rate = total_time :=
by
  sorry

#check highest_temperature_correct

end highest_temperature_correct_l1980_198073


namespace polygon_with_720_degree_sum_is_hexagon_l1980_198057

/-- A polygon with interior angles summing to 720° has 6 sides -/
theorem polygon_with_720_degree_sum_is_hexagon :
  ∀ (n : ℕ), (n - 2) * 180 = 720 → n = 6 :=
by sorry

end polygon_with_720_degree_sum_is_hexagon_l1980_198057


namespace total_cost_proof_l1980_198065

def hand_mitts_cost : ℝ := 14
def apron_cost : ℝ := 16
def utensils_cost : ℝ := 10
def knife_cost : ℝ := 2 * utensils_cost
def discount_rate : ℝ := 0.25
def num_sets : ℕ := 3

def total_cost : ℝ := num_sets * ((hand_mitts_cost + apron_cost + utensils_cost + knife_cost) * (1 - discount_rate))

theorem total_cost_proof : total_cost = 135 := by
  sorry

end total_cost_proof_l1980_198065


namespace football_yardage_l1980_198016

theorem football_yardage (total_yardage running_yardage : ℕ) 
  (h1 : total_yardage = 150)
  (h2 : running_yardage = 90) :
  total_yardage - running_yardage = 60 := by
  sorry

end football_yardage_l1980_198016


namespace find_a_value_l1980_198069

theorem find_a_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end find_a_value_l1980_198069


namespace subtraction_problem_l1980_198028

theorem subtraction_problem (x : ℝ) (h : 40 / x = 5) : 20 - x = 12 := by
  sorry

end subtraction_problem_l1980_198028


namespace work_completion_time_l1980_198098

theorem work_completion_time (b a_and_b : ℝ) (hb : b = 8) (hab : a_and_b = 4.8) :
  let a := (1 / a_and_b - 1 / b)⁻¹
  a = 12 := by
sorry

end work_completion_time_l1980_198098


namespace smallest_sum_of_factors_l1980_198052

theorem smallest_sum_of_factors (x y z w : ℕ+) : 
  x * y * z * w = 362880 → 
  ∀ a b c d : ℕ+, a * b * c * d = 362880 → x + y + z + w ≤ a + b + c + d →
  x + y + z + w = 69 :=
sorry

end smallest_sum_of_factors_l1980_198052


namespace square_roots_problem_l1980_198000

theorem square_roots_problem (n : ℝ) (h : n > 0) :
  (∃ a : ℝ, (a + 2)^2 = n ∧ (2*a - 11)^2 = n) → n = 225 := by
sorry

end square_roots_problem_l1980_198000


namespace cara_seating_arrangements_l1980_198019

theorem cara_seating_arrangements (n : ℕ) (h : n = 6) : 
  Nat.choose n 2 = 15 := by
  sorry

end cara_seating_arrangements_l1980_198019


namespace boys_camp_total_l1980_198033

theorem boys_camp_total (total : ℕ) : 
  (total : ℝ) * 0.2 * 0.7 = 63 → total = 450 := by
  sorry

end boys_camp_total_l1980_198033


namespace chocolate_game_student_count_l1980_198034

def is_valid_student_count (n : ℕ) : Prop :=
  n > 0 ∧ (120 - 1) % n = 0

theorem chocolate_game_student_count :
  {n : ℕ | is_valid_student_count n} = {7, 17} := by
  sorry

end chocolate_game_student_count_l1980_198034


namespace range_of_c_over_a_l1980_198047

theorem range_of_c_over_a (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 1) (h3 : a + b + c = 0) :
  ∀ x, (x = c / a) → -2 < x ∧ x < -1 := by
sorry

end range_of_c_over_a_l1980_198047


namespace f_nonnegative_implies_a_eq_four_l1980_198082

/-- The function f(x) = ax^3 - 3x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3*x + 1

/-- Theorem: If f(x) ≥ 0 for all x in [-1, 1], then a = 4 -/
theorem f_nonnegative_implies_a_eq_four (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ 0) → a = 4 := by
  sorry

end f_nonnegative_implies_a_eq_four_l1980_198082


namespace symmetric_function_product_l1980_198094

/-- A function f(x) that is symmetric about the line x = 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (x^2 - 1) * (-x^2 + a*x - b)

/-- The symmetry condition for f(x) about x = 2 -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x : ℝ, f a b (2 - x) = f a b (2 + x)

/-- Theorem: If f(x) is symmetric about x = 2, then ab = 120 -/
theorem symmetric_function_product (a b : ℝ) :
  is_symmetric a b → a * b = 120 := by sorry

end symmetric_function_product_l1980_198094


namespace not_all_angles_exceed_90_l1980_198081

/-- A plane quadrilateral is a geometric figure with four sides and four angles in a plane. -/
structure PlaneQuadrilateral where
  angles : Fin 4 → ℝ
  sum_of_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360

/-- Theorem: In a plane quadrilateral, it is impossible for all four internal angles to exceed 90°. -/
theorem not_all_angles_exceed_90 (q : PlaneQuadrilateral) : 
  ¬(∀ i : Fin 4, q.angles i > 90) := by
  sorry

end not_all_angles_exceed_90_l1980_198081


namespace no_m_for_all_x_x_range_for_bounded_m_l1980_198015

-- Define the inequality function
def f (m x : ℝ) : ℝ := m * x^2 - 2*x - m + 1

-- Statement 1
theorem no_m_for_all_x : ¬ ∃ m : ℝ, ∀ x : ℝ, f m x < 0 := by sorry

-- Statement 2
theorem x_range_for_bounded_m :
  ∀ m : ℝ, |m| ≤ 2 →
  ∀ x : ℝ, ((-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2) →
  f m x < 0 := by sorry

end no_m_for_all_x_x_range_for_bounded_m_l1980_198015


namespace sufficient_not_necessary_l1980_198001

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → 2 / a < 1) ∧
  (∃ a, 2 / a < 1 ∧ a ≤ 2) :=
by sorry

end sufficient_not_necessary_l1980_198001


namespace min_value_theorem_l1980_198063

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f (x : ℝ) := x^2 - 2*x + 2
  let g (x : ℝ) := -x^2 + a*x + b
  let f' (x : ℝ) := 2*x - 2
  let g' (x : ℝ) := -2*x + a
  ∃ x₀ : ℝ, f x₀ = g x₀ ∧ f' x₀ * g' x₀ = -1 →
  (1/a + 4/b) ≥ 18/5 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 4/b₀ = 18/5 :=
by sorry

end min_value_theorem_l1980_198063
