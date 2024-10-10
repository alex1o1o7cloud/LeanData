import Mathlib

namespace parallelogram_sum_l2292_229240

/-- A parallelogram with sides 12, 4z + 2, 3x - 1, and 7y + 3 -/
structure Parallelogram (x y z : ℚ) where
  side1 : ℚ := 12
  side2 : ℚ := 4 * z + 2
  side3 : ℚ := 3 * x - 1
  side4 : ℚ := 7 * y + 3
  opposite_sides_equal1 : side1 = side3
  opposite_sides_equal2 : side2 = side4

/-- The sum of x, y, and z in the parallelogram equals 121/21 -/
theorem parallelogram_sum (x y z : ℚ) (p : Parallelogram x y z) : x + y + z = 121/21 := by
  sorry


end parallelogram_sum_l2292_229240


namespace hyperbola_standard_equation_l2292_229267

/-- The standard equation of a hyperbola with foci on the y-axis -/
def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  y^2 / a^2 - x^2 / b^2 = 1

/-- Theorem: The standard equation of a hyperbola with foci on the y-axis,
    semi-minor axis length of 4, and semi-focal distance of 6 -/
theorem hyperbola_standard_equation :
  let b : ℝ := 4  -- semi-minor axis length
  let c : ℝ := 6  -- semi-focal distance
  let a : ℝ := (c^2 - b^2).sqrt  -- semi-major axis length
  ∀ x y : ℝ, hyperbola_equation a b x y ↔ y^2 / 20 - x^2 / 16 = 1 :=
by sorry

end hyperbola_standard_equation_l2292_229267


namespace power_plus_one_prime_l2292_229260

theorem power_plus_one_prime (a n : ℕ) (ha : a > 1) (hprime : Nat.Prime (a^n + 1)) :
  Even a ∧ ∃ k : ℕ, n = 2^k :=
sorry

end power_plus_one_prime_l2292_229260


namespace scientific_notation_of_830_billion_l2292_229238

theorem scientific_notation_of_830_billion :
  (830 : ℝ) * (10^9 : ℝ) = 8.3 * (10^11 : ℝ) :=
by sorry

end scientific_notation_of_830_billion_l2292_229238


namespace swimmer_speed_l2292_229214

/-- A swimmer's speed in still water, given stream conditions -/
theorem swimmer_speed (v s : ℝ) (h1 : s = 1.5) (h2 : (v - s)⁻¹ = 2 * (v + s)⁻¹) : v = 4.5 := by
  sorry

end swimmer_speed_l2292_229214


namespace prime_even_intersection_l2292_229239

def P : Set ℕ := {n : ℕ | Nat.Prime n}
def Q : Set ℕ := {n : ℕ | Even n}

theorem prime_even_intersection : P ∩ Q = {2} := by
  sorry

end prime_even_intersection_l2292_229239


namespace new_person_age_l2292_229279

theorem new_person_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) : 
  n = 9 → initial_avg = 15 → new_avg = 17 → 
  ∃ (new_person_age : ℝ), 
    (n * initial_avg + new_person_age) / (n + 1) = new_avg ∧ 
    new_person_age = 35 := by
  sorry

end new_person_age_l2292_229279


namespace negation_equivalence_l2292_229221

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end negation_equivalence_l2292_229221


namespace room_dimension_is_15_l2292_229237

/-- Represents the dimensions and costs related to whitewashing a room --/
structure RoomWhitewash where
  length : ℝ
  width : ℝ
  height : ℝ
  doorLength : ℝ
  doorWidth : ℝ
  windowLength : ℝ
  windowWidth : ℝ
  numWindows : ℕ
  costPerSquareFoot : ℝ
  totalCost : ℝ

/-- Theorem stating that the unknown dimension of the room is 15 feet --/
theorem room_dimension_is_15 (r : RoomWhitewash) 
  (h1 : r.length = 25)
  (h2 : r.height = 12)
  (h3 : r.doorLength = 6)
  (h4 : r.doorWidth = 3)
  (h5 : r.windowLength = 4)
  (h6 : r.windowWidth = 3)
  (h7 : r.numWindows = 3)
  (h8 : r.costPerSquareFoot = 4)
  (h9 : r.totalCost = 3624)
  : r.width = 15 := by
  sorry

end room_dimension_is_15_l2292_229237


namespace school_supplies_cost_l2292_229293

/-- The total cost of school supplies given the number of cartons, boxes per carton, and cost per unit -/
def total_cost (pencil_cartons : ℕ) (pencil_boxes_per_carton : ℕ) (pencil_cost_per_box : ℕ)
                (marker_cartons : ℕ) (marker_boxes_per_carton : ℕ) (marker_cost_per_carton : ℕ) : ℕ :=
  (pencil_cartons * pencil_boxes_per_carton * pencil_cost_per_box) +
  (marker_cartons * marker_cost_per_carton)

/-- Theorem stating that the total cost for the school's purchase is $440 -/
theorem school_supplies_cost :
  total_cost 20 10 2 10 5 4 = 440 := by
  sorry

end school_supplies_cost_l2292_229293


namespace circle_equation_l2292_229297

/-- The equation of a circle in its general form -/
def is_circle (h x y a : ℝ) : Prop :=
  ∃ (c_x c_y r : ℝ), (x - c_x)^2 + (y - c_y)^2 = r^2 ∧ r > 0

/-- The given equation -/
def given_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 5*a = 0

theorem circle_equation (x y : ℝ) :
  is_circle 0 x y 1 ↔ given_equation x y 1 :=
sorry

end circle_equation_l2292_229297


namespace davids_physics_marks_l2292_229205

theorem davids_physics_marks :
  let english_marks : ℕ := 51
  let math_marks : ℕ := 65
  let chemistry_marks : ℕ := 67
  let biology_marks : ℕ := 85
  let average_marks : ℕ := 70
  let total_subjects : ℕ := 5

  let total_marks : ℕ := average_marks * total_subjects
  let known_marks : ℕ := english_marks + math_marks + chemistry_marks + biology_marks
  let physics_marks : ℕ := total_marks - known_marks

  physics_marks = 82 :=
by
  sorry

end davids_physics_marks_l2292_229205


namespace bowling_ball_volume_l2292_229250

/-- The remaining volume of a bowling ball after drilling holes -/
theorem bowling_ball_volume (π : ℝ) (h : π > 0) : 
  let sphere_volume := (4/3) * π * (12^3)
  let small_hole_volume := π * (3/2)^2 * 10
  let large_hole_volume := π * 2^2 * 10
  sphere_volume - (2 * small_hole_volume + large_hole_volume) = 2219 * π := by
  sorry

#check bowling_ball_volume

end bowling_ball_volume_l2292_229250


namespace prime_factorization_of_large_number_l2292_229203

theorem prime_factorization_of_large_number :
  1007021035035021007001 = 7^7 * 11^7 * 13^7 := by
  sorry

end prime_factorization_of_large_number_l2292_229203


namespace A_intersect_B_equals_closed_open_interval_l2292_229274

def A : Set ℝ := {x : ℝ | |x| ≥ 2}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

theorem A_intersect_B_equals_closed_open_interval :
  A ∩ B = Set.Icc 2 3 \ {3} :=
by sorry

end A_intersect_B_equals_closed_open_interval_l2292_229274


namespace odd_function_when_c_zero_increasing_when_b_zero_central_symmetry_more_than_two_roots_possible_l2292_229276

-- Define the function f
def f (b c x : ℝ) : ℝ := x * |x| + b * x + c

-- Statement 1
theorem odd_function_when_c_zero (b : ℝ) :
  (∀ x : ℝ, f b 0 (-x) = -(f b 0 x)) := by sorry

-- Statement 2
theorem increasing_when_b_zero (c : ℝ) :
  Monotone (f 0 c) := by sorry

-- Statement 3
theorem central_symmetry (b c : ℝ) :
  ∀ x : ℝ, f b c (-x) + f b c x = 2 * c := by sorry

-- Statement 4 (negation of the original statement)
theorem more_than_two_roots_possible :
  ∃ b c : ℝ, ∃ x₁ x₂ x₃ : ℝ, (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
  (f b c x₁ = 0 ∧ f b c x₂ = 0 ∧ f b c x₃ = 0) := by sorry

end odd_function_when_c_zero_increasing_when_b_zero_central_symmetry_more_than_two_roots_possible_l2292_229276


namespace quadratic_inequality_solution_set_l2292_229295

theorem quadratic_inequality_solution_set (x : ℝ) :
  3 * x^2 + 7 * x + 2 < 0 ↔ -1 < x ∧ x < -2/3 := by
  sorry

end quadratic_inequality_solution_set_l2292_229295


namespace pie_sugar_percentage_l2292_229225

/-- Given a pie weighing 200 grams with 50 grams of sugar, 
    prove that 75% of the pie is not sugar. -/
theorem pie_sugar_percentage 
  (total_weight : ℝ) 
  (sugar_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : sugar_weight = 50) : 
  (total_weight - sugar_weight) / total_weight * 100 = 75 := by
  sorry

end pie_sugar_percentage_l2292_229225


namespace parabola_coefficient_sum_min_l2292_229234

/-- Given a parabola y = ax^2 + bx + c with positive integer coefficients that intersects
    the x-axis at two distinct points within distance 1 of the origin, 
    the sum of its coefficients is at least 11. -/
theorem parabola_coefficient_sum_min (a b c : ℕ+) 
  (h_distinct : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0)
  (h_distance : ∀ x : ℝ, a * x^2 + b * x + c = 0 → |x| < 1) :
  a + b + c ≥ 11 := by
  sorry

end parabola_coefficient_sum_min_l2292_229234


namespace soda_packs_minimum_l2292_229232

def min_packs (total : ℕ) (pack_sizes : List ℕ) : ℕ :=
  sorry

theorem soda_packs_minimum :
  min_packs 120 [8, 15, 30] = 4 :=
sorry

end soda_packs_minimum_l2292_229232


namespace base8_square_unique_l2292_229227

/-- Converts a base-10 number to base-8 --/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 8) :: aux (m / 8)
  aux n |>.reverse

/-- Checks if a list contains each number from 0 to 7 exactly once --/
def containsEachDigitOnce (l : List ℕ) : Prop :=
  ∀ d, d ∈ Finset.range 8 → (l.count d = 1)

/-- The main theorem --/
theorem base8_square_unique : 
  ∃! n : ℕ, 
    (toBase8 n).length = 3 ∧ 
    containsEachDigitOnce (toBase8 n) ∧
    containsEachDigitOnce (toBase8 (n * n)) ∧
    n = 256 := by sorry

end base8_square_unique_l2292_229227


namespace compute_expression_l2292_229263

theorem compute_expression : 20 * (144 / 3 + 36 / 6 + 16 / 32 + 2) = 1130 := by
  sorry

end compute_expression_l2292_229263


namespace total_protest_days_l2292_229285

/-- Given a first protest lasting 4 days and a second protest lasting 25% longer,
    prove that the total number of days spent protesting is 9. -/
theorem total_protest_days : 
  let first_protest_days : ℕ := 4
  let second_protest_days : ℕ := first_protest_days + first_protest_days / 4
  first_protest_days + second_protest_days = 9 := by sorry

end total_protest_days_l2292_229285


namespace smallest_ten_digit_divisible_by_first_five_primes_l2292_229271

/-- The product of the first five prime numbers -/
def first_five_primes_product : ℕ := 2 * 3 * 5 * 7 * 11

/-- A number is a 10-digit number if it's between 1000000000 and 9999999999 -/
def is_ten_digit (n : ℕ) : Prop := 1000000000 ≤ n ∧ n ≤ 9999999999

theorem smallest_ten_digit_divisible_by_first_five_primes :
  ∀ n : ℕ, is_ten_digit n ∧ n % first_five_primes_product = 0 → n ≥ 1000000310 :=
by sorry

end smallest_ten_digit_divisible_by_first_five_primes_l2292_229271


namespace area_of_annular_region_area_of_specific_annular_region_l2292_229244

/-- The area of an annular region between two concentric circles -/
theorem area_of_annular_region (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > r₁) : 
  π * r₂^2 - π * r₁^2 = π * (r₂^2 - r₁^2) :=
by sorry

/-- The area of the annular region between two concentric circles with radii 4 and 7 is 33π -/
theorem area_of_specific_annular_region : 
  π * 7^2 - π * 4^2 = 33 * π :=
by sorry

end area_of_annular_region_area_of_specific_annular_region_l2292_229244


namespace range_of_a_l2292_229212

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l2292_229212


namespace blake_poured_out_02_gallons_l2292_229255

/-- The amount of water Blake poured out, given initial and remaining amounts -/
def water_poured_out (initial : Real) (remaining : Real) : Real :=
  initial - remaining

/-- Theorem: Blake poured out 0.2 gallons of water -/
theorem blake_poured_out_02_gallons :
  let initial := 0.8
  let remaining := 0.6
  water_poured_out initial remaining = 0.2 := by
  sorry

end blake_poured_out_02_gallons_l2292_229255


namespace lcm_36_105_l2292_229286

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l2292_229286


namespace num_adults_on_trip_l2292_229243

def total_eggs : ℕ := 36
def eggs_per_adult : ℕ := 3
def num_girls : ℕ := 7
def num_boys : ℕ := 10
def eggs_per_girl : ℕ := 1
def eggs_per_boy : ℕ := eggs_per_girl + 1

theorem num_adults_on_trip : 
  total_eggs - (num_girls * eggs_per_girl + num_boys * eggs_per_boy) = 3 * eggs_per_adult := by
  sorry

end num_adults_on_trip_l2292_229243


namespace not_right_triangle_l2292_229235

theorem not_right_triangle (a b c : ℝ) : 
  (a = 3 ∧ b = 4 ∧ c = 5) ∨ 
  (a = 1 ∧ b = Real.sqrt 3 ∧ c = 2) ∨ 
  (a = Real.sqrt 11 ∧ b = 2 ∧ c = 4) ∨ 
  (a^2 = (c+b)*(c-b)) →
  (¬(a^2 + b^2 = c^2) ↔ a = Real.sqrt 11 ∧ b = 2 ∧ c = 4) :=
by sorry

end not_right_triangle_l2292_229235


namespace complement_of_A_in_U_l2292_229201

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1}

theorem complement_of_A_in_U : (U \ A) = {0, 2} := by sorry

end complement_of_A_in_U_l2292_229201


namespace cubic_root_sum_l2292_229249

theorem cubic_root_sum (a b c : ℝ) : 
  (a^3 = a + 1) → (b^3 = b + 1) → (c^3 = c + 1) →
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -9 := by
sorry

end cubic_root_sum_l2292_229249


namespace max_true_statements_l2292_229278

theorem max_true_statements (y : ℝ) : 
  let statements := [
    (0 < y^3 ∧ y^3 < 2),
    (y^3 > 2),
    (-2 < y ∧ y < 0),
    (0 < y ∧ y < 2),
    (0 < y - y^3 ∧ y - y^3 < 2)
  ]
  ∀ (s : Finset (Fin 5)), (∀ i ∈ s, statements[i]) → s.card ≤ 2 :=
by sorry

end max_true_statements_l2292_229278


namespace winning_team_fourth_quarter_points_l2292_229277

/-- The points scored by the winning team in the fourth quarter of a basketball game. -/
def fourth_quarter_points (first_quarter_losing : ℕ) 
                          (second_quarter_increase : ℕ) 
                          (third_quarter_increase : ℕ) 
                          (total_points : ℕ) : ℕ :=
  let first_quarter_winning := 2 * first_quarter_losing
  let second_quarter_winning := first_quarter_winning + second_quarter_increase
  let third_quarter_winning := second_quarter_winning + third_quarter_increase
  total_points - third_quarter_winning

/-- Theorem stating that the winning team scored 30 points in the fourth quarter. -/
theorem winning_team_fourth_quarter_points : 
  fourth_quarter_points 10 10 20 80 = 30 := by
  sorry

end winning_team_fourth_quarter_points_l2292_229277


namespace sin_negative_1740_degrees_l2292_229262

theorem sin_negative_1740_degrees : 
  Real.sin ((-1740 : ℝ) * π / 180) = (Real.sqrt 3) / 2 := by sorry

end sin_negative_1740_degrees_l2292_229262


namespace tangent_segment_length_l2292_229280

/-- Given a square and a circle with the following properties:
    - The circle has a radius of 10
    - The circle is tangent to two adjacent sides of the square
    - The circle intersects the other two sides of the square, cutting off segments of 4 and 2 from the vertices
    This theorem proves that the length of the segment cut off from the vertex at the point of tangency is 8. -/
theorem tangent_segment_length (square_side : ℝ) (circle_radius : ℝ) (cut_segment1 : ℝ) (cut_segment2 : ℝ) :
  circle_radius = 10 →
  cut_segment1 = 4 →
  cut_segment2 = 2 →
  square_side = circle_radius + (square_side - cut_segment1 - cut_segment2) / 2 →
  square_side - circle_radius = 8 :=
by sorry

end tangent_segment_length_l2292_229280


namespace baseton_transaction_baseton_base_equation_baseton_base_value_l2292_229275

/-- The base of the number system in Baseton -/
def r : ℕ := sorry

/-- The cost of the laptop in base r -/
def laptop_cost : ℕ := 534

/-- The amount paid in base r -/
def amount_paid : ℕ := 1000

/-- The change received in base r -/
def change_received : ℕ := 366

/-- Conversion from base r to base 10 -/
def to_base_10 (n : ℕ) : ℕ := 
  (n / 100) * r^2 + ((n / 10) % 10) * r + (n % 10)

theorem baseton_transaction :
  to_base_10 laptop_cost + to_base_10 change_received = to_base_10 amount_paid :=
by sorry

theorem baseton_base_equation :
  r^3 - 8*r^2 - 9*r - 10 = 0 :=
by sorry

theorem baseton_base_value : r = 10 :=
by sorry

end baseton_transaction_baseton_base_equation_baseton_base_value_l2292_229275


namespace work_problem_solution_l2292_229208

def work_problem (a_days b_days remaining_days : ℚ) : Prop :=
  let a_rate : ℚ := 1 / a_days
  let b_rate : ℚ := 1 / b_days
  let combined_rate : ℚ := a_rate + b_rate
  let x : ℚ := 2  -- Days A and B worked together
  combined_rate * x + b_rate * remaining_days = 1

theorem work_problem_solution :
  work_problem 4 8 2 = true :=
sorry

end work_problem_solution_l2292_229208


namespace equal_quotient_remainder_divisible_by_seven_l2292_229229

theorem equal_quotient_remainder_divisible_by_seven :
  {n : ℕ | ∃ (q : ℕ), n = 7 * q + q ∧ q < 7} = {8, 16, 24, 32, 40, 48} := by
  sorry

end equal_quotient_remainder_divisible_by_seven_l2292_229229


namespace valid_monomial_l2292_229211

def is_valid_monomial (m : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b : ℕ), ∀ x y, m x y = -2 * x^a * y^b ∧ a + b = 3

theorem valid_monomial : 
  is_valid_monomial (fun x y ↦ -2 * x^2 * y) := by sorry

end valid_monomial_l2292_229211


namespace infinite_primes_dividing_f_values_l2292_229241

theorem infinite_primes_dividing_f_values
  (f : ℕ+ → ℕ+)
  (h_non_constant : ∃ a b : ℕ+, f a ≠ f b)
  (h_divides : ∀ a b : ℕ+, a ≠ b → (a - b) ∣ (f a - f b)) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ c : ℕ+, p ∣ f c} :=
sorry

end infinite_primes_dividing_f_values_l2292_229241


namespace prime_sum_squares_l2292_229231

theorem prime_sum_squares (p q m : ℕ) : 
  p.Prime → q.Prime → p ≠ q →
  p^2 - 2001*p + m = 0 →
  q^2 - 2001*q + m = 0 →
  p^2 + q^2 = 3996005 := by
  sorry

end prime_sum_squares_l2292_229231


namespace sqrt_180_simplification_l2292_229265

theorem sqrt_180_simplification : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end sqrt_180_simplification_l2292_229265


namespace tan_ratio_difference_l2292_229200

theorem tan_ratio_difference (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) - (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) - (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) - (Real.tan y / Real.tan x) = 2/3 := by
  sorry

end tan_ratio_difference_l2292_229200


namespace max_area_enclosure_l2292_229253

/-- Represents a rectangular enclosure. -/
structure Enclosure where
  length : ℝ
  width : ℝ

/-- The perimeter of the enclosure is exactly 420 feet. -/
def perimeterConstraint (e : Enclosure) : Prop :=
  2 * e.length + 2 * e.width = 420

/-- The length of the enclosure is at least 100 feet. -/
def lengthConstraint (e : Enclosure) : Prop :=
  e.length ≥ 100

/-- The width of the enclosure is at least 60 feet. -/
def widthConstraint (e : Enclosure) : Prop :=
  e.width ≥ 60

/-- The area of the enclosure. -/
def area (e : Enclosure) : ℝ :=
  e.length * e.width

/-- The theorem stating that the maximum area is achieved when length = width = 105 feet. -/
theorem max_area_enclosure :
  ∀ e : Enclosure,
    perimeterConstraint e → lengthConstraint e → widthConstraint e →
    area e ≤ 11025 ∧
    (area e = 11025 ↔ e.length = 105 ∧ e.width = 105) :=
by sorry

end max_area_enclosure_l2292_229253


namespace compute_expression_l2292_229252

theorem compute_expression : 25 * (216 / 3 + 49 / 7 + 16 / 25 + 2) = 2041 := by
  sorry

end compute_expression_l2292_229252


namespace landscape_breadth_l2292_229256

/-- Proves that the breadth of a rectangular landscape is 480 meters given the specified conditions -/
theorem landscape_breadth :
  ∀ (length breadth : ℝ),
  breadth = 8 * length →
  3200 = (1 / 9) * (length * breadth) →
  breadth = 480 :=
by
  sorry

end landscape_breadth_l2292_229256


namespace bus_ride_time_l2292_229230

def total_trip_time : ℕ := 8 * 60  -- 8 hours in minutes
def walk_time : ℕ := 15
def train_ride_time : ℕ := 6 * 60  -- 6 hours in minutes

def wait_time : ℕ := 2 * walk_time

def time_without_bus : ℕ := train_ride_time + walk_time + wait_time

theorem bus_ride_time : total_trip_time - time_without_bus = 75 := by
  sorry

end bus_ride_time_l2292_229230


namespace sam_non_black_cows_l2292_229288

/-- Given a herd of cows, calculate the number of non-black cows. -/
def non_black_cows (total : ℕ) (black : ℕ) : ℕ :=
  total - black

theorem sam_non_black_cows :
  let total := 18
  let black := (total / 2) + 5
  non_black_cows total black = 4 := by
sorry

end sam_non_black_cows_l2292_229288


namespace newspapers_sold_l2292_229272

theorem newspapers_sold (total : ℝ) (magazines : ℕ) 
  (h1 : total = 425.0) (h2 : magazines = 150) : 
  total - magazines = 275 := by
  sorry

end newspapers_sold_l2292_229272


namespace cloth_sale_calculation_l2292_229217

/-- Given a shopkeeper selling cloth with a total selling price, loss per metre, and cost price per metre,
    prove that the number of metres sold is as calculated. -/
theorem cloth_sale_calculation
  (total_selling_price : ℕ)
  (loss_per_metre : ℕ)
  (cost_price_per_metre : ℕ)
  (h1 : total_selling_price = 36000)
  (h2 : loss_per_metre = 10)
  (h3 : cost_price_per_metre = 70) :
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 600 := by
  sorry

#check cloth_sale_calculation

end cloth_sale_calculation_l2292_229217


namespace branch_A_more_profitable_choose_branch_A_l2292_229219

/-- Represents the grades of products --/
inductive Grade
| A
| B
| C
| D

/-- Represents the branches of the factory --/
inductive Branch
| A
| B

/-- Processing fee for each grade --/
def processingFee (g : Grade) : Int :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Processing cost for each branch --/
def processingCost (b : Branch) : Int :=
  match b with
  | Branch.A => 25
  | Branch.B => 20

/-- Frequency distribution for each branch --/
def frequency (b : Branch) (g : Grade) : Int :=
  match b, g with
  | Branch.A, Grade.A => 40
  | Branch.A, Grade.B => 20
  | Branch.A, Grade.C => 20
  | Branch.A, Grade.D => 20
  | Branch.B, Grade.A => 28
  | Branch.B, Grade.B => 17
  | Branch.B, Grade.C => 34
  | Branch.B, Grade.D => 21

/-- Calculate average profit for a branch --/
def averageProfit (b : Branch) : Int :=
  (processingFee Grade.A - processingCost b) * frequency b Grade.A +
  (processingFee Grade.B - processingCost b) * frequency b Grade.B +
  (processingFee Grade.C - processingCost b) * frequency b Grade.C +
  (processingFee Grade.D - processingCost b) * frequency b Grade.D

/-- Theorem: Branch A has higher average profit than Branch B --/
theorem branch_A_more_profitable :
  averageProfit Branch.A > averageProfit Branch.B :=
by sorry

/-- Corollary: Factory should choose Branch A --/
theorem choose_branch_A :
  ∀ b : Branch, b ≠ Branch.A → averageProfit Branch.A > averageProfit b :=
by sorry

end branch_A_more_profitable_choose_branch_A_l2292_229219


namespace tinas_career_win_loss_difference_l2292_229228

/-- Represents Tina's boxing career -/
structure BoxingCareer where
  initial_wins : ℕ
  additional_wins_before_first_loss : ℕ
  wins_doubled : Bool

/-- Calculates the total number of wins in Tina's career -/
def total_wins (career : BoxingCareer) : ℕ :=
  let wins_before_doubling := career.initial_wins + career.additional_wins_before_first_loss
  if career.wins_doubled then
    2 * wins_before_doubling
  else
    wins_before_doubling

/-- Calculates the total number of losses in Tina's career -/
def total_losses (career : BoxingCareer) : ℕ :=
  if career.wins_doubled then 2 else 1

/-- Theorem stating the difference between wins and losses in Tina's career -/
theorem tinas_career_win_loss_difference :
  ∀ (career : BoxingCareer),
    career.initial_wins = 10 →
    career.additional_wins_before_first_loss = 5 →
    career.wins_doubled = true →
    total_wins career - total_losses career = 28 := by
  sorry

end tinas_career_win_loss_difference_l2292_229228


namespace calculate_expression_l2292_229202

theorem calculate_expression : (Real.pi - Real.sqrt 3) ^ 0 - 2 * Real.sin (45 * π / 180) + |-Real.sqrt 2| + Real.sqrt 8 = 1 + 2 * Real.sqrt 2 := by
  sorry

end calculate_expression_l2292_229202


namespace valid_numbers_l2292_229206

def is_valid_number (N : ℕ) : Prop :=
  ∃ (a b k : ℕ),
    10 ≤ a ∧ a ≤ 99 ∧
    0 ≤ b ∧ b < 10^k ∧
    N = 10^k * a + b ∧
    Odd N ∧
    10^k * a + b = 149 * b

theorem valid_numbers :
  ∀ N : ℕ, is_valid_number N → (N = 745 ∨ N = 3725) :=
by sorry

end valid_numbers_l2292_229206


namespace topsoil_cost_calculation_l2292_229216

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The amount of topsoil in cubic yards -/
def topsoil_amount : ℝ := 8

/-- The total cost of topsoil in dollars -/
def total_cost : ℝ := topsoil_amount * cubic_yards_to_cubic_feet * topsoil_cost_per_cubic_foot

theorem topsoil_cost_calculation : total_cost = 1728 := by
  sorry

end topsoil_cost_calculation_l2292_229216


namespace system_solution_l2292_229210

theorem system_solution : 
  ∃ (x y : ℚ), 4 * x - 35 * y = -1 ∧ 3 * y - x = 5 ∧ x = -172/23 ∧ y = -19/23 := by
  sorry

end system_solution_l2292_229210


namespace problem_solution_l2292_229284

theorem problem_solution : ∃ (S L x : ℕ), 
  S = 18 ∧ 
  S + L = 51 ∧ 
  L = 2 * S - x ∧ 
  x > 0 ∧ 
  x = 3 := by
  sorry

end problem_solution_l2292_229284


namespace booklet_word_count_l2292_229223

theorem booklet_word_count (total_pages : Nat) (max_words_per_page : Nat) (remainder : Nat) (modulus : Nat) : 
  total_pages = 154 →
  max_words_per_page = 120 →
  remainder = 207 →
  modulus = 221 →
  ∃ (words_per_page : Nat),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % modulus = remainder ∧
    words_per_page = 100 := by
  sorry

end booklet_word_count_l2292_229223


namespace test_maximum_marks_l2292_229236

theorem test_maximum_marks :
  let passing_percentage : ℚ := 60 / 100
  let student_score : ℕ := 80
  let marks_needed_to_pass : ℕ := 100
  let maximum_marks : ℕ := 300
  passing_percentage * maximum_marks = student_score + marks_needed_to_pass →
  maximum_marks = 300 := by
sorry

end test_maximum_marks_l2292_229236


namespace x_squared_coefficient_l2292_229233

def expand_polynomial (x : ℝ) := x * (x - 1) * (x + 1)^4

theorem x_squared_coefficient :
  ∃ (a b c d e : ℝ),
    expand_polynomial x = a*x^5 + b*x^4 + c*x^3 + 5*x^2 + d*x + e :=
by
  sorry

end x_squared_coefficient_l2292_229233


namespace fifteen_subcommittees_l2292_229289

/-- The number of ways to form a two-person sub-committee from a larger committee,
    where one member must be from a designated group. -/
def subcommittee_count (total : ℕ) (designated : ℕ) : ℕ :=
  designated * (total - designated)

/-- Theorem stating that for a committee of 8 people with a designated group of 3,
    there are 15 possible two-person sub-committees. -/
theorem fifteen_subcommittees :
  subcommittee_count 8 3 = 15 := by
  sorry

#eval subcommittee_count 8 3

end fifteen_subcommittees_l2292_229289


namespace toddler_count_l2292_229281

theorem toddler_count (bill_count : ℕ) (double_counted : ℕ) (missed : ℕ) : 
  bill_count = 26 → double_counted = 8 → missed = 3 → 
  bill_count - double_counted + missed = 21 := by
sorry

end toddler_count_l2292_229281


namespace expression_evaluation_l2292_229215

theorem expression_evaluation :
  let a : ℚ := -1/3
  (3*a - 1)^2 + 3*a*(3*a + 2) = 3 := by sorry

end expression_evaluation_l2292_229215


namespace perpendicular_vectors_l2292_229248

/-- Given vectors a and b, find the unique value of t such that a is perpendicular to (t * a + b) -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (1, -1)) (h2 : b = (6, -4)) :
  ∃! t : ℝ, (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) ∧ t = -5 := by
  sorry

end perpendicular_vectors_l2292_229248


namespace fraction_equality_existence_l2292_229283

theorem fraction_equality_existence :
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c + (1 : ℚ) / d) ∧
  (∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c + (1 : ℚ) / d + (1 : ℚ) / e) :=
by sorry

end fraction_equality_existence_l2292_229283


namespace real_roots_quadratic_equation_l2292_229287

theorem real_roots_quadratic_equation (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x - 1 = 0) ↔ k ≥ -4 := by
  sorry

end real_roots_quadratic_equation_l2292_229287


namespace gretchen_earnings_l2292_229213

/-- Calculates the total earnings for Gretchen's caricature drawings over a weekend -/
def weekend_earnings (price_per_drawing : ℕ) (saturday_sales : ℕ) (sunday_sales : ℕ) : ℕ :=
  price_per_drawing * (saturday_sales + sunday_sales)

/-- Proves that Gretchen's earnings for the weekend are $800 -/
theorem gretchen_earnings :
  weekend_earnings 20 24 16 = 800 := by
sorry

end gretchen_earnings_l2292_229213


namespace fixed_point_exponential_function_l2292_229294

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1)
  f 1 = 1 := by sorry

end fixed_point_exponential_function_l2292_229294


namespace matrix_equality_l2292_229218

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = 2 * A * B)
  (h2 : A * B = !![5, 1; -2, 4]) :
  B * A = !![10, 2; -4, 8] := by
sorry

end matrix_equality_l2292_229218


namespace factors_of_8_to_15_l2292_229270

/-- The number of positive factors of 8^15 is 46 -/
theorem factors_of_8_to_15 : Nat.card (Nat.divisors (8^15)) = 46 := by
  sorry

end factors_of_8_to_15_l2292_229270


namespace sine_equation_solution_l2292_229246

theorem sine_equation_solution (x : ℝ) (h1 : 3 * Real.sin (2 * x) = 2 * Real.sin x) 
  (h2 : 0 < x ∧ x < π) : x = Real.arccos (1/3) := by
  sorry

end sine_equation_solution_l2292_229246


namespace range_of_f_l2292_229299

open Real

theorem range_of_f (x : ℝ) (h : x ∈ Set.Icc 0 (π / 2)) :
  let f := λ x : ℝ => 3 * sin (2 * x - π / 6)
  ∃ y, y ∈ Set.Icc (-3/2) 3 ∧ ∃ x, x ∈ Set.Icc 0 (π / 2) ∧ f x = y :=
by sorry

end range_of_f_l2292_229299


namespace factorization_cubic_minus_quadratic_l2292_229247

theorem factorization_cubic_minus_quadratic (x y : ℝ) :
  y^3 - 4*x^2*y = y*(y+2*x)*(y-2*x) := by sorry

end factorization_cubic_minus_quadratic_l2292_229247


namespace start_page_second_day_l2292_229261

/-- Given a book with 200 pages, and 20% read on the first day,
    prove that the page number to start reading on the second day is 41. -/
theorem start_page_second_day (total_pages : ℕ) (percent_read : ℚ) : 
  total_pages = 200 → percent_read = 1/5 → 
  (total_pages : ℚ) * percent_read + 1 = 41 := by
  sorry

end start_page_second_day_l2292_229261


namespace point_b_value_l2292_229226

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- The distance between two points on a number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_b_value (a b : Point) :
  a.value = 1 → distance a b = 3 → b.value = 4 ∨ b.value = -2 := by
  sorry

end point_b_value_l2292_229226


namespace angle_trisector_theorem_l2292_229266

/-- 
Given a triangle ABC with angle γ = ∠ACB, if the trisectors of γ divide 
the opposite side AB into segments d, e, f, then cos²(γ/3) = ((d+e)(e+f))/(4df)
-/
theorem angle_trisector_theorem (d e f : ℝ) (γ : ℝ) 
  (h1 : d > 0) (h2 : e > 0) (h3 : f > 0) (h4 : γ > 0) (h5 : γ < π) :
  (Real.cos (γ / 3))^2 = ((d + e) * (e + f)) / (4 * d * f) :=
sorry

end angle_trisector_theorem_l2292_229266


namespace lily_total_books_l2292_229254

def mike_books_tuesday : ℕ := 45
def corey_books_tuesday : ℕ := 2 * mike_books_tuesday
def mike_gave_to_lily : ℕ := 10
def corey_gave_to_lily : ℕ := mike_gave_to_lily + 15

theorem lily_total_books : mike_gave_to_lily + corey_gave_to_lily = 35 :=
by sorry

end lily_total_books_l2292_229254


namespace arithmetic_geometric_mean_inequality_l2292_229209

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end arithmetic_geometric_mean_inequality_l2292_229209


namespace integer_roots_iff_m_in_M_l2292_229282

/-- The set of values for m where the equation has only integer roots -/
def M : Set ℝ := {3, 7, 15, 6, 9}

/-- The quadratic equation in x parameterized by m -/
def equation (m : ℝ) (x : ℝ) : ℝ :=
  (m - 6) * (m - 9) * x^2 + (15 * m - 117) * x + 54

/-- A predicate to check if a real number is an integer -/
def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- The main theorem stating that the equation has only integer roots iff m ∈ M -/
theorem integer_roots_iff_m_in_M (m : ℝ) : 
  (∀ x : ℝ, equation m x = 0 → is_integer x) ↔ m ∈ M := by sorry

end integer_roots_iff_m_in_M_l2292_229282


namespace car_y_time_is_one_third_correct_graph_is_c_l2292_229259

/-- Represents a car's travel characteristics -/
structure Car where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The scenario of two cars traveling the same distance -/
def TwoCarScenario (x y : Car) : Prop :=
  x.distance = y.distance ∧ y.speed = 3 * x.speed

/-- Theorem: In the given scenario, Car Y's time is one-third of Car X's time -/
theorem car_y_time_is_one_third (x y : Car) 
  (h : TwoCarScenario x y) : y.time = x.time / 3 := by
  sorry

/-- Theorem: The correct graph representation matches option C -/
theorem correct_graph_is_c (x y : Car) 
  (h : TwoCarScenario x y) : 
  (x.speed = y.speed / 3 ∧ x.time = y.time * 3) := by
  sorry

end car_y_time_is_one_third_correct_graph_is_c_l2292_229259


namespace x_values_l2292_229251

theorem x_values (x n : ℕ) (h1 : x = 2^n - 32) 
  (h2 : (Nat.factors x).card = 3) 
  (h3 : 3 ∈ Nat.factors x) : 
  x = 480 ∨ x = 2016 := by
sorry

end x_values_l2292_229251


namespace range_of_function_l2292_229292

theorem range_of_function (x : ℝ) (h : x^2 ≥ 1) :
  x^2 + Real.sqrt (x^2 - 1) ≥ 1 := by
  sorry

end range_of_function_l2292_229292


namespace punch_water_calculation_l2292_229222

/-- Calculates the amount of water needed for a punch mixture -/
def water_needed (total_volume : ℚ) (water_parts : ℕ) (juice_parts : ℕ) : ℚ :=
  (total_volume * water_parts) / (water_parts + juice_parts)

/-- Theorem stating the amount of water needed for the specific punch recipe -/
theorem punch_water_calculation :
  water_needed 3 5 3 = 15 / 8 := by
  sorry

end punch_water_calculation_l2292_229222


namespace average_visitors_is_288_l2292_229257

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def average_visitors_per_day (sunday_visitors : ℕ) (other_day_visitors : ℕ) : ℚ :=
  let num_sundays : ℕ := 30 / 7
  let num_other_days : ℕ := 30 - num_sundays
  let total_visitors : ℕ := sunday_visitors * num_sundays + other_day_visitors * num_other_days
  (total_visitors : ℚ) / 30

/-- Theorem stating that the average number of visitors per day is 288 -/
theorem average_visitors_is_288 :
  average_visitors_per_day 600 240 = 288 := by
  sorry

end average_visitors_is_288_l2292_229257


namespace gcd_upper_bound_l2292_229258

theorem gcd_upper_bound (a b : ℕ+) : Nat.gcd a.val b.val ≤ Real.sqrt (a.val + b.val : ℝ) := by sorry

end gcd_upper_bound_l2292_229258


namespace ababab_divisible_by_13_l2292_229269

theorem ababab_divisible_by_13 (a b : Nat) (h1 : a < 10) (h2 : b < 10) :
  ∃ k : Nat, 100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b = 13 * k := by
  sorry

end ababab_divisible_by_13_l2292_229269


namespace circumscribed_iff_similar_when_moved_l2292_229207

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- A function to check if a polygon is convex -/
def isConvex (p : Polygon) : Prop :=
  sorry

/-- A function to move all sides of a polygon outward by a distance -/
def moveOutward (p : Polygon) (distance : ℝ) : Polygon :=
  sorry

/-- A function to check if two polygons are similar -/
def areSimilar (p1 p2 : Polygon) : Prop :=
  sorry

/-- A function to check if a polygon is circumscribed -/
def isCircumscribed (p : Polygon) : Prop :=
  sorry

/-- Theorem: A convex polygon is circumscribed if and only if 
    moving all its sides outward by a distance of 1 results 
    in a polygon similar to the original one -/
theorem circumscribed_iff_similar_when_moved (p : Polygon) :
  isConvex p →
  isCircumscribed p ↔ areSimilar p (moveOutward p 1) :=
by sorry

end circumscribed_iff_similar_when_moved_l2292_229207


namespace select_four_with_girl_l2292_229242

/-- The number of ways to select 4 people from 4 boys and 2 girls with at least one girl -/
def select_with_girl (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose boys to_select

theorem select_four_with_girl :
  select_with_girl 6 4 2 4 = 14 := by
  sorry

end select_four_with_girl_l2292_229242


namespace quadratic_equation_solution_l2292_229224

theorem quadratic_equation_solution :
  ∃ (x1 x2 : ℝ), 
    x1 > 0 ∧ x2 > 0 ∧
    (1/2 * (4 * x1^2 - 1) = (x1^2 - 75*x1 - 15) * (x1^2 + 50*x1 + 10)) ∧
    (1/2 * (4 * x2^2 - 1) = (x2^2 - 75*x2 - 15) * (x2^2 + 50*x2 + 10)) ∧
    x1 = (75 + Real.sqrt 5773) / 2 ∧
    x2 = (-50 + Real.sqrt 2356) / 2 :=
by sorry

end quadratic_equation_solution_l2292_229224


namespace target_scientific_notation_l2292_229298

/-- Represents one billion in decimal notation -/
def billion : ℕ := 100000000

/-- The number we want to express in scientific notation -/
def target : ℕ := 1360000000

/-- Scientific notation for the target number -/
def scientific_notation (n : ℕ) : ℚ := 1.36 * (10 : ℚ) ^ n

theorem target_scientific_notation :
  ∃ n : ℕ, scientific_notation n = target ∧ n = 9 := by
  sorry

end target_scientific_notation_l2292_229298


namespace joes_lift_l2292_229291

theorem joes_lift (total : ℕ) (diff : ℕ) (first_lift : ℕ) (second_lift : ℕ) 
  (h1 : total = 600)
  (h2 : first_lift + second_lift = total)
  (h3 : 2 * first_lift = second_lift + diff)
  (h4 : diff = 300) : 
  first_lift = 300 := by
sorry

end joes_lift_l2292_229291


namespace cubic_resonance_intervals_sqrt_resonance_interval_l2292_229220

-- Definition of a resonance interval
def is_resonance_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a ≤ b ∧
  Monotone f ∧
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

-- Theorem for the cubic function
theorem cubic_resonance_intervals :
  (is_resonance_interval (fun x ↦ x^3) (-1) 0) ∧
  (is_resonance_interval (fun x ↦ x^3) (-1) 1) ∧
  (is_resonance_interval (fun x ↦ x^3) 0 1) :=
sorry

-- Theorem for the square root function
theorem sqrt_resonance_interval (k : ℝ) :
  (∃ a b, is_resonance_interval (fun x ↦ Real.sqrt (x + 1) - k) a b) ↔
  (1 ≤ k ∧ k < 5/4) :=
sorry

end cubic_resonance_intervals_sqrt_resonance_interval_l2292_229220


namespace water_container_problem_l2292_229268

/-- Given a container with capacity 120 liters, if adding 48 liters makes it 3/4 full,
    then the initial percentage of water in the container was 35%. -/
theorem water_container_problem :
  let capacity : ℝ := 120
  let added_water : ℝ := 48
  let final_fraction : ℝ := 3/4
  let initial_percentage : ℝ := (final_fraction * capacity - added_water) / capacity * 100
  initial_percentage = 35 := by sorry

end water_container_problem_l2292_229268


namespace jerry_shelf_difference_l2292_229273

/-- Calculates the difference between action figures and books on Jerry's shelf -/
def action_figure_book_difference (
  initial_books : ℕ
  ) (initial_action_figures : ℕ)
  (added_action_figures : ℕ) : ℕ :=
  (initial_action_figures + added_action_figures) - initial_books

/-- Proves that the difference between action figures and books is 3 -/
theorem jerry_shelf_difference :
  action_figure_book_difference 3 4 2 = 3 := by
  sorry

end jerry_shelf_difference_l2292_229273


namespace simplify_fraction_product_l2292_229290

theorem simplify_fraction_product : 
  (256 : ℚ) / 20 * (10 : ℚ) / 160 * ((16 : ℚ) / 6)^2 = (256 : ℚ) / 45 := by
  sorry

end simplify_fraction_product_l2292_229290


namespace intersection_A_B_l2292_229264

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - 2*x ≤ 0}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end intersection_A_B_l2292_229264


namespace mass_of_Al2O3_solution_l2292_229245

-- Define the atomic masses
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_O : ℝ := 16.00

-- Define the volume and concentration of the solution
def volume : ℝ := 2.5
def concentration : ℝ := 4

-- Define the molecular weight of Al2O3
def molecular_weight_Al2O3 : ℝ := 2 * atomic_mass_Al + 3 * atomic_mass_O

-- State the theorem
theorem mass_of_Al2O3_solution :
  let moles : ℝ := volume * concentration
  let mass : ℝ := moles * molecular_weight_Al2O3
  mass = 1019.6 := by sorry

end mass_of_Al2O3_solution_l2292_229245


namespace vanessa_camera_pictures_l2292_229204

/-- The number of pictures Vanessa uploaded from her camera -/
def camera_pictures (phone_pictures album_count pictures_per_album : ℕ) : ℕ :=
  album_count * pictures_per_album - phone_pictures

/-- Proof that Vanessa uploaded 7 pictures from her camera -/
theorem vanessa_camera_pictures :
  camera_pictures 23 5 6 = 7 := by
  sorry

end vanessa_camera_pictures_l2292_229204


namespace largest_root_l2292_229296

theorem largest_root (p q r : ℝ) 
  (sum_eq : p + q + r = 1)
  (sum_prod_eq : p * q + p * r + q * r = -8)
  (prod_eq : p * q * r = 15) :
  max p (max q r) = 3 := by sorry

end largest_root_l2292_229296
