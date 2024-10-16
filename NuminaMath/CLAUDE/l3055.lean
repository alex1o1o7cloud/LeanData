import Mathlib

namespace NUMINAMATH_CALUDE_salary_increase_after_five_years_l3055_305598

/-- The annual raise rate -/
def annual_raise : ℝ := 1.08

/-- The number of years -/
def years : ℕ := 5

/-- The total percentage increase after 'years' number of raises -/
def total_increase : ℝ := (annual_raise ^ years - 1) * 100

/-- Theorem stating that the total percentage increase after 5 years of 8% annual raises is approximately 47% -/
theorem salary_increase_after_five_years :
  ∃ ε > 0, abs (total_increase - 47) < ε := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_after_five_years_l3055_305598


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3055_305585

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  a : ℝ
  is_positive : 0 < a

/-- Checks if the given side lengths can form a valid triangle -/
def is_valid_triangle (t : IsoscelesTriangle) : Prop :=
  3 + t.a > 6 ∧ 3 + 6 > t.a ∧ t.a + 6 > 3

/-- Calculates the perimeter of the triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  3 + t.a + 6

/-- Theorem: If a valid isosceles triangle can be formed with side lengths 3, a, and 6,
    then its perimeter is 15 -/
theorem isosceles_triangle_perimeter
  (t : IsoscelesTriangle)
  (h : is_valid_triangle t) :
  perimeter t = 15 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3055_305585


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l3055_305571

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) :
  perpPlane m α → perpPlane n β → perp m n → perpPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l3055_305571


namespace NUMINAMATH_CALUDE_white_tshirts_per_package_l3055_305521

theorem white_tshirts_per_package (total_tshirts : ℕ) (num_packages : ℕ) 
  (h1 : total_tshirts = 426) (h2 : num_packages = 71) :
  total_tshirts / num_packages = 6 := by
  sorry

end NUMINAMATH_CALUDE_white_tshirts_per_package_l3055_305521


namespace NUMINAMATH_CALUDE_gcf_of_lcms_main_result_l3055_305508

theorem gcf_of_lcms (a b c d : ℕ) : 
  Nat.gcd (Nat.lcm a b) (Nat.lcm c d) = Nat.gcd (Nat.lcm 16 21) (Nat.lcm 14 18) := by
  sorry

theorem main_result : Nat.gcd (Nat.lcm 16 21) (Nat.lcm 14 18) = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_main_result_l3055_305508


namespace NUMINAMATH_CALUDE_days_per_month_is_30_l3055_305502

/-- Represents the number of trees a single logger can cut down in one day. -/
def trees_per_logger_per_day : ℕ := 6

/-- Represents the length of the forest in miles. -/
def forest_length : ℕ := 4

/-- Represents the width of the forest in miles. -/
def forest_width : ℕ := 6

/-- Represents the number of trees in each square mile of the forest. -/
def trees_per_square_mile : ℕ := 600

/-- Represents the number of loggers working on cutting down the trees. -/
def num_loggers : ℕ := 8

/-- Represents the number of months it takes to cut down all trees. -/
def num_months : ℕ := 10

/-- Theorem stating that the number of days in each month is 30. -/
theorem days_per_month_is_30 :
  ∃ (days_per_month : ℕ),
    days_per_month = 30 ∧
    (forest_length * forest_width * trees_per_square_mile =
     num_loggers * trees_per_logger_per_day * num_months * days_per_month) :=
by sorry

end NUMINAMATH_CALUDE_days_per_month_is_30_l3055_305502


namespace NUMINAMATH_CALUDE_set_membership_implies_value_l3055_305538

theorem set_membership_implies_value (m : ℤ) : 
  3 ∈ ({1, m+2} : Set ℤ) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_value_l3055_305538


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l3055_305572

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = ![![3, -2], ![1, 4]] →
  (B^3)⁻¹ = ![![7, -70], ![35, 42]] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l3055_305572


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_by_nine_l3055_305596

def original_number : ℕ := 228712

theorem least_addition_for_divisibility_by_nine :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), m < n → ¬((original_number + m) % 9 = 0)) ∧
  ((original_number + n) % 9 = 0) :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_by_nine_l3055_305596


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3055_305554

theorem arithmetic_geometric_sequence_ratio 
  (a b c : ℝ) 
  (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_arith : b - a = c - b) 
  (h_geom : c^2 = a * b) : 
  ∃ (k : ℝ), k ≠ 0 ∧ a = 4*k ∧ b = k ∧ c = -2*k := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3055_305554


namespace NUMINAMATH_CALUDE_jenny_friends_count_l3055_305557

theorem jenny_friends_count (cost_per_night : ℕ) (nights : ℕ) (total_cost : ℕ) : 
  cost_per_night = 40 →
  nights = 3 →
  total_cost = 360 →
  (1 + 2) * (cost_per_night * nights) = total_cost :=
by
  sorry

#check jenny_friends_count

end NUMINAMATH_CALUDE_jenny_friends_count_l3055_305557


namespace NUMINAMATH_CALUDE_prob_roll_three_l3055_305514

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Nat)
  (fair : sides = 6)

/-- The probability of rolling a specific number on a fair die -/
def prob_roll (d : FairDie) (n : Nat) : ℚ :=
  1 / d.sides

/-- The sequence of previous rolls -/
def previous_rolls : List Nat := [6, 6, 6, 6, 6, 6]

/-- Theorem: The probability of rolling a 3 on a fair six-sided die is 1/6,
    regardless of previous rolls -/
theorem prob_roll_three (d : FairDie) (prev : List Nat) :
  prob_roll d 3 = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_prob_roll_three_l3055_305514


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3055_305500

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 15) → (∃ m : ℤ, N = 13 * m + 2) :=
by sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3055_305500


namespace NUMINAMATH_CALUDE_min_digit_sum_of_sum_l3055_305507

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_are_different (a b : ℕ) : Prop :=
  let digits_a := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let digits_b := [(b / 100) % 10, (b / 10) % 10, b % 10]
  ∀ i j, i ≠ j → (digits_a ++ digits_b).nthLe i sorry ≠ (digits_a ++ digits_b).nthLe j sorry

def first_digit_less_than_five (n : ℕ) : Prop := (n / 100) % 10 < 5 ∧ (n / 100) % 10 ≠ 0

def digit_sum (n : ℕ) : ℕ := (n / 100) % 10 + (n / 10) % 10 + n % 10

theorem min_digit_sum_of_sum (a b : ℕ) :
  is_three_digit a →
  is_three_digit b →
  digits_are_different a b →
  first_digit_less_than_five a →
  is_three_digit (a + b) →
  ∀ (x y : ℕ), 
    is_three_digit x →
    is_three_digit y →
    digits_are_different x y →
    first_digit_less_than_five x →
    is_three_digit (x + y) →
    digit_sum (a + b) ≤ digit_sum (x + y) ∧
    digit_sum (a + b) = 15 :=
sorry

end NUMINAMATH_CALUDE_min_digit_sum_of_sum_l3055_305507


namespace NUMINAMATH_CALUDE_special_shape_perimeter_l3055_305563

/-- A shape with right angles, a base of 12 feet, 10 congruent sides of 2 feet each, and an area of 132 square feet -/
structure SpecialShape where
  base : ℝ
  congruent_side : ℝ
  num_congruent_sides : ℕ
  area : ℝ
  base_eq : base = 12
  congruent_side_eq : congruent_side = 2
  num_congruent_sides_eq : num_congruent_sides = 10
  area_eq : area = 132

/-- The perimeter of the SpecialShape is 54 feet -/
theorem special_shape_perimeter (s : SpecialShape) : 
  s.base + s.congruent_side * s.num_congruent_sides + 4 + 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_special_shape_perimeter_l3055_305563


namespace NUMINAMATH_CALUDE_cave_depth_calculation_l3055_305518

theorem cave_depth_calculation (total_depth remaining_distance : ℕ) 
  (h1 : total_depth = 974)
  (h2 : remaining_distance = 386) :
  total_depth - remaining_distance = 588 := by
sorry

end NUMINAMATH_CALUDE_cave_depth_calculation_l3055_305518


namespace NUMINAMATH_CALUDE_forgotten_angles_sum_l3055_305501

/-- The sum of interior angles of a polygon with n sides --/
def polygon_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

/-- A convex polygon with n sides where n ≥ 3 --/
structure ConvexPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

theorem forgotten_angles_sum (p : ConvexPolygon) 
  (partial_sum : ℝ) (h_partial_sum : partial_sum = 2345) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 175 ∧ 
  polygon_angle_sum p.n = partial_sum + a + b := by
  sorry

end NUMINAMATH_CALUDE_forgotten_angles_sum_l3055_305501


namespace NUMINAMATH_CALUDE_bus_average_speed_l3055_305524

/-- The average speed of a bus traveling three equal-length sections of a road -/
theorem bus_average_speed (a : ℝ) (h : a > 0) : 
  let v1 : ℝ := 50  -- speed of first section in km/h
  let v2 : ℝ := 30  -- speed of second section in km/h
  let v3 : ℝ := 70  -- speed of third section in km/h
  let total_distance : ℝ := 3 * a  -- total distance traveled
  let total_time : ℝ := a / v1 + a / v2 + a / v3  -- total time taken
  let average_speed : ℝ := total_distance / total_time
  ∃ (ε : ℝ), ε > 0 ∧ |average_speed - 44| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_bus_average_speed_l3055_305524


namespace NUMINAMATH_CALUDE_common_solution_y_value_l3055_305505

theorem common_solution_y_value : ∃! y : ℝ, 
  ∃ x : ℝ, (x^2 + y^2 - 4 = 0) ∧ (x^2 - 4*y + 8 = 0) ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_common_solution_y_value_l3055_305505


namespace NUMINAMATH_CALUDE_lcm_problem_l3055_305580

theorem lcm_problem (a b c : ℕ+) (ha : a = 10) (hc : c = 20) (hlcm : Nat.lcm a (Nat.lcm b c) = 140) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3055_305580


namespace NUMINAMATH_CALUDE_total_triangles_is_sixteen_l3055_305540

/-- Represents the count of triangles in each size category -/
structure TriangleCounts where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of triangles -/
def totalTriangles (counts : TriangleCounts) : Nat :=
  counts.small + counts.medium + counts.large

/-- The given triangle counts for the figure -/
def figureCounts : TriangleCounts :=
  { small := 11, medium := 4, large := 1 }

/-- Theorem stating that the total number of triangles in the figure is 16 -/
theorem total_triangles_is_sixteen :
  totalTriangles figureCounts = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_sixteen_l3055_305540


namespace NUMINAMATH_CALUDE_total_spent_is_64_l3055_305590

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * (victor_decks + friend_decks)

/-- Proof that Victor and his friend spent $64 in total -/
theorem total_spent_is_64 :
  total_spent 8 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_64_l3055_305590


namespace NUMINAMATH_CALUDE_smallest_natural_with_remainder_one_l3055_305517

theorem smallest_natural_with_remainder_one : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 3 = 1 ∧ 
  n % 5 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 3 = 1 ∧ m % 5 = 1 → n ≤ m) ∧
  n = 16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_natural_with_remainder_one_l3055_305517


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l3055_305549

theorem divisibility_of_sum_of_squares (p k a b : ℤ) : 
  Prime p → 
  p = 4*k + 3 → 
  p ∣ (a^2 + b^2) → 
  p ∣ a ∧ p ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l3055_305549


namespace NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_of_180_l3055_305544

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def prime_factors (n : ℕ) : Set ℕ := {p : ℕ | is_prime p ∧ p ∣ n}

theorem sum_two_smallest_prime_factors_of_180 :
  ∃ (p q : ℕ), p ∈ prime_factors 180 ∧ q ∈ prime_factors 180 ∧
  p < q ∧
  (∀ r ∈ prime_factors 180, r ≠ p → r ≥ q) ∧
  p + q = 5 :=
sorry

end NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_of_180_l3055_305544


namespace NUMINAMATH_CALUDE_x_value_l3055_305547

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 6 = 3 * y) : x = 18 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3055_305547


namespace NUMINAMATH_CALUDE_average_difference_l3055_305576

theorem average_difference : 
  let set1 : List ℝ := [10, 20, 60]
  let set2 : List ℝ := [10, 40, 25]
  (set1.sum / set1.length) - (set2.sum / set2.length) = 5 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l3055_305576


namespace NUMINAMATH_CALUDE_smallest_n_with_property_l3055_305595

def has_property (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = Finset.range (n - 2) ⊔ {3, 4} → A ∩ B = ∅ →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a * b = c)

theorem smallest_n_with_property :
  (∀ k < 243, ¬ has_property k) ∧ has_property 243 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_property_l3055_305595


namespace NUMINAMATH_CALUDE_distance_AB_is_five_halves_l3055_305506

-- Define the lines l₁ and l₂
def l₁ (t : ℝ) : ℝ × ℝ := (1 + 3*t, 2 - 4*t)
def l₂ (x y : ℝ) : Prop := 2*x - 4*y = 5

-- Define point A
def A : ℝ × ℝ := (1, 2)

-- Define the intersection point B
def B : ℝ × ℝ := sorry

-- Theorem statement
theorem distance_AB_is_five_halves :
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 5/2 :=
sorry

end NUMINAMATH_CALUDE_distance_AB_is_five_halves_l3055_305506


namespace NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3055_305568

theorem fewer_bees_than_flowers :
  let num_flowers : ℕ := 5
  let num_bees : ℕ := 3
  num_flowers - num_bees = 2 := by
sorry

end NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3055_305568


namespace NUMINAMATH_CALUDE_peters_erasers_l3055_305584

theorem peters_erasers (x : ℕ) : x + 3 = 11 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_peters_erasers_l3055_305584


namespace NUMINAMATH_CALUDE_product_selection_proof_l3055_305599

def total_products : ℕ := 100
def qualified_products : ℕ := 98
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem product_selection_proof :
  (Nat.choose total_products selected_products = 161700) ∧
  (Nat.choose defective_products 1 * Nat.choose qualified_products 2 = 9506) ∧
  (Nat.choose total_products selected_products - Nat.choose qualified_products selected_products = 9604) :=
by sorry

end NUMINAMATH_CALUDE_product_selection_proof_l3055_305599


namespace NUMINAMATH_CALUDE_equation_one_real_root_l3055_305588

/-- The equation x + √(x-4) = 6 has exactly one real root. -/
theorem equation_one_real_root :
  ∃! x : ℝ, x + Real.sqrt (x - 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_real_root_l3055_305588


namespace NUMINAMATH_CALUDE_real_y_condition_l3055_305545

theorem real_y_condition (x y : ℝ) : 
  (∃ y, 2 * y^2 + 3 * x * y - x + 8 = 0) ↔ (x ≤ -23/9 ∨ x ≥ 5/3) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l3055_305545


namespace NUMINAMATH_CALUDE_nathan_basketball_games_l3055_305537

/-- Calculates the number of basketball games played given the number of air hockey games,
    the cost per game, and the total tokens used. -/
def basketball_games (air_hockey_games : ℕ) (cost_per_game : ℕ) (total_tokens : ℕ) : ℕ :=
  (total_tokens - air_hockey_games * cost_per_game) / cost_per_game

/-- Proves that Nathan played 4 basketball games given the problem conditions. -/
theorem nathan_basketball_games :
  basketball_games 2 3 18 = 4 := by
  sorry

#eval basketball_games 2 3 18

end NUMINAMATH_CALUDE_nathan_basketball_games_l3055_305537


namespace NUMINAMATH_CALUDE_polynomial_no_integral_roots_l3055_305533

/-- A polynomial with integral coefficients that has odd integer values at 0 and 1 has no integral roots. -/
theorem polynomial_no_integral_roots 
  (p : Polynomial ℤ) 
  (h0 : Odd (p.eval 0)) 
  (h1 : Odd (p.eval 1)) : 
  ∀ (x : ℤ), p.eval x ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_no_integral_roots_l3055_305533


namespace NUMINAMATH_CALUDE_boys_usual_time_to_school_l3055_305520

theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  (usual_time > 0) →
  (3 / 2 * usual_rate * (usual_time - 4) = usual_rate * usual_time) →
  usual_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_usual_time_to_school_l3055_305520


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_l3055_305513

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4cm,
    prove that the area of the sector formed by this central angle is 4 cm². -/
theorem sector_area_from_arc_length (r : ℝ) (h : 2 * r = 4) : 
  (1 / 2) * r^2 * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_length_l3055_305513


namespace NUMINAMATH_CALUDE_park_area_l3055_305566

/-- A rectangular park with specific length-width relationship and perimeter --/
structure RectangularPark where
  width : ℝ
  length : ℝ
  length_eq : length = 3 * width + 30
  perimeter_eq : 2 * (length + width) = 780

/-- The area of the rectangular park is 27000 square meters --/
theorem park_area (park : RectangularPark) : park.length * park.width = 27000 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l3055_305566


namespace NUMINAMATH_CALUDE_solve_for_x_l3055_305531

theorem solve_for_x (M N : ℝ) (h1 : M = 2*x - 4) (h2 : N = 2*x + 3) (h3 : 3*M - N = 1) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3055_305531


namespace NUMINAMATH_CALUDE_sarah_copies_pages_l3055_305562

theorem sarah_copies_pages (people meeting_size copies_per_person contract_pages : ℕ) 
  (h1 : meeting_size = 15)
  (h2 : copies_per_person = 5)
  (h3 : contract_pages = 35) :
  people = meeting_size * copies_per_person * contract_pages :=
by sorry

end NUMINAMATH_CALUDE_sarah_copies_pages_l3055_305562


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3055_305532

/-- Given an arithmetic sequence where the third term is 23 and the sixth term is 29,
    prove that the ninth term is 35. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℕ)  -- a is the sequence
  (h1 : a 3 = 23)  -- third term is 23
  (h2 : a 6 = 29)  -- sixth term is 29
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- arithmetic sequence property
  : a 9 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3055_305532


namespace NUMINAMATH_CALUDE_red_card_value_is_three_l3055_305525

/-- The value of a red card in credits -/
def red_card_value : ℕ := sorry

/-- The value of a blue card in credits -/
def blue_card_value : ℕ := 5

/-- The total number of cards needed to play a game -/
def total_cards : ℕ := 20

/-- The total number of credits available to buy cards -/
def total_credits : ℕ := 84

/-- The number of red cards used when playing -/
def red_cards_used : ℕ := 8

theorem red_card_value_is_three :
  red_card_value = 3 :=
by sorry

end NUMINAMATH_CALUDE_red_card_value_is_three_l3055_305525


namespace NUMINAMATH_CALUDE_simple_interest_months_l3055_305522

/-- Simple interest calculation -/
theorem simple_interest_months (principal : ℝ) (rate : ℝ) (interest : ℝ) : 
  principal = 10000 →
  rate = 0.08 →
  interest = 800 →
  (interest / (principal * rate)) * 12 = 12 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_months_l3055_305522


namespace NUMINAMATH_CALUDE_sum_of_integers_l3055_305546

theorem sum_of_integers (a b : ℕ+) 
  (h1 : a.val^2 + b.val^2 = 585)
  (h2 : Nat.gcd a.val b.val + Nat.lcm a.val b.val = 87) :
  a.val + b.val = 33 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3055_305546


namespace NUMINAMATH_CALUDE_smallest_n_sqrt_difference_l3055_305581

theorem smallest_n_sqrt_difference (n : ℕ) : n ≥ 626 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.02 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_sqrt_difference_l3055_305581


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3055_305541

open Real

theorem min_value_trig_expression (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) :
  ∃ (min_val : ℝ), min_val = 9 * sqrt 3 ∧
  ∀ θ', 0 < θ' ∧ θ' < π / 2 →
    3 * sin θ' + 4 * (1 / cos θ') + 2 * sqrt 3 * tan θ' ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3055_305541


namespace NUMINAMATH_CALUDE_min_tiles_for_square_l3055_305510

def tile_length : ℕ := 6
def tile_width : ℕ := 4

def tile_area : ℕ := tile_length * tile_width

def square_side : ℕ := Nat.lcm tile_length tile_width

theorem min_tiles_for_square :
  (square_side * square_side) / tile_area = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_square_l3055_305510


namespace NUMINAMATH_CALUDE_chloe_profit_l3055_305560

/-- Calculates the profit from selling chocolate-dipped strawberries -/
def strawberry_profit (cost_per_dozen : ℚ) (price_per_half_dozen : ℚ) (dozens_sold : ℕ) : ℚ :=
  let profit_per_half_dozen := price_per_half_dozen - (cost_per_dozen / 2)
  let total_half_dozens := dozens_sold * 2
  profit_per_half_dozen * total_half_dozens

/-- Theorem: Chloe's profit from selling chocolate-dipped strawberries is $500 -/
theorem chloe_profit :
  strawberry_profit 50 30 50 = 500 := by
  sorry

end NUMINAMATH_CALUDE_chloe_profit_l3055_305560


namespace NUMINAMATH_CALUDE_log_product_range_l3055_305515

theorem log_product_range :
  let y := Real.log 6 / Real.log 5 *
           Real.log 7 / Real.log 6 *
           Real.log 8 / Real.log 7 *
           Real.log 9 / Real.log 8 *
           Real.log 10 / Real.log 9
  1 < y ∧ y < 2 := by sorry

end NUMINAMATH_CALUDE_log_product_range_l3055_305515


namespace NUMINAMATH_CALUDE_subtraction_problem_l3055_305509

theorem subtraction_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3055_305509


namespace NUMINAMATH_CALUDE_log_equation_solution_l3055_305583

-- Define the logarithm function for base 16
noncomputable def log16 (x : ℝ) : ℝ := Real.log x / Real.log 16

-- State the theorem
theorem log_equation_solution :
  ∀ y : ℝ, log16 (3 * y - 4) = 2 → y = 260 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3055_305583


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3055_305503

/-- Two arithmetic sequences and their sum sequences -/
structure ArithmeticSequencePair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum sequence for a
  T : ℕ → ℚ  -- Sum sequence for b

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequencePair)
  (h_sum_ratio : ∀ n : ℕ, seq.S n / seq.T n = 2 * n / (3 * n + 1)) :
  seq.a 10 / seq.b 10 = 19 / 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3055_305503


namespace NUMINAMATH_CALUDE_hall_area_l3055_305543

/-- The area of a rectangular hall with given length and breadth relationship -/
theorem hall_area (length breadth : ℝ) : 
  length = 30 ∧ length = breadth + 5 → length * breadth = 750 := by
  sorry

end NUMINAMATH_CALUDE_hall_area_l3055_305543


namespace NUMINAMATH_CALUDE_expression_nonnegative_iff_x_in_interval_l3055_305556

/-- The expression (x-12x^2+36x^3)/(9-x^3) is nonnegative if and only if x is in the interval [0, 3). -/
theorem expression_nonnegative_iff_x_in_interval :
  ∀ x : ℝ, (x - 12 * x^2 + 36 * x^3) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Icc 0 3 ∧ x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonnegative_iff_x_in_interval_l3055_305556


namespace NUMINAMATH_CALUDE_rectangle_cutting_l3055_305582

theorem rectangle_cutting (a b : ℕ) (h_ab : a ≤ b) 
  (h_2 : a * (b - 1) + b * (a - 1) = 940)
  (h_3 : a * (b - 2) + b * (a - 2) = 894) :
  a * (b - 4) + b * (a - 4) = 802 :=
sorry

end NUMINAMATH_CALUDE_rectangle_cutting_l3055_305582


namespace NUMINAMATH_CALUDE_matches_played_is_ten_l3055_305567

/-- The number of matches a player has played, given their current average and the effect of a future match on that average. -/
def matches_played (current_average : ℚ) (future_score : ℚ) (average_increase : ℚ) : ℕ :=
  let n : ℕ := sorry
  n

/-- Theorem stating that the number of matches played is 10 under the given conditions. -/
theorem matches_played_is_ten :
  matches_played 32 76 4 = 10 := by sorry

end NUMINAMATH_CALUDE_matches_played_is_ten_l3055_305567


namespace NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_l3055_305512

theorem ellipse_foci_on_y_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 / (25 - m) + y^2 / (16 + m) = 1) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧
    c^2 = b^2 - a^2 ∧ 
    ∀ x y : ℝ, (x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) →
  m > 9/2 ∧ m < 25 :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_l3055_305512


namespace NUMINAMATH_CALUDE_no_periodic_sum_with_periods_2_and_pi_div_2_l3055_305527

/-- A function is periodic if it takes at least two different values and there exists a positive period. -/
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ ∃ p > 0, ∀ x, f (x + p) = f x

/-- The period of a function is a positive real number p such that f(x + p) = f(x) for all x. -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

theorem no_periodic_sum_with_periods_2_and_pi_div_2 :
  ¬ ∃ (g h : ℝ → ℝ),
    Periodic g ∧ Periodic h ∧ IsPeriod g 2 ∧ IsPeriod h (π / 2) ∧ Periodic (g + h) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_sum_with_periods_2_and_pi_div_2_l3055_305527


namespace NUMINAMATH_CALUDE_amanda_remaining_money_l3055_305589

/-- Calculates the remaining money after purchases -/
def remaining_money (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Theorem: Amanda's remaining money after purchases -/
theorem amanda_remaining_money :
  remaining_money 50 9 2 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_amanda_remaining_money_l3055_305589


namespace NUMINAMATH_CALUDE_complex_number_modulus_l3055_305591

theorem complex_number_modulus : Complex.abs ((1 - Complex.I) / Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l3055_305591


namespace NUMINAMATH_CALUDE_min_sum_p_q_l3055_305594

theorem min_sum_p_q (p q : ℝ) : 
  0 < p → 0 < q → 
  (∃ x : ℝ, x^2 + p*x + 2*q = 0) → 
  (∃ x : ℝ, x^2 + 2*q*x + p = 0) → 
  6 ≤ p + q ∧ ∃ p₀ q₀ : ℝ, 0 < p₀ ∧ 0 < q₀ ∧ p₀ + q₀ = 6 ∧ 
    (∃ x : ℝ, x^2 + p₀*x + 2*q₀ = 0) ∧ 
    (∃ x : ℝ, x^2 + 2*q₀*x + p₀ = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_p_q_l3055_305594


namespace NUMINAMATH_CALUDE_function_periodicity_l3055_305529

open Real

theorem function_periodicity 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h_a : a > 0) 
  (h_f : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  ∀ x : ℝ, f (x + 2 * a) = f x := by
sorry

end NUMINAMATH_CALUDE_function_periodicity_l3055_305529


namespace NUMINAMATH_CALUDE_apple_count_difference_l3055_305550

theorem apple_count_difference (initial_green : ℕ) (delivered_green : ℕ) (final_difference : ℕ) : 
  initial_green = 32 →
  delivered_green = 340 →
  initial_green + delivered_green = initial_green + final_difference + 140 →
  ∃ (initial_red : ℕ), initial_red - initial_green = 200 :=
by sorry

end NUMINAMATH_CALUDE_apple_count_difference_l3055_305550


namespace NUMINAMATH_CALUDE_sum_of_fractions_in_different_bases_l3055_305579

/-- Converts a number from a given base to base 10 --/
def toBase10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem sum_of_fractions_in_different_bases : 
  let a := toBase10 [2, 5, 4] 8
  let b := toBase10 [1, 2] 4
  let c := toBase10 [1, 3, 2] 5
  let d := toBase10 [2, 3] 3
  roundToNearest ((a / b : ℚ) + (c / d : ℚ)) = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_in_different_bases_l3055_305579


namespace NUMINAMATH_CALUDE_xiaoming_ticket_arrangements_l3055_305555

/-- The number of children with 1 yuan each -/
def num_friends : ℕ := 6

/-- The minimum number of friends that must be before Xiaoming -/
def min_friends_before : ℕ := 4

/-- The total number of children including Xiaoming -/
def total_children : ℕ := num_friends + 1

/-- The number of ways to arrange the children so Xiaoming can get change -/
def valid_arrangements : ℕ := Nat.choose num_friends min_friends_before * Nat.factorial num_friends

theorem xiaoming_ticket_arrangements : valid_arrangements = 10800 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_ticket_arrangements_l3055_305555


namespace NUMINAMATH_CALUDE_sqrt_twelve_is_quadratic_radical_l3055_305592

/-- Definition of a quadratic radical -/
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), y ≥ 0 ∧ x = Real.sqrt y

/-- Theorem stating that √12 is a quadratic radical -/
theorem sqrt_twelve_is_quadratic_radical :
  is_quadratic_radical (Real.sqrt 12) :=
by
  sorry


end NUMINAMATH_CALUDE_sqrt_twelve_is_quadratic_radical_l3055_305592


namespace NUMINAMATH_CALUDE_scientific_notation_of_3790000_l3055_305553

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Theorem stating that 3,790,000 in scientific notation is 3.79 × 10^6 -/
theorem scientific_notation_of_3790000 :
  toScientificNotation 3790000 = ScientificNotation.mk 3.79 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_3790000_l3055_305553


namespace NUMINAMATH_CALUDE_probability_is_half_l3055_305569

/-- A game board represented as a regular hexagon -/
structure HexagonalBoard :=
  (total_segments : ℕ)
  (shaded_segments : ℕ)
  (is_regular : total_segments = 6)
  (shaded_constraint : shaded_segments = 3)

/-- The probability of a spinner landing on a shaded region of a hexagonal board -/
def probability_shaded (board : HexagonalBoard) : ℚ :=
  board.shaded_segments / board.total_segments

/-- Theorem stating that the probability of landing on a shaded region is 1/2 -/
theorem probability_is_half (board : HexagonalBoard) :
  probability_shaded board = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l3055_305569


namespace NUMINAMATH_CALUDE_function_symmetry_l3055_305593

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x - b * Real.cos x

theorem function_symmetry 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (h_sym : ∀ x : ℝ, f a b (π/4 + x) = f a b (π/4 - x)) :
  let y := fun x => f a b (3*π/4 - x)
  (∀ x : ℝ, y (-x) = -y x) ∧ 
  (∀ x : ℝ, y (2*π - x) = y x) := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_l3055_305593


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l3055_305575

/-- A geometric sequence with positive terms where the 7th term is √2/2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ 
  (∀ n : ℕ, a n > 0) ∧
  (a 7 = Real.sqrt 2 / 2)

/-- The minimum value of 1/a_3 + 2/a_11 for the given geometric sequence is 4 -/
theorem geometric_sequence_min_value (a : ℕ → ℝ) (h : GeometricSequence a) :
  (1 / a 3 + 2 / a 11) ≥ 4 ∧ ∃ b : ℕ → ℝ, GeometricSequence b ∧ 1 / b 3 + 2 / b 11 = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l3055_305575


namespace NUMINAMATH_CALUDE_horse_cow_price_system_l3055_305573

/-- Represents the price of a horse in yuan -/
def horse_price : ℝ := sorry

/-- Represents the price of a cow in yuan -/
def cow_price : ℝ := sorry

/-- The system of equations correctly represents the given conditions about horse and cow prices -/
theorem horse_cow_price_system :
  (2 * horse_price + cow_price - 10000 = (1/2) * horse_price) ∧
  (10000 - (horse_price + 2 * cow_price) = (1/2) * cow_price) := by
  sorry

end NUMINAMATH_CALUDE_horse_cow_price_system_l3055_305573


namespace NUMINAMATH_CALUDE_twelve_customers_in_line_l3055_305539

/-- The number of customers in a restaurant line -/
def customers_in_line (people_behind_front : ℕ) : ℕ :=
  people_behind_front + 1

/-- Theorem: Given 11 people behind the front person, there are 12 customers in line -/
theorem twelve_customers_in_line :
  customers_in_line 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_customers_in_line_l3055_305539


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3055_305542

theorem smallest_solution_of_equation (x : ℝ) : 
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → 
  (x ≥ 4 - Real.sqrt 2 ∧ 
   (1 / ((4 - Real.sqrt 2) - 3) + 1 / ((4 - Real.sqrt 2) - 5) = 4 / ((4 - Real.sqrt 2) - 4))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3055_305542


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3055_305519

theorem fraction_to_decimal (h : 625 = 5^4) : 17 / 625 = 0.0272 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3055_305519


namespace NUMINAMATH_CALUDE_unique_number_with_special_properties_l3055_305577

/-- Returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Returns the product of digits of a natural number -/
def prod_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := sorry

theorem unique_number_with_special_properties : 
  ∃! x : ℕ, 
    prod_of_digits x = 44 * x - 86868 ∧ 
    is_perfect_cube (sum_of_digits x) ∧
    x = 1989 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_special_properties_l3055_305577


namespace NUMINAMATH_CALUDE_celine_change_l3055_305565

def laptop_price : ℕ := 600
def smartphone_price : ℕ := 400
def laptops_bought : ℕ := 2
def smartphones_bought : ℕ := 4
def total_money : ℕ := 3000

theorem celine_change : 
  total_money - (laptop_price * laptops_bought + smartphone_price * smartphones_bought) = 200 := by
  sorry

end NUMINAMATH_CALUDE_celine_change_l3055_305565


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l3055_305548

def is_valid_pair (square B : Nat) : Prop :=
  (square, B) ∈ [(0, 3), (2, 1), (4, 2), (6, 0), (8, 1)]

theorem six_digit_divisibility :
  ∀ (square B : Nat),
    square ≤ 9 ∧ B ≤ 9 →
    Even square →
    (532900 + square * 10 + B) % 6 = 0 ↔ is_valid_pair square B :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l3055_305548


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3055_305526

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = -3/2 ∧ 
  (∀ x : ℝ, 2*x*(x-5) = 3*(5-x) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3055_305526


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3055_305516

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : geometric_sequence a q)
  (h_condition : a 5 ^ 2 = 2 * a 3 * a 9) :
  q = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3055_305516


namespace NUMINAMATH_CALUDE_division_threefold_change_l3055_305535

theorem division_threefold_change (a b c d : ℤ) (h : a = b * c + d) :
  ∃ (d' : ℤ), (3 * a) = (3 * b) * c + d' ∧ d' = 3 * d :=
sorry

end NUMINAMATH_CALUDE_division_threefold_change_l3055_305535


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l3055_305552

theorem not_p_necessary_not_sufficient_for_not_q 
  (h1 : p → q) 
  (h2 : ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l3055_305552


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3055_305597

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- (a_1, a_3, a_4) forms a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, (a 3) / (a 1) = r ∧ (a 4) / (a 3) = r

/-- Theorem: If {a_n} is an arithmetic sequence with common difference 2
    and (a_1, a_3, a_4) forms a geometric sequence, then a_2 = -6 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) (h_geom : geometric_subseq a) : 
  a 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3055_305597


namespace NUMINAMATH_CALUDE_distance_point_to_line_l3055_305551

def point : ℝ × ℝ × ℝ := (0, 3, -1)
def linePoint1 : ℝ × ℝ × ℝ := (1, -2, 0)
def linePoint2 : ℝ × ℝ × ℝ := (3, 1, 4)

def distancePointToLine (p : ℝ × ℝ × ℝ) (l1 l2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_point_to_line :
  distancePointToLine point linePoint1 linePoint2 = Real.sqrt 22058 / 29 := by
  sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l3055_305551


namespace NUMINAMATH_CALUDE_smallest_n_for_unique_k_l3055_305586

theorem smallest_n_for_unique_k : ∃ (n : ℕ),
  n > 0 ∧
  (∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 10/19) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬∃! (k : ℤ), (9 : ℚ)/17 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 10/19) ∧
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_unique_k_l3055_305586


namespace NUMINAMATH_CALUDE_number_of_schnauzers_l3055_305511

theorem number_of_schnauzers : ℕ := by
  -- Define the number of Doberman puppies
  let doberman : ℕ := 20

  -- Define the equation from the problem
  let equation (s : ℕ) : Prop := 3 * doberman - 5 + (doberman - s) = 90

  -- Assert that the equation holds for s = 55
  have h : equation 55 := by sorry

  -- Prove that 55 is the unique solution
  have unique : ∀ s : ℕ, equation s → s = 55 := by sorry

  -- Conclude that the number of Schnauzers is 55
  exact 55

end NUMINAMATH_CALUDE_number_of_schnauzers_l3055_305511


namespace NUMINAMATH_CALUDE_parallelogram_area_l3055_305561

/-- The area of a parallelogram with a 150-degree angle and consecutive sides of 10 and 20 units --/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin θ = 100 * Real.sqrt 3 := by
  sorry

#check parallelogram_area

end NUMINAMATH_CALUDE_parallelogram_area_l3055_305561


namespace NUMINAMATH_CALUDE_moon_permutations_eq_twelve_l3055_305530

/-- The number of distinct permutations of the letters in "MOON" -/
def moon_permutations : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

theorem moon_permutations_eq_twelve :
  moon_permutations = 12 := by
  sorry

#eval moon_permutations

end NUMINAMATH_CALUDE_moon_permutations_eq_twelve_l3055_305530


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3055_305564

theorem meaningful_fraction (x : ℝ) :
  (2 * x - 1 ≠ 0) ↔ (x ≠ 1/2) := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3055_305564


namespace NUMINAMATH_CALUDE_square_rotation_around_hexagon_l3055_305574

theorem square_rotation_around_hexagon :
  let hexagon_angle : ℝ := 120
  let square_angle : ℝ := 90
  let rotation_per_movement : ℝ := 360 - (hexagon_angle + square_angle)
  let total_rotation : ℝ := 3 * rotation_per_movement
  total_rotation % 360 = 90 := by sorry

end NUMINAMATH_CALUDE_square_rotation_around_hexagon_l3055_305574


namespace NUMINAMATH_CALUDE_dave_remaining_candy_l3055_305528

/-- The number of chocolate candy boxes Dave bought -/
def total_boxes : ℕ := 12

/-- The number of boxes Dave gave to his little brother -/
def given_boxes : ℕ := 5

/-- The number of candy pieces in each box -/
def pieces_per_box : ℕ := 3

/-- The number of candy pieces Dave still has -/
def remaining_pieces : ℕ := (total_boxes - given_boxes) * pieces_per_box

theorem dave_remaining_candy : remaining_pieces = 21 := by
  sorry

end NUMINAMATH_CALUDE_dave_remaining_candy_l3055_305528


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3055_305534

theorem inequality_solution_set (x : ℝ) :
  (1 / x < 2 ∧ 1 / x > -3) ↔ (x > 1 / 2 ∨ x < -1 / 3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3055_305534


namespace NUMINAMATH_CALUDE_inequality_solution_l3055_305578

open Set

def solution_set : Set ℝ :=
  Ioo (-3 : ℝ) (-8/3) ∪ Ioo ((1 - Real.sqrt 89) / 4) ((1 + Real.sqrt 89) / 4)

theorem inequality_solution :
  {x : ℝ | (x - 2) / (x + 3) > (4 * x + 5) / (3 * x + 8) ∧ x ≠ -3 ∧ x ≠ -8/3} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3055_305578


namespace NUMINAMATH_CALUDE_large_block_volume_l3055_305570

/-- Volume of a rectangular block -/
def volume (width depth length : ℝ) : ℝ := width * depth * length

theorem large_block_volume :
  ∀ (w d l : ℝ),
  volume w d l = 4 →
  volume (2 * w) (2 * d) (2 * l) = 32 := by
  sorry

end NUMINAMATH_CALUDE_large_block_volume_l3055_305570


namespace NUMINAMATH_CALUDE_product_minus_third_lower_bound_l3055_305504

theorem product_minus_third_lower_bound 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (a : ℝ) 
  (h1 : x * y - z = a) 
  (h2 : y * z - x = a) 
  (h3 : z * x - y = a) : 
  a ≥ -1/4 := by
sorry

end NUMINAMATH_CALUDE_product_minus_third_lower_bound_l3055_305504


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l3055_305587

theorem square_sum_given_difference_and_product (a b : ℝ) 
  (h1 : a - b = 8) (h2 : a * b = 20) : a^2 + b^2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l3055_305587


namespace NUMINAMATH_CALUDE_smallest_possible_abs_z_l3055_305523

theorem smallest_possible_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 17) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 8) + Complex.abs (w - Complex.I * 7) = 17 ∧ Complex.abs w = 7 / Real.sqrt 113 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_abs_z_l3055_305523


namespace NUMINAMATH_CALUDE_no_intersection_in_S_l3055_305559

-- Define the set S of polynomials
inductive S : (ℝ → ℝ) → Prop
  | base : S (λ x => x)
  | mul {f} : S f → S (λ x => x * f x)
  | add {f} : S f → S (λ x => x + (1 - x) * f x)

-- Theorem statement
theorem no_intersection_in_S (f g : ℝ → ℝ) (hf : S f) (hg : S g) (h_neq : f ≠ g) :
  ∀ x, 0 < x → x < 1 → f x ≠ g x :=
sorry

end NUMINAMATH_CALUDE_no_intersection_in_S_l3055_305559


namespace NUMINAMATH_CALUDE_henry_workout_convergence_l3055_305558

theorem henry_workout_convergence (gym_distance : ℝ) (walk_fraction : ℝ) : 
  gym_distance = 3 →
  walk_fraction = 2/3 →
  ∃ (A B : ℝ), 
    (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
      |A - (gym_distance - walk_fraction^n * gym_distance)| < ε ∧
      |B - (walk_fraction * gym_distance - walk_fraction^n * (walk_fraction * gym_distance - gym_distance))| < ε) ∧
    |A - B| = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_henry_workout_convergence_l3055_305558


namespace NUMINAMATH_CALUDE_complex_product_real_l3055_305536

theorem complex_product_real (m : ℝ) : 
  (Complex.I : ℂ) * (1 - m * Complex.I) + (m^2 : ℂ) * (1 - m * Complex.I) ∈ Set.range Complex.ofReal → 
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_real_l3055_305536
