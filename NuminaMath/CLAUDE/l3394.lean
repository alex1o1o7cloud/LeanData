import Mathlib

namespace problem_1_l3394_339433

theorem problem_1 : -2.8 + (-3.6) + 3 - (-3.6) = 0.2 := by
  sorry

end problem_1_l3394_339433


namespace quadratic_equation_solution_l3394_339475

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 - x = 0} = {0, 1} := by sorry

end quadratic_equation_solution_l3394_339475


namespace set_intersection_problem_l3394_339470

theorem set_intersection_problem (M N : Set ℤ) : 
  M = {-1, 0, 1} → N = {0, 1, 2} → M ∩ N = {0, 1} := by
  sorry

end set_intersection_problem_l3394_339470


namespace always_odd_l3394_339432

theorem always_odd (p m : ℤ) (h : Odd p) : Odd (p^2 + 2*m*p) := by
  sorry

end always_odd_l3394_339432


namespace no_lines_satisfying_conditions_l3394_339455

-- Define the plane and points A and B
def Plane : Type := ℝ × ℝ
def A : Plane := sorry
def B : Plane := sorry

-- Define the distance between two points in the plane
def distance (p q : Plane) : ℝ := sorry

-- Define a line in the plane
def Line : Type := Plane → Prop

-- Define the distance from a point to a line
def point_to_line_distance (p : Plane) (l : Line) : ℝ := sorry

-- Define the angle between two lines
def angle_between_lines (l1 l2 : Line) : ℝ := sorry

-- Define the line y = x
def y_equals_x : Line := sorry

-- State the theorem
theorem no_lines_satisfying_conditions :
  ∀ (l : Line),
    distance A B = 8 →
    point_to_line_distance A l = 3 →
    point_to_line_distance B l = 4 →
    angle_between_lines l y_equals_x = π/4 →
    False :=
sorry

end no_lines_satisfying_conditions_l3394_339455


namespace tea_bags_count_l3394_339453

/-- Represents the number of tea bags in a box -/
def n : ℕ := sorry

/-- Represents the number of cups Natasha made -/
def natasha_cups : ℕ := 41

/-- Represents the number of cups Inna made -/
def inna_cups : ℕ := 58

/-- The number of cups made from Natasha's box is between 2n and 3n -/
axiom natasha_range : 2 * n ≤ natasha_cups ∧ natasha_cups ≤ 3 * n

/-- The number of cups made from Inna's box is between 2n and 3n -/
axiom inna_range : 2 * n ≤ inna_cups ∧ inna_cups ≤ 3 * n

/-- The number of tea bags in the box is 20 -/
theorem tea_bags_count : n = 20 := by sorry

end tea_bags_count_l3394_339453


namespace two_digit_number_interchange_l3394_339414

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 6 → 
  (10 * x + y) - (10 * y + x) = 54 := by
  sorry

end two_digit_number_interchange_l3394_339414


namespace andrey_gifts_l3394_339482

theorem andrey_gifts :
  ∃ (n a : ℕ), 
    n > 0 ∧ 
    a > 0 ∧ 
    n * (n - 2) = a * (n - 1) + 16 ∧ 
    n = 18 := by
  sorry

end andrey_gifts_l3394_339482


namespace polar_to_rectangular_coordinates_l3394_339485

theorem polar_to_rectangular_coordinates :
  let r : ℝ := 4
  let θ : ℝ := 5 * π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = -2 * Real.sqrt 3 ∧ y = 2 := by
  sorry

end polar_to_rectangular_coordinates_l3394_339485


namespace angle_between_vectors_l3394_339418

def vector_a : ℝ × ℝ := (3, -4)

theorem angle_between_vectors (b : ℝ × ℝ) 
  (h1 : ‖b‖ = 2) 
  (h2 : vector_a.fst * b.fst + vector_a.snd * b.snd = -5) : 
  Real.arccos ((vector_a.fst * b.fst + vector_a.snd * b.snd) / (‖vector_a‖ * ‖b‖)) = 2 * Real.pi / 3 := by
  sorry

end angle_between_vectors_l3394_339418


namespace grain_spilled_calculation_l3394_339452

/-- Calculates the amount of grain spilled into the water -/
def grain_spilled (original : ℕ) (remaining : ℕ) : ℕ :=
  original - remaining

/-- Theorem: The amount of grain spilled is the difference between original and remaining -/
theorem grain_spilled_calculation (original remaining : ℕ) 
  (h1 : original = 50870)
  (h2 : remaining = 918) :
  grain_spilled original remaining = 49952 := by
  sorry

#eval grain_spilled 50870 918

end grain_spilled_calculation_l3394_339452


namespace sector_perimeter_l3394_339466

/-- Given a circular sector with area 2 and central angle 4 radians, its perimeter is 6. -/
theorem sector_perimeter (r : ℝ) (h1 : r > 0) : 
  (1/2 * r * (4 * r) = 2) → (4 * r + 2 * r = 6) := by
  sorry

end sector_perimeter_l3394_339466


namespace initial_tomatoes_correct_l3394_339487

/-- Represents the initial number of tomatoes in the garden -/
def initial_tomatoes : ℕ := 175

/-- Represents the initial number of potatoes in the garden -/
def initial_potatoes : ℕ := 77

/-- Represents the number of potatoes picked -/
def picked_potatoes : ℕ := 172

/-- Represents the total number of tomatoes and potatoes left after picking -/
def remaining_total : ℕ := 80

/-- Theorem stating that the initial number of tomatoes is correct given the conditions -/
theorem initial_tomatoes_correct : 
  initial_tomatoes + initial_potatoes - picked_potatoes = remaining_total :=
by sorry


end initial_tomatoes_correct_l3394_339487


namespace line_intersects_x_axis_l3394_339492

/-- The line equation 4y - 3x = 16 intersects the x-axis at (-16/3, 0) -/
theorem line_intersects_x_axis :
  let line := λ x y : ℚ => 4 * y - 3 * x = 16
  let x_axis := λ x y : ℚ => y = 0
  let intersection_point := (-16/3, 0)
  line intersection_point.1 intersection_point.2 ∧ x_axis intersection_point.1 intersection_point.2 := by
  sorry


end line_intersects_x_axis_l3394_339492


namespace equal_division_theorem_l3394_339474

theorem equal_division_theorem (total : ℕ) (people : ℕ) (share : ℕ) : 
  total = 2400 → people = 4 → share * people = total → share = 600 := by
  sorry

end equal_division_theorem_l3394_339474


namespace complex_number_modulus_l3394_339458

theorem complex_number_modulus (z : ℂ) : z = (1 - Complex.I) / (1 + Complex.I) → Complex.abs z = 1 := by
  sorry

end complex_number_modulus_l3394_339458


namespace total_toys_l3394_339401

/-- The number of toys each person has -/
structure ToyCount where
  jaxon : ℕ
  gabriel : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def toy_conditions (t : ToyCount) : Prop :=
  t.jaxon = 15 ∧ 
  t.gabriel = 2 * t.jaxon ∧ 
  t.jerry = t.gabriel + 8

/-- The theorem stating the total number of toys -/
theorem total_toys (t : ToyCount) (h : toy_conditions t) : 
  t.jaxon + t.gabriel + t.jerry = 83 := by
  sorry

end total_toys_l3394_339401


namespace community_age_is_35_l3394_339488

/-- Represents the average age of a community given specific demographic information. -/
def community_average_age (women_ratio men_ratio : ℚ) (women_avg_age men_avg_age children_avg_age : ℚ) (children_ratio : ℚ) : ℚ :=
  let total_population := women_ratio + men_ratio + children_ratio * men_ratio
  let total_age := women_ratio * women_avg_age + men_ratio * men_avg_age + children_ratio * men_ratio * children_avg_age
  total_age / total_population

/-- Theorem stating that the average age of the community is 35 years given the specified conditions. -/
theorem community_age_is_35 :
  community_average_age 3 2 40 36 10 (1/3) = 35 := by
  sorry

end community_age_is_35_l3394_339488


namespace elective_schemes_count_l3394_339473

/-- The number of courses offered -/
def total_courses : ℕ := 10

/-- The number of courses that can't be chosen together -/
def conflicting_courses : ℕ := 3

/-- The number of courses each student must choose -/
def courses_to_choose : ℕ := 3

/-- The number of different elective schemes -/
def num_elective_schemes : ℕ := 98

theorem elective_schemes_count :
  (total_courses = 10) →
  (conflicting_courses = 3) →
  (courses_to_choose = 3) →
  (num_elective_schemes = Nat.choose (total_courses - conflicting_courses) courses_to_choose +
                          conflicting_courses * Nat.choose (total_courses - conflicting_courses) (courses_to_choose - 1)) :=
by sorry

end elective_schemes_count_l3394_339473


namespace geometric_sequence_product_l3394_339402

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 2 * a 3 = 5 →
  a 7 * a 8 * a 9 = 10 →
  a 4 * a 5 * a 6 = 10 / (a 1)^3 := by sorry

end geometric_sequence_product_l3394_339402


namespace cube_sum_reciprocal_l3394_339465

theorem cube_sum_reciprocal (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 = 970 := by
  sorry

end cube_sum_reciprocal_l3394_339465


namespace apps_deleted_minus_added_l3394_339422

theorem apps_deleted_minus_added (initial_apps added_apps final_apps : ℕ) : 
  initial_apps = 15 → added_apps = 71 → final_apps = 14 →
  (initial_apps + added_apps - final_apps) - added_apps = 1 := by
  sorry

end apps_deleted_minus_added_l3394_339422


namespace seven_n_implies_n_is_sum_of_squares_l3394_339448

theorem seven_n_implies_n_is_sum_of_squares (n : ℤ) (A B : ℤ) (h : 7 * n = A^2 + 3 * B^2) :
  ∃ (a b : ℤ), n = a^2 + 3 * b^2 := by
  sorry

end seven_n_implies_n_is_sum_of_squares_l3394_339448


namespace custom_operations_result_l3394_339449

def star (a b : ℤ) : ℤ := a + b - 1

def hash (a b : ℤ) : ℤ := a * b - 1

theorem custom_operations_result : (star (star 6 8) (hash 3 5)) = 26 := by
  sorry

end custom_operations_result_l3394_339449


namespace f_odd_and_decreasing_l3394_339484

def f (x : ℝ) := -x^3

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end f_odd_and_decreasing_l3394_339484


namespace dunkers_lineups_l3394_339423

/-- The number of players in the team -/
def total_players : ℕ := 15

/-- The number of players who can't play together -/
def special_players : ℕ := 3

/-- The number of players in a starting lineup -/
def lineup_size : ℕ := 5

/-- The number of possible starting lineups -/
def possible_lineups : ℕ := 2277

/-- Function to calculate binomial coefficient -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem dunkers_lineups :
  choose (total_players - special_players) lineup_size +
  special_players * choose (total_players - special_players) (lineup_size - 1) =
  possible_lineups :=
sorry

end dunkers_lineups_l3394_339423


namespace custom_mult_four_three_l3394_339412

/-- Custom multiplication operation -/
def customMult (x y : ℝ) : ℝ := x^2 - x*y + y^2

/-- Theorem stating that 4 * 3 = 13 under the custom multiplication -/
theorem custom_mult_four_three : customMult 4 3 = 13 := by sorry

end custom_mult_four_three_l3394_339412


namespace sarah_bottle_caps_l3394_339450

/-- Given that Sarah initially had 26 bottle caps and now has 29 in total,
    prove that she bought 3 bottle caps. -/
theorem sarah_bottle_caps (initial : ℕ) (total : ℕ) (bought : ℕ) 
    (h1 : initial = 26) 
    (h2 : total = 29) 
    (h3 : total = initial + bought) : bought = 3 := by
  sorry

end sarah_bottle_caps_l3394_339450


namespace parabola_and_line_intersection_l3394_339489

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line y = x + m -/
structure Line where
  m : ℝ

/-- The distance from a point to the y-axis -/
def distToAxis (pt : Point) : ℝ := |pt.x|

theorem parabola_and_line_intersection
  (para : Parabola)
  (A : Point)
  (l : Line)
  (h1 : A.y^2 = 2 * para.p * A.x) -- A is on the parabola
  (h2 : A.x = 2) -- x-coordinate of A is 2
  (h3 : distToAxis A = 4) -- distance from A to axis is 4
  (h4 : ∃ (P Q : Point), P ≠ Q ∧
        P.y^2 = 2 * para.p * P.x ∧ Q.y^2 = 2 * para.p * Q.x ∧
        P.y = P.x + l.m ∧ Q.y = Q.x + l.m) -- l intersects parabola at distinct P and Q
  (h5 : ∃ (P Q : Point), P ≠ Q ∧
        P.y^2 = 2 * para.p * P.x ∧ Q.y^2 = 2 * para.p * Q.x ∧
        P.y = P.x + l.m ∧ Q.y = Q.x + l.m ∧
        P.x * Q.x + P.y * Q.y = 0) -- OP ⊥ OQ
  : para.p = 4 ∧ l.m = -8 := by
  sorry

end parabola_and_line_intersection_l3394_339489


namespace negation_existence_quadratic_l3394_339445

theorem negation_existence_quadratic (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) := by
  sorry

end negation_existence_quadratic_l3394_339445


namespace workers_savings_l3394_339444

theorem workers_savings (monthly_pay : ℝ) (savings_fraction : ℝ) 
  (h1 : savings_fraction = 1 / 7)
  (h2 : savings_fraction > 0)
  (h3 : savings_fraction < 1) : 
  12 * (savings_fraction * monthly_pay) = 2 * ((1 - savings_fraction) * monthly_pay) := by
  sorry

end workers_savings_l3394_339444


namespace unique_c_for_complex_magnitude_l3394_339464

theorem unique_c_for_complex_magnitude : ∃! c : ℝ, Complex.abs (1 - 2 * c * Complex.I) = 1 := by
  sorry

end unique_c_for_complex_magnitude_l3394_339464


namespace abs_sum_leq_sum_abs_l3394_339495

theorem abs_sum_leq_sum_abs (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  |a| + |b| ≤ |a + b| := by
sorry

end abs_sum_leq_sum_abs_l3394_339495


namespace quadratic_inequality_solution_sets_l3394_339461

theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
    sorry

end quadratic_inequality_solution_sets_l3394_339461


namespace constant_term_zero_implies_a_equals_six_l3394_339441

theorem constant_term_zero_implies_a_equals_six (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, (a + 2) * x^2 + b * x + (a - 6) = 0) → a = 6 := by
  sorry

end constant_term_zero_implies_a_equals_six_l3394_339441


namespace hemisphere_surface_area_l3394_339429

/-- The surface area of a hemisphere given its base area -/
theorem hemisphere_surface_area (base_area : ℝ) (h : base_area = 3) :
  let r : ℝ := Real.sqrt (base_area / Real.pi)
  2 * Real.pi * r^2 + base_area = 9 := by
  sorry


end hemisphere_surface_area_l3394_339429


namespace second_smallest_coprime_to_210_l3394_339446

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem second_smallest_coprime_to_210 :
  ∃ (x : ℕ), x > 1 ∧ 
  is_relatively_prime x 210 ∧
  (∃ (y : ℕ), y > 1 ∧ y < x ∧ is_relatively_prime y 210) ∧
  (∀ (z : ℕ), z > 1 ∧ z < x ∧ is_relatively_prime z 210 → z = 11) ∧
  x = 13 := by
sorry

end second_smallest_coprime_to_210_l3394_339446


namespace pencil_count_is_830_l3394_339413

/-- The final number of pencils in the drawer after a series of additions and removals. -/
def final_pencil_count (initial : ℕ) (nancy_adds : ℕ) (steven_adds : ℕ) (maria_adds : ℕ) (kim_removes : ℕ) (george_removes : ℕ) : ℕ :=
  initial + nancy_adds + steven_adds + maria_adds - kim_removes - george_removes

/-- Theorem stating that the final number of pencils in the drawer is 830. -/
theorem pencil_count_is_830 :
  final_pencil_count 200 375 150 250 85 60 = 830 := by
  sorry

end pencil_count_is_830_l3394_339413


namespace sales_theorem_l3394_339486

def sales_problem (sales1 sales2 sales3 sales5 sales6 : ℕ) (average : ℕ) : Prop :=
  let total_sales := average * 6
  let known_sales := sales1 + sales2 + sales3 + sales5 + sales6
  let sales4 := total_sales - known_sales
  sales4 = 11707

theorem sales_theorem :
  sales_problem 5266 5768 5922 6029 4937 5600 :=
by
  sorry

end sales_theorem_l3394_339486


namespace initial_avg_equals_correct_avg_l3394_339442

-- Define the number of elements
def n : ℕ := 10

-- Define the correct average
def correct_avg : ℚ := 22

-- Define the difference between the correct and misread value
def misread_diff : ℤ := 10

-- Theorem statement
theorem initial_avg_equals_correct_avg :
  let correct_sum := n * correct_avg
  let initial_sum := correct_sum - misread_diff
  initial_sum / n = correct_avg := by
sorry

end initial_avg_equals_correct_avg_l3394_339442


namespace book_arrangement_theorem_l3394_339419

/-- The number of ways to arrange books on a shelf -/
def arrange_books (math_books : ℕ) (history_books : ℕ) (english_books : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial history_books) * (Nat.factorial english_books)

/-- Theorem: The number of ways to arrange 3 math books, 4 history books, and 5 English books
    on a shelf, where all books of the same subject must stay together and books within
    each subject are distinct, is equal to 103680. -/
theorem book_arrangement_theorem :
  arrange_books 3 4 5 = 103680 := by
  sorry

end book_arrangement_theorem_l3394_339419


namespace michelle_crayons_l3394_339479

/-- The number of crayons Michelle has -/
def total_crayons (num_boxes : ℕ) (crayons_per_box : ℕ) : ℕ :=
  num_boxes * crayons_per_box

/-- Proof that Michelle has 35 crayons -/
theorem michelle_crayons : total_crayons 7 5 = 35 := by
  sorry

end michelle_crayons_l3394_339479


namespace polar_to_rectangular_conversion_l3394_339439

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (2, 2 * Real.sqrt 3) :=
by sorry

end polar_to_rectangular_conversion_l3394_339439


namespace valentines_remaining_l3394_339462

theorem valentines_remaining (initial : ℕ) (children neighbors coworkers : ℕ) :
  initial ≥ children + neighbors + coworkers →
  initial - (children + neighbors + coworkers) =
  initial - children - neighbors - coworkers :=
by sorry

end valentines_remaining_l3394_339462


namespace circular_garden_radius_l3394_339498

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 6) * π * r^2 → r = 12 := by
  sorry

end circular_garden_radius_l3394_339498


namespace colonization_combinations_l3394_339497

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 8

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 7

/-- Represents the colonization effort required for an Earth-like planet -/
def earth_like_effort : ℕ := 2

/-- Represents the colonization effort required for a Mars-like planet -/
def mars_like_effort : ℕ := 1

/-- Represents the total available colonization effort -/
def total_effort : ℕ := 18

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Theorem stating the number of distinct combinations of planets that can be fully colonized -/
theorem colonization_combinations : 
  (choose earth_like_planets 8 * choose mars_like_planets 2) +
  (choose earth_like_planets 7 * choose mars_like_planets 4) +
  (choose earth_like_planets 6 * choose mars_like_planets 6) = 497 := by
  sorry

end colonization_combinations_l3394_339497


namespace largest_non_square_sum_diff_smallest_n_with_square_digit_sum_n_66_has_square_digit_sum_l3394_339430

def is_sum_of_squares (x : ℕ) : Prop := ∃ a b : ℕ, x = a^2 + b^2

def is_diff_of_squares (x : ℕ) : Prop := ∃ a b : ℕ, x = a^2 - b^2

def sum_of_digit_squares (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.map (λ d => d^2)).sum

def largest_n_digit_number (n : ℕ) : ℕ := 10^n - 1

theorem largest_non_square_sum_diff (n : ℕ) (h : n > 2) :
  ∀ k : ℕ, k ≤ largest_n_digit_number n →
    (¬ is_sum_of_squares k ∧ ¬ is_diff_of_squares k) →
    k ≤ 10^n - 2 :=
sorry

theorem smallest_n_with_square_digit_sum :
  ∀ n : ℕ, n < 66 → ¬ ∃ k : ℕ, sum_of_digit_squares n = k^2 :=
sorry

theorem n_66_has_square_digit_sum :
  ∃ k : ℕ, sum_of_digit_squares 66 = k^2 :=
sorry

end largest_non_square_sum_diff_smallest_n_with_square_digit_sum_n_66_has_square_digit_sum_l3394_339430


namespace jesses_room_difference_l3394_339404

/-- Jesse's room dimensions and length-width difference --/
theorem jesses_room_difference (length width : ℕ) (h1 : length = 20) (h2 : width = 19) :
  length - width = 1 := by
  sorry

end jesses_room_difference_l3394_339404


namespace complex_symmetry_division_l3394_339431

/-- Two complex numbers are symmetric about the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_about_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- The main theorem: If z₁ and z₂ are symmetric about the imaginary axis and z₁ = -1 + i, then z₁ / z₂ = i -/
theorem complex_symmetry_division (z₁ z₂ : ℂ) 
  (h_sym : symmetric_about_imaginary_axis z₁ z₂) 
  (h_z₁ : z₁ = -1 + Complex.I) : 
  z₁ / z₂ = Complex.I :=
sorry

end complex_symmetry_division_l3394_339431


namespace jellybeans_count_l3394_339481

/-- The number of jellybeans in a bag with black, green, and orange beans -/
def total_jellybeans (black green orange : ℕ) : ℕ := black + green + orange

/-- Theorem: Given the conditions, the total number of jellybeans is 27 -/
theorem jellybeans_count :
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  orange = green - 1 →
  total_jellybeans black green orange = 27 := by
sorry

end jellybeans_count_l3394_339481


namespace president_savings_l3394_339426

/-- Calculates the amount saved by the president for his reelection campaign --/
theorem president_savings (total_funds : ℝ) (friends_percentage : ℝ) (family_percentage : ℝ)
  (h1 : total_funds = 10000)
  (h2 : friends_percentage = 0.4)
  (h3 : family_percentage = 0.3) :
  total_funds - (friends_percentage * total_funds + family_percentage * (total_funds - friends_percentage * total_funds)) = 4200 :=
by sorry

end president_savings_l3394_339426


namespace davids_chemistry_marks_l3394_339408

/-- Given David's marks in four subjects and the average across five subjects, 
    prove that his marks in Chemistry must be 87. -/
theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 86) 
  (h2 : mathematics = 85) 
  (h3 : physics = 92) 
  (h4 : biology = 95) 
  (h5 : average = 89) 
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) : 
  chemistry = 87 := by
  sorry

end davids_chemistry_marks_l3394_339408


namespace money_distribution_theorem_l3394_339496

/-- Represents the money distribution problem --/
structure MoneyDistribution where
  total : ℚ
  first_share : ℚ
  second_share : ℚ
  third_share : ℚ

/-- Checks if the given distribution satisfies the initial conditions --/
def valid_initial_distribution (d : MoneyDistribution) : Prop :=
  d.first_share = d.total / 2 ∧
  d.second_share = d.total / 3 ∧
  d.third_share = d.total / 6

/-- Calculates the amount each person saves --/
def savings (d : MoneyDistribution) : (ℚ × ℚ × ℚ) :=
  (d.first_share / 2, d.second_share / 3, d.third_share / 6)

/-- Calculates the total amount saved --/
def total_savings (d : MoneyDistribution) : ℚ :=
  let (s1, s2, s3) := savings d
  s1 + s2 + s3

/-- Checks if the final distribution is equal for all three people --/
def equal_final_distribution (d : MoneyDistribution) : Prop :=
  let total_saved := total_savings d
  d.first_share + total_saved / 3 =
  d.second_share + total_saved / 3 ∧
  d.second_share + total_saved / 3 =
  d.third_share + total_saved / 3

/-- The main theorem stating the existence of a valid solution --/
theorem money_distribution_theorem :
  ∃ (d : MoneyDistribution),
    valid_initial_distribution d ∧
    equal_final_distribution d ∧
    d.first_share = 23.5 ∧
    d.second_share = 15 + 2/3 ∧
    d.third_share = 7 + 5/6 :=
sorry

end money_distribution_theorem_l3394_339496


namespace function_properties_l3394_339415

/-- Given functions f and g satisfying certain properties, prove specific characteristics -/
theorem function_properties (f g : ℝ → ℝ) 
  (h1 : ∀ x y, f (x - y) = f x * g y - g x * f y)
  (h2 : f (-2) = f 1)
  (h3 : f 1 ≠ 0) : 
  (g 0 = 1) ∧ 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ (x : ℝ) (k : ℤ), f x = f (x + 3 * ↑k)) := by
  sorry

end function_properties_l3394_339415


namespace not_blessed_2017_l3394_339416

def is_valid_date (month day : ℕ) : Prop :=
  1 ≤ month ∧ month ≤ 12 ∧ 1 ≤ day ∧ day ≤ 31

def concat_mmdd (month day : ℕ) : ℕ :=
  month * 100 + day

def is_blessed_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), is_valid_date month day ∧ concat_mmdd month day = year % 100

theorem not_blessed_2017 : ¬ is_blessed_year 2017 :=
sorry

end not_blessed_2017_l3394_339416


namespace complex_argument_bounds_l3394_339490

variable (b : ℝ) (hb : b ≠ 0)
variable (y : ℂ)

theorem complex_argument_bounds :
  (Complex.abs (b * y + y⁻¹) = Real.sqrt 2) →
  (Complex.arg y = π / 4 ∨ Complex.arg y = 7 * π / 4) ∧
  (∀ z : ℂ, Complex.abs (b * z + z⁻¹) = Real.sqrt 2 →
    π / 4 ≤ Complex.arg z ∧ Complex.arg z ≤ 7 * π / 4) :=
by sorry

end complex_argument_bounds_l3394_339490


namespace cloth_cost_price_l3394_339417

/-- Proves that the cost price of one meter of cloth is 85 rupees given the selling price and profit per meter. -/
theorem cloth_cost_price
  (selling_price : ℕ)
  (cloth_length : ℕ)
  (profit_per_meter : ℕ)
  (h1 : selling_price = 8500)
  (h2 : cloth_length = 85)
  (h3 : profit_per_meter = 15) :
  (selling_price - profit_per_meter * cloth_length) / cloth_length = 85 :=
by
  sorry

end cloth_cost_price_l3394_339417


namespace haley_trees_count_l3394_339443

/-- The number of trees that died after the typhoon -/
def dead_trees : ℕ := 5

/-- The number of trees left after the typhoon -/
def remaining_trees : ℕ := 12

/-- The total number of trees Haley grew -/
def total_trees : ℕ := dead_trees + remaining_trees

theorem haley_trees_count : total_trees = 17 := by
  sorry

end haley_trees_count_l3394_339443


namespace slope_of_line_l3394_339405

theorem slope_of_line (x y : ℝ) :
  3 * y = 4 * x - 9 → (∃ m b : ℝ, y = m * x + b ∧ m = 4/3) :=
by sorry

end slope_of_line_l3394_339405


namespace decagon_area_theorem_l3394_339407

/-- A rectangle with an inscribed decagon -/
structure DecagonInRectangle where
  perimeter : ℝ
  length_width_ratio : ℝ
  inscribed_decagon : Unit

/-- Calculate the area of the inscribed decagon -/
def area_of_inscribed_decagon (r : DecagonInRectangle) : ℝ :=
  sorry

/-- The theorem statement -/
theorem decagon_area_theorem (r : DecagonInRectangle) 
  (h_perimeter : r.perimeter = 160)
  (h_ratio : r.length_width_ratio = 3 / 2) :
  area_of_inscribed_decagon r = 1413.12 := by
  sorry

end decagon_area_theorem_l3394_339407


namespace no_positive_integer_solution_l3394_339456

theorem no_positive_integer_solution :
  ¬ ∃ (x : ℕ), (x > 0) ∧ ((5 * x + 1) / (x - 1) > 2 * x + 2) := by
  sorry

end no_positive_integer_solution_l3394_339456


namespace exam_score_calculation_l3394_339459

/-- Given an examination with the following conditions:
  - Total number of questions is 120
  - Each correct answer scores 3 marks
  - Each wrong answer loses 1 mark
  - The total score is 180 marks
  This theorem proves that the number of correctly answered questions is 75. -/
theorem exam_score_calculation (total_questions : ℕ) (correct_score wrong_score total_score : ℤ) 
  (h1 : total_questions = 120)
  (h2 : correct_score = 3)
  (h3 : wrong_score = -1)
  (h4 : total_score = 180) :
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_score + (total_questions - correct_answers) * wrong_score = total_score ∧ 
    correct_answers = 75 := by
  sorry

end exam_score_calculation_l3394_339459


namespace circle_in_rectangle_l3394_339499

theorem circle_in_rectangle (rectangle_side : Real) (circle_area : Real) : 
  rectangle_side = 14 →
  circle_area = 153.93804002589985 →
  (circle_area = π * (rectangle_side / 2)^2) →
  rectangle_side = 14 :=
by
  sorry

end circle_in_rectangle_l3394_339499


namespace physics_marks_l3394_339438

theorem physics_marks (P C M : ℝ) 
  (avg_all : (P + C + M) / 3 = 60)
  (avg_pm : (P + M) / 2 = 90)
  (avg_pc : (P + C) / 2 = 70) :
  P = 140 := by
sorry

end physics_marks_l3394_339438


namespace q_satisfies_conditions_l3394_339468

def q (x : ℝ) : ℝ := -x^3 + 2*x^2 + 3*x

theorem q_satisfies_conditions :
  (q 3 = 0) ∧ 
  (q (-1) = 0) ∧ 
  (∃ (a b c : ℝ), ∀ x, q x = a*x^3 + b*x^2 + c*x) ∧
  (q 4 = -20) := by
  sorry

end q_satisfies_conditions_l3394_339468


namespace cubic_roots_sum_cubes_l3394_339451

theorem cubic_roots_sum_cubes (p q r : ℝ) : 
  (3 * p^3 + 4 * p^2 - 200 * p + 5 = 0) →
  (3 * q^3 + 4 * q^2 - 200 * q + 5 = 0) →
  (3 * r^3 + 4 * r^2 - 200 * r + 5 = 0) →
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 24 :=
by sorry

end cubic_roots_sum_cubes_l3394_339451


namespace order_of_abc_l3394_339420

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_abc : a > c ∧ c > b := by sorry

end order_of_abc_l3394_339420


namespace percentage_calculation_l3394_339436

theorem percentage_calculation (first_number second_number : ℝ) 
  (h1 : first_number = 110)
  (h2 : second_number = 22) :
  (second_number / first_number) * 100 = 20 := by
  sorry

end percentage_calculation_l3394_339436


namespace last_three_digits_of_square_l3394_339400

theorem last_three_digits_of_square (n : ℕ) : ∃ n, n^2 % 1000 = 689 ∧ ¬∃ m, m^2 % 1000 = 759 := by
  sorry

end last_three_digits_of_square_l3394_339400


namespace incenter_position_l3394_339493

-- Define a triangle PQR
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

-- Define the side lengths
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (11, 5, 8)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Theorem stating the position of the incenter
theorem incenter_position (t : Triangle) :
  let (p, q, r) := side_lengths t
  let J := incenter t
  J = (11/24 * t.P.1 + 5/24 * t.Q.1 + 8/24 * t.R.1,
       11/24 * t.P.2 + 5/24 * t.Q.2 + 8/24 * t.R.2) :=
by sorry

end incenter_position_l3394_339493


namespace team_selection_count_l3394_339472

/-- The number of ways to select a team of 8 members with an equal number of boys and girls
    from a group of 10 boys and 12 girls. -/
def select_team (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) : ℕ :=
  Nat.choose total_boys (team_size / 2) * Nat.choose total_girls (team_size / 2)

/-- Theorem stating that the number of ways to select the team is 103950. -/
theorem team_selection_count :
  select_team 10 12 8 = 103950 := by
  sorry

end team_selection_count_l3394_339472


namespace value_of_a_minus_b_l3394_339471

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 3015 * a + 3021 * b = 3025)
  (eq2 : 3017 * a + 3023 * b = 3027) : 
  a - b = -7/3 := by
sorry

end value_of_a_minus_b_l3394_339471


namespace sine_inequalities_l3394_339435

theorem sine_inequalities :
  (∀ x : ℝ, |Real.sin (2 * x)| ≤ 2 * |Real.sin x|) ∧
  (∀ n : ℕ, n > 0 → ∀ x : ℝ, |Real.sin (n * x)| ≤ n * |Real.sin x|) := by
  sorry

end sine_inequalities_l3394_339435


namespace halloween_candy_count_l3394_339434

/-- The number of candy pieces Robin scored on Halloween -/
def initial_candy : ℕ := 23

/-- The number of candy pieces Robin ate -/
def eaten_candy : ℕ := 7

/-- The number of candy pieces Robin's sister gave her -/
def sister_candy : ℕ := 21

/-- The number of candy pieces Robin has now -/
def current_candy : ℕ := 37

/-- Theorem stating that the initial candy count is correct -/
theorem halloween_candy_count : 
  initial_candy - eaten_candy + sister_candy = current_candy := by
  sorry

end halloween_candy_count_l3394_339434


namespace perpendicular_line_through_point_l3394_339411

/-- Given a line L1 with equation x - 2y + 1 = 0, prove that the line L2 with equation 2x + y + 1 = 0
    passes through the point (-2, 3) and is perpendicular to L1. -/
theorem perpendicular_line_through_point :
  let L1 : ℝ → ℝ → Prop := λ x y => x - 2*y + 1 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => 2*x + y + 1 = 0
  let point : ℝ × ℝ := (-2, 3)
  (L2 point.1 point.2) ∧
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → L2 x1 y1 → L2 x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) *
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) =
    ((x2 - x1) * (y2 - y1) - (y2 - y1) * (x2 - x1)) *
    ((x2 - x1) * (y2 - y1) - (y2 - y1) * (x2 - x1))) :=
by sorry


end perpendicular_line_through_point_l3394_339411


namespace f_monotonicity_and_inequality_l3394_339409

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 2 * Real.exp 1 * Real.log x

theorem f_monotonicity_and_inequality :
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f x > f y) ∧
  (∀ x ∈ Set.Ioi 1, ∀ y ∈ Set.Ioi 1, x < y → f x < f y) ∧
  (∀ b ≤ Real.exp 1, ∀ x > 0, f x ≥ b * (x^2 - 2*x + 2)) :=
by sorry

end f_monotonicity_and_inequality_l3394_339409


namespace card_covers_at_least_twelve_squares_l3394_339447

/-- Represents a square card with a given side length -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard with squares of a given side length -/
structure Checkerboard where
  square_side_length : ℝ

/-- Calculates the maximum number of squares that can be covered by a card on a checkerboard -/
def max_squares_covered (card : Card) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating that a 1.5-inch square card can cover at least 12 one-inch squares on a checkerboard -/
theorem card_covers_at_least_twelve_squares :
  ∀ (card : Card) (board : Checkerboard),
    card.side_length = 1.5 ∧ board.square_side_length = 1 →
    max_squares_covered card board ≥ 12 :=
  sorry

end card_covers_at_least_twelve_squares_l3394_339447


namespace binary_calculation_theorem_l3394_339477

/-- Represents a binary number as a list of bits (least significant bit first) -/
def BinaryNum := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNum) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : ℕ) : BinaryNum :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : BinaryNum :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNum) : BinaryNum :=
  decimal_to_binary (binary_to_decimal a * binary_to_decimal b)

/-- Divides a binary number by 2^n (equivalent to right shift by n) -/
def binary_divide_by_power_of_two (b : BinaryNum) (n : ℕ) : BinaryNum :=
  decimal_to_binary (binary_to_decimal b / 2^n)

theorem binary_calculation_theorem :
  let a : BinaryNum := [false, true, false, true, false, false, true, true]  -- 11001010₂
  let b : BinaryNum := [false, true, false, true, true]                      -- 11010₂
  let divisor : BinaryNum := [false, false, true]                            -- 100₂
  binary_divide_by_power_of_two (binary_multiply a b) 2 =
  [false, false, true, false, true, true, true, true, false, false]          -- 1001110100₂
  := by sorry

end binary_calculation_theorem_l3394_339477


namespace sunny_lead_in_second_race_l3394_339403

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure RaceScenario where
  sunny : Runner
  windy : Runner
  race_distance : ℝ
  sunny_lead : ℝ

/-- Calculates the distance covered by a runner in a given time -/
def distance_covered (runner : Runner) (time : ℝ) : ℝ :=
  runner.speed * time

/-- Represents the conditions of the problem -/
def race_conditions : RaceScenario :=
  { sunny := { speed := 8 },
    windy := { speed := 7 },
    race_distance := 400,
    sunny_lead := 50 }

/-- Theorem statement -/
theorem sunny_lead_in_second_race :
  let first_race := race_conditions
  let second_race := { first_race with
    sunny := { speed := first_race.sunny.speed * 0.9 },
    sunny_lead := -50 }
  let sunny_time := (second_race.race_distance - second_race.sunny_lead) / second_race.sunny.speed
  let windy_distance := distance_covered second_race.windy sunny_time
  (second_race.race_distance - second_race.sunny_lead) - windy_distance = 12.5 := by
  sorry


end sunny_lead_in_second_race_l3394_339403


namespace proportion_equality_false_l3394_339427

theorem proportion_equality_false : 
  ¬(∀ (A B C : ℚ), (A / B = C / 4 ∧ A = 4) → B = C) :=
by sorry

end proportion_equality_false_l3394_339427


namespace geometric_sequence_ratio_l3394_339428

/-- A positive geometric sequence with specific sum conditions has a common ratio of 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- The sequence is positive
  (∃ q > 0, ∀ n, a (n + 1) = a n * q) →  -- The sequence is geometric with positive ratio
  a 3 + a 5 = 5 →  -- First condition
  a 5 + a 7 = 20 →  -- Second condition
  (∃ q > 0, ∀ n, a (n + 1) = a n * q ∧ q = 2) :=  -- The common ratio is 2
by sorry

end geometric_sequence_ratio_l3394_339428


namespace plywood_cut_squares_l3394_339424

/-- Represents the number of squares obtained from cutting a square plywood --/
def num_squares (side : ℕ) (cut_size1 cut_size2 : ℕ) (total_cut_length : ℕ) : ℕ :=
  sorry

/-- The theorem statement --/
theorem plywood_cut_squares :
  num_squares 50 10 20 280 = 16 :=
sorry

end plywood_cut_squares_l3394_339424


namespace polynomial_division_remainder_l3394_339476

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, 3 * X^5 + X^4 + 3 = (X - 2)^2 * q + (13 * X - 9) :=
sorry

end polynomial_division_remainder_l3394_339476


namespace anglets_in_sixth_circle_is_6000_l3394_339467

-- Define constants
def full_circle_degrees : ℕ := 360
def anglets_per_degree : ℕ := 100

-- Define the number of anglets in a sixth of a circle
def anglets_in_sixth_circle : ℕ := (full_circle_degrees / 6) * anglets_per_degree

-- Theorem statement
theorem anglets_in_sixth_circle_is_6000 : anglets_in_sixth_circle = 6000 := by
  sorry

end anglets_in_sixth_circle_is_6000_l3394_339467


namespace quadratic_polynomial_value_at_10_l3394_339440

-- Define a quadratic polynomial
def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the divisibility condition
def divisibility_condition (q : ℝ → ℝ) : Prop :=
  ∃ (p : ℝ → ℝ), ∀ (x : ℝ), q x^3 - 3*x = p x * (x - 2) * (x + 2) * (x - 5)

theorem quadratic_polynomial_value_at_10 
  (a b c : ℝ) 
  (h : divisibility_condition (quadratic_polynomial a b c)) :
  quadratic_polynomial a b c 10 = (96 * Real.rpow 15 (1/3) - 135 * Real.rpow 6 (1/3)) / 21 := by
  sorry


end quadratic_polynomial_value_at_10_l3394_339440


namespace square_of_105_l3394_339480

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by
  sorry

end square_of_105_l3394_339480


namespace weekly_pill_count_l3394_339454

/-- Calculates the total number of pills taken in a week given daily intake of different types of pills -/
theorem weekly_pill_count 
  (insulin_daily : ℕ) 
  (blood_pressure_daily : ℕ) 
  (anticonvulsant_multiplier : ℕ) :
  insulin_daily = 2 →
  blood_pressure_daily = 3 →
  anticonvulsant_multiplier = 2 →
  (insulin_daily + blood_pressure_daily + anticonvulsant_multiplier * blood_pressure_daily) * 7 = 77 := by
  sorry

#check weekly_pill_count

end weekly_pill_count_l3394_339454


namespace profit_reduction_theorem_l3394_339463

/-- Initial daily sales -/
def initial_sales : ℕ := 30

/-- Initial profit per unit in yuan -/
def initial_profit_per_unit : ℕ := 50

/-- Sales increase per yuan of price reduction -/
def sales_increase_rate : ℕ := 2

/-- Calculate daily profit based on price reduction -/
def daily_profit (price_reduction : ℝ) : ℝ :=
  (initial_profit_per_unit - price_reduction) * (initial_sales + sales_increase_rate * price_reduction)

/-- Price reduction needed for a specific daily profit -/
def price_reduction_for_profit (target_profit : ℝ) : ℝ :=
  20  -- This is the value we want to prove

/-- Price reduction for maximum profit -/
def price_reduction_for_max_profit : ℝ :=
  17.5  -- This is the value we want to prove

theorem profit_reduction_theorem :
  daily_profit (price_reduction_for_profit 2100) = 2100 ∧
  ∀ x, daily_profit x ≤ daily_profit price_reduction_for_max_profit :=
by sorry

end profit_reduction_theorem_l3394_339463


namespace percentage_of_360_is_180_l3394_339478

theorem percentage_of_360_is_180 : 
  let whole : ℝ := 360
  let part : ℝ := 180
  let percentage : ℝ := (part / whole) * 100
  percentage = 50 := by sorry

end percentage_of_360_is_180_l3394_339478


namespace x_yz_equals_12_l3394_339421

theorem x_yz_equals_12 (x y z : ℝ) (h : x * (x + y + z) = x^2 + 12) : x * (y + z) = 12 := by
  sorry

end x_yz_equals_12_l3394_339421


namespace survey_results_l3394_339437

/-- Represents the survey results for a subject -/
structure SubjectSurvey where
  yes : Nat
  no : Nat
  unsure : Nat

/-- The main theorem about the survey results -/
theorem survey_results 
  (total_students : Nat)
  (subject_m : SubjectSurvey)
  (subject_r : SubjectSurvey)
  (yes_only_m : Nat)
  (h1 : total_students = 800)
  (h2 : subject_m.yes = 500)
  (h3 : subject_m.no = 200)
  (h4 : subject_m.unsure = 100)
  (h5 : subject_r.yes = 400)
  (h6 : subject_r.no = 100)
  (h7 : subject_r.unsure = 300)
  (h8 : yes_only_m = 150)
  (h9 : subject_m.yes + subject_m.no + subject_m.unsure = total_students)
  (h10 : subject_r.yes + subject_r.no + subject_r.unsure = total_students) :
  total_students - (subject_m.yes + subject_r.yes - yes_only_m) = 400 := by
  sorry

end survey_results_l3394_339437


namespace max_x_given_lcm_l3394_339460

theorem max_x_given_lcm (x : ℕ) : 
  (Nat.lcm x (Nat.lcm 15 21) = 105) → x ≤ 105 :=
by sorry

end max_x_given_lcm_l3394_339460


namespace factor_bound_l3394_339494

/-- The number of ways to factor a positive integer into a product of integers greater than 1 -/
def f (k : ℕ) : ℕ := sorry

/-- Theorem: For any positive integer n > 1 and any prime factor p of n,
    the number of ways to factor n is less than or equal to n/p -/
theorem factor_bound {n p : ℕ} (h1 : n > 1) (h2 : p.Prime) (h3 : p ∣ n) : f n ≤ n / p := by
  sorry

end factor_bound_l3394_339494


namespace probability_of_selection_l3394_339491

/-- The number of students -/
def n : ℕ := 5

/-- The number of students to be chosen -/
def k : ℕ := 2

/-- Binomial coefficient -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The probability of selecting 2 students out of 5, where student A is selected and student B is not -/
theorem probability_of_selection : 
  (choose (n - 2) (k - 1) : ℚ) / (choose n k) = 3 / 10 := by
  sorry

end probability_of_selection_l3394_339491


namespace class_composition_after_adding_boys_l3394_339425

theorem class_composition_after_adding_boys (initial_boys initial_girls added_boys : ℕ) 
  (h1 : initial_boys = 11)
  (h2 : initial_girls = 13)
  (h3 : added_boys = 1) :
  (initial_girls : ℚ) / ((initial_boys + initial_girls + added_boys) : ℚ) = 52 / 100 := by
  sorry

end class_composition_after_adding_boys_l3394_339425


namespace individuals_from_c_is_twenty_l3394_339469

/-- Represents the ratio of individuals in strata A, B, and C -/
structure StrataRatio :=
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)

/-- Calculates the number of individuals to be drawn from stratum C -/
def individualsFromC (ratio : StrataRatio) (sampleSize : ℕ) : ℕ :=
  (ratio.c * sampleSize) / (ratio.a + ratio.b + ratio.c)

/-- Theorem: Given the specified ratio and sample size, 20 individuals should be drawn from C -/
theorem individuals_from_c_is_twenty :
  let ratio := StrataRatio.mk 5 3 2
  let sampleSize := 100
  individualsFromC ratio sampleSize = 20 := by
  sorry

end individuals_from_c_is_twenty_l3394_339469


namespace tan_150_degrees_l3394_339410

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end tan_150_degrees_l3394_339410


namespace opposite_face_of_B_l3394_339406

/-- Represents a square on the 3x3 grid --/
inductive Square
| A | B | C | D | E | F | G | H | I

/-- Represents the cube formed by folding the grid --/
structure Cube where
  open_face : Square
  opposite_pairs : List (Square × Square)

/-- Defines the folding of the 3x3 grid into a cube --/
def fold_grid (open_face : Square) : Cube :=
  { open_face := open_face,
    opposite_pairs := sorry }  -- The actual folding logic would go here

/-- The main theorem to prove --/
theorem opposite_face_of_B (c : Cube) :
  c.open_face = Square.F → (Square.B, Square.I) ∈ c.opposite_pairs :=
sorry

end opposite_face_of_B_l3394_339406


namespace square_areas_equality_l3394_339483

theorem square_areas_equality (a : ℝ) :
  let M := a^2 + (a+3)^2 + (a+5)^2 + (a+6)^2
  let N := (a+1)^2 + (a+2)^2 + (a+4)^2 + (a+7)^2
  M = N := by sorry

end square_areas_equality_l3394_339483


namespace square_sum_roots_l3394_339457

theorem square_sum_roots (a b c : ℝ) (h : a ≠ 0) : 
  let roots_sum := -b / a
  (x^2 + b*x + c = 0) → roots_sum^2 = 36 :=
by
  sorry

end square_sum_roots_l3394_339457
