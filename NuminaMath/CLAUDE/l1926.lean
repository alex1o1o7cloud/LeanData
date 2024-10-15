import Mathlib

namespace NUMINAMATH_CALUDE_geometric_ratio_in_arithmetic_sequence_l1926_192698

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- State the theorem
theorem geometric_ratio_in_arithmetic_sequence
  (a₁ d : ℝ) (h : d ≠ 0) :
  let a := arithmetic_sequence a₁ d
  (a 2) * (a 6) = (a 3)^2 →
  (a 3) / (a 2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_ratio_in_arithmetic_sequence_l1926_192698


namespace NUMINAMATH_CALUDE_square_plus_one_positive_l1926_192642

theorem square_plus_one_positive (a : ℚ) : 0 < a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_positive_l1926_192642


namespace NUMINAMATH_CALUDE_tv_show_total_watch_time_l1926_192651

theorem tv_show_total_watch_time :
  let regular_seasons : ℕ := 9
  let episodes_per_regular_season : ℕ := 22
  let extra_episodes_in_final_season : ℕ := 4
  let hours_per_episode : ℚ := 1/2

  let total_episodes : ℕ := 
    regular_seasons * episodes_per_regular_season + 
    (episodes_per_regular_season + extra_episodes_in_final_season)

  let total_watch_time : ℚ := total_episodes * hours_per_episode

  total_watch_time = 112 := by sorry

end NUMINAMATH_CALUDE_tv_show_total_watch_time_l1926_192651


namespace NUMINAMATH_CALUDE_union_of_sets_l1926_192622

open Set

theorem union_of_sets (M N : Set ℝ) : 
  M = {x : ℝ | 1 < x ∧ x ≤ 3} → 
  N = {x : ℝ | 2 < x ∧ x ≤ 5} → 
  M ∪ N = {x : ℝ | 1 < x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1926_192622


namespace NUMINAMATH_CALUDE_jack_first_half_time_l1926_192660

/-- Jack and Jill's hill race problem -/
theorem jack_first_half_time (jill_finish_time jack_second_half_time : ℕ)
  (h1 : jill_finish_time = 32)
  (h2 : jack_second_half_time = 6) :
  let jack_finish_time := jill_finish_time - 7
  jack_finish_time - jack_second_half_time = 19 := by
  sorry

end NUMINAMATH_CALUDE_jack_first_half_time_l1926_192660


namespace NUMINAMATH_CALUDE_triangle_inequality_l1926_192640

/-- 
Given a triangle ABC with circumradius R = 1 and area S = 1/4, 
prove that sqrt(a) + sqrt(b) + sqrt(c) < 1/a + 1/b + 1/c, 
where a, b, and c are the side lengths of the triangle.
-/
theorem triangle_inequality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_circumradius : (a * b * c) / (4 * (1/4)) = 1) 
  (h_area : (1/4) > 0) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1/a + 1/b + 1/c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1926_192640


namespace NUMINAMATH_CALUDE_sin_405_plus_cos_neg_270_l1926_192679

theorem sin_405_plus_cos_neg_270 : 
  Real.sin (405 * π / 180) + Real.cos (-270 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_405_plus_cos_neg_270_l1926_192679


namespace NUMINAMATH_CALUDE_value_of_a_l1926_192659

theorem value_of_a (x a : ℝ) : (x + 1) * (x - 3) = x^2 + a*x - 3 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1926_192659


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l1926_192686

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the centers of the circles
def center1 : ℝ × ℝ := (1, -2)
def center2 : ℝ × ℝ := (2, 0)

-- Define the equation of the line connecting the centers
def connecting_line (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Theorem statement
theorem perpendicular_bisector_equation :
  connecting_line (Prod.fst center1) (Prod.snd center1) ∧
  connecting_line (Prod.fst center2) (Prod.snd center2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l1926_192686


namespace NUMINAMATH_CALUDE_girls_in_class_l1926_192614

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (girls : ℕ) :
  total = 35 →
  ratio_girls = 3 →
  ratio_boys = 4 →
  girls * ratio_boys = (total - girls) * ratio_girls →
  girls = 15 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l1926_192614


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_40_l1926_192658

/-- Represents a triangle divided into four triangles and one quadrilateral -/
structure DividedTriangle where
  total_area : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  triangle3_area : ℝ
  triangle4_area : ℝ
  quadrilateral_area : ℝ

/-- The sum of all areas equals the total area of the triangle -/
def area_sum (dt : DividedTriangle) : Prop :=
  dt.total_area = dt.triangle1_area + dt.triangle2_area + dt.triangle3_area + dt.triangle4_area + dt.quadrilateral_area

/-- The theorem stating that given the areas of the four triangles, the area of the quadrilateral is 40 -/
theorem quadrilateral_area_is_40 (dt : DividedTriangle) 
  (h1 : dt.triangle1_area = 5)
  (h2 : dt.triangle2_area = 10)
  (h3 : dt.triangle3_area = 10)
  (h4 : dt.triangle4_area = 15)
  (h_sum : area_sum dt) : 
  dt.quadrilateral_area = 40 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_40_l1926_192658


namespace NUMINAMATH_CALUDE_find_number_l1926_192631

theorem find_number : ∃ x : ℝ, x - (3/5) * x = 56 ∧ x = 140 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1926_192631


namespace NUMINAMATH_CALUDE_broadway_show_attendance_l1926_192618

/-- The number of children attending a Broadway show -/
def num_children : ℕ := 200

/-- The number of adults attending the Broadway show -/
def num_adults : ℕ := 400

/-- The price of a child's ticket in dollars -/
def child_ticket_price : ℕ := 16

/-- The price of an adult ticket in dollars -/
def adult_ticket_price : ℕ := 32

/-- The total amount collected from ticket sales in dollars -/
def total_amount : ℕ := 16000

theorem broadway_show_attendance :
  num_children = 200 ∧
  num_adults = 400 ∧
  adult_ticket_price = 2 * child_ticket_price ∧
  adult_ticket_price = 32 ∧
  total_amount = num_adults * adult_ticket_price + num_children * child_ticket_price :=
by sorry

end NUMINAMATH_CALUDE_broadway_show_attendance_l1926_192618


namespace NUMINAMATH_CALUDE_square_side_length_l1926_192678

theorem square_side_length (s : ℝ) (h : s > 0) : s^2 = 6 * (4 * s) → s = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1926_192678


namespace NUMINAMATH_CALUDE_angle_c_measure_l1926_192688

theorem angle_c_measure (A B C : ℝ) (h : A + B = 90) : C = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_c_measure_l1926_192688


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l1926_192639

theorem abs_inequality_equivalence (x : ℝ) : 
  |((x + 4) / 2)| < 3 ↔ -10 < x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l1926_192639


namespace NUMINAMATH_CALUDE_existence_of_equal_elements_l1926_192611

theorem existence_of_equal_elements (n p q : ℕ) (x : ℕ → ℤ)
  (h_pos : 0 < n ∧ 0 < p ∧ 0 < q)
  (h_n_gt : n > p + q)
  (h_x_bounds : x 0 = 0 ∧ x n = 0)
  (h_x_diff : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (x i - x (i-1) = p ∨ x i - x (i-1) = -q)) :
  ∃ (i j : ℕ), i < j ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
by sorry

end NUMINAMATH_CALUDE_existence_of_equal_elements_l1926_192611


namespace NUMINAMATH_CALUDE_fraction_sum_proof_l1926_192699

theorem fraction_sum_proof : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_proof_l1926_192699


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1926_192604

-- Define the asymptotes
def asymptote1 (x : ℝ) : ℝ := 2 * x + 3
def asymptote2 (x : ℝ) : ℝ := -2 * x + 1

-- Define the point the hyperbola passes through
def point : ℝ × ℝ := (5, 7)

-- Define the hyperbola (implicitly)
def is_on_hyperbola (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    ((y - 2)^2 / a^2) - ((x + 1/2)^2 / b^2) = 1

-- Theorem statement
theorem hyperbola_foci_distance :
  ∃ (f1 f2 : ℝ × ℝ),
    is_on_hyperbola point.1 point.2 ∧
    ‖f1 - f2‖ = 15 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1926_192604


namespace NUMINAMATH_CALUDE_parallel_line_to_plane_transitive_parallel_planes_skew_lines_parallel_planes_l1926_192644

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)

-- Define the lines and planes
variable (l m n : Line)
variable (α β γ : Plane)

-- Axioms
axiom different_lines : m ≠ n
axiom different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Theorem A
theorem parallel_line_to_plane 
  (h1 : parallel_lines l m) 
  (h2 : line_in_plane m α) 
  (h3 : ¬ line_in_plane l α) : 
  parallel_line_plane l α :=
sorry

-- Theorem C
theorem transitive_parallel_planes 
  (h1 : parallel_planes α β) 
  (h2 : parallel_planes β γ) : 
  parallel_planes α γ :=
sorry

-- Theorem D
theorem skew_lines_parallel_planes 
  (h1 : skew_lines l m)
  (h2 : parallel_line_plane l α)
  (h3 : parallel_line_plane m α)
  (h4 : parallel_line_plane l β)
  (h5 : parallel_line_plane m β) :
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_parallel_line_to_plane_transitive_parallel_planes_skew_lines_parallel_planes_l1926_192644


namespace NUMINAMATH_CALUDE_smallest_square_partition_l1926_192690

theorem smallest_square_partition : ∃ (n : ℕ),
  (n > 0) ∧ 
  (∃ (a b c : ℕ), 
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
    (a + b + c = 12) ∧ 
    (a ≥ 9) ∧
    (n^2 = a * 1^2 + b * 2^2 + c * 3^2)) ∧
  (∀ (m : ℕ), m < n → 
    ¬(∃ (a b c : ℕ),
      (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
      (a + b + c = 12) ∧
      (a ≥ 9) ∧
      (m^2 = a * 1^2 + b * 2^2 + c * 3^2))) ∧
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_partition_l1926_192690


namespace NUMINAMATH_CALUDE_rabbit_beaver_time_difference_rabbit_beaver_time_difference_holds_l1926_192650

/-- The time difference between a rabbit digging a hole and a beaver building a dam -/
theorem rabbit_beaver_time_difference : ℝ → Prop :=
  fun time_difference =>
    ∀ (rabbit_count rabbit_time hole_count : ℝ)
      (beaver_count beaver_time dam_count : ℝ),
    rabbit_count > 0 →
    rabbit_time > 0 →
    hole_count > 0 →
    beaver_count > 0 →
    beaver_time > 0 →
    dam_count > 0 →
    rabbit_count * rabbit_time * 60 / hole_count = 100 →
    beaver_count * beaver_time / dam_count = 90 →
    rabbit_count = 3 →
    rabbit_time = 5 →
    hole_count = 9 →
    beaver_count = 5 →
    beaver_time = 36 / 60 →
    dam_count = 2 →
    time_difference = 10

theorem rabbit_beaver_time_difference_holds : rabbit_beaver_time_difference 10 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_beaver_time_difference_rabbit_beaver_time_difference_holds_l1926_192650


namespace NUMINAMATH_CALUDE_fair_spending_l1926_192697

def money_at_arrival : ℕ := 87
def money_at_departure : ℕ := 16

theorem fair_spending : money_at_arrival - money_at_departure = 71 := by
  sorry

end NUMINAMATH_CALUDE_fair_spending_l1926_192697


namespace NUMINAMATH_CALUDE_rhombus_area_l1926_192600

/-- The area of a rhombus with side length 20 and one diagonal of length 16 is 64√21 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) : 
  side = 20 → diagonal1 = 16 → diagonal2 = 8 * Real.sqrt 21 →
  (1/2) * diagonal1 * diagonal2 = 64 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1926_192600


namespace NUMINAMATH_CALUDE_operation_result_l1926_192627

def universal_set : Set ℝ := Set.univ

def operation (M N : Set ℝ) : Set ℝ := M ∩ (universal_set \ N)

def set_M : Set ℝ := {x : ℝ | |x| ≤ 2}

def set_N : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}

theorem operation_result :
  operation set_M set_N = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_operation_result_l1926_192627


namespace NUMINAMATH_CALUDE_modulo_equivalence_unique_l1926_192666

theorem modulo_equivalence_unique : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 15725 [MOD 16] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_unique_l1926_192666


namespace NUMINAMATH_CALUDE_dividend_calculation_l1926_192672

theorem dividend_calculation (divisor quotient remainder : ℕ) (h1 : divisor = 36) (h2 : quotient = 20) (h3 : remainder = 5) :
  divisor * quotient + remainder = 725 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1926_192672


namespace NUMINAMATH_CALUDE_sally_pens_home_l1926_192695

/-- The number of pens Sally takes home given the initial conditions -/
def pens_taken_home (total_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) : ℕ :=
  let pens_given := num_students * pens_per_student
  let pens_remaining := total_pens - pens_given
  pens_remaining / 2

theorem sally_pens_home :
  pens_taken_home 342 44 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sally_pens_home_l1926_192695


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l1926_192610

-- Define the propositions P and q as functions of a
def P (a : ℝ) : Prop := a ≤ -1 ∨ a ≥ 2

def q (a : ℝ) : Prop := a > 3

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 3)

-- Theorem statement
theorem range_of_a_theorem :
  ∀ a : ℝ, (¬(P a ∧ q a) ∧ (P a ∨ q a)) → range_of_a a :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l1926_192610


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_sixth_l1926_192665

theorem sqrt_fifth_power_sixth : (Real.sqrt ((Real.sqrt 5)^4))^6 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_sixth_l1926_192665


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_l1926_192669

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def T (n : ℕ) : ℕ := (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def S (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_number (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), is_four_digit x ∧ T x = p^k ∧ S x = p^p - 5 ∧
  ∀ (y : ℕ), is_four_digit y ∧ T y = p^k ∧ S y = p^p - 5 → x ≤ y :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_l1926_192669


namespace NUMINAMATH_CALUDE_no_integer_solution_l1926_192652

theorem no_integer_solution : ∀ x y : ℤ, (x + 7) * (x + 6) ≠ 8 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1926_192652


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1926_192634

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 4 * p - 7 = 0) → 
  (3 * q^2 + 4 * q - 7 = 0) → 
  (p - 2) * (q - 2) = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1926_192634


namespace NUMINAMATH_CALUDE_total_stripes_eq_22_l1926_192609

/-- The number of stripes on one of Olga's shoes -/
def olga_stripes_per_shoe : ℕ := 3

/-- The number of stripes on one of Rick's shoes -/
def rick_stripes_per_shoe : ℕ := olga_stripes_per_shoe - 1

/-- The number of stripes on one of Hortense's shoes -/
def hortense_stripes_per_shoe : ℕ := olga_stripes_per_shoe * 2

/-- The number of shoes each person has -/
def shoes_per_person : ℕ := 2

/-- The total number of stripes on all pairs of tennis shoes -/
def total_stripes : ℕ := 
  (olga_stripes_per_shoe * shoes_per_person) + 
  (rick_stripes_per_shoe * shoes_per_person) + 
  (hortense_stripes_per_shoe * shoes_per_person)

theorem total_stripes_eq_22 : total_stripes = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_stripes_eq_22_l1926_192609


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l1926_192648

def ends_in_6 (n : ℕ) : Prop := n % 10 = 6

def move_6_to_front (n : ℕ) : ℕ :=
  let d := (Nat.log 10 n) + 1
  6 * (10 ^ d) + (n / 10)

theorem smallest_number_with_properties :
  ∃ (n : ℕ),
    ends_in_6 n ∧
    move_6_to_front n = 4 * n ∧
    ∀ (m : ℕ), (ends_in_6 m ∧ move_6_to_front m = 4 * m) → n ≤ m ∧
    n = 153846 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_properties_l1926_192648


namespace NUMINAMATH_CALUDE_john_bought_490_packs_l1926_192676

/-- The number of packs John buys for each student -/
def packsPerStudent : ℕ := 4

/-- The number of extra packs John purchases for supplies -/
def extraPacks : ℕ := 10

/-- The number of students in each class -/
def studentsPerClass : List ℕ := [24, 18, 30, 20, 28]

/-- The total number of packs John bought -/
def totalPacks : ℕ := 
  (studentsPerClass.map (· * packsPerStudent)).sum + extraPacks

theorem john_bought_490_packs : totalPacks = 490 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_490_packs_l1926_192676


namespace NUMINAMATH_CALUDE_base5_132_to_base10_l1926_192670

/-- Converts a base-5 digit to its base-10 equivalent --/
def base5ToBase10Digit (d : Nat) : Nat :=
  if d < 5 then d else 0

/-- Converts a 3-digit base-5 number to its base-10 equivalent --/
def base5ToBase10 (d2 d1 d0 : Nat) : Nat :=
  (base5ToBase10Digit d2) * 25 + (base5ToBase10Digit d1) * 5 + (base5ToBase10Digit d0)

/-- Theorem stating that the base-10 representation of the base-5 number 132 is 42 --/
theorem base5_132_to_base10 : base5ToBase10 1 3 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_base5_132_to_base10_l1926_192670


namespace NUMINAMATH_CALUDE_election_winner_margin_l1926_192633

theorem election_winner_margin (total_votes : ℕ) (winner_votes : ℕ) :
  (winner_votes : ℝ) = 0.56 * total_votes →
  winner_votes = 1344 →
  winner_votes - (total_votes - winner_votes) = 288 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_margin_l1926_192633


namespace NUMINAMATH_CALUDE_cloth_sale_theorem_l1926_192626

/-- Represents the sale of cloth with given parameters. -/
structure ClothSale where
  totalSellingPrice : ℕ
  lossPerMetre : ℕ
  costPricePerMetre : ℕ

/-- Calculates the number of metres of cloth sold. -/
def metresSold (sale : ClothSale) : ℕ :=
  sale.totalSellingPrice / (sale.costPricePerMetre - sale.lossPerMetre)

/-- Theorem stating that for the given parameters, 600 metres of cloth were sold. -/
theorem cloth_sale_theorem (sale : ClothSale) 
  (h1 : sale.totalSellingPrice = 18000)
  (h2 : sale.lossPerMetre = 5)
  (h3 : sale.costPricePerMetre = 35) :
  metresSold sale = 600 := by
  sorry

#eval metresSold { totalSellingPrice := 18000, lossPerMetre := 5, costPricePerMetre := 35 }

end NUMINAMATH_CALUDE_cloth_sale_theorem_l1926_192626


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1926_192623

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 - 4*x^3 + 6*x^3 
  = 4*x^3 - x^2 + 23*x - 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1926_192623


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_144_l1926_192681

theorem greatest_prime_factor_of_144 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 144 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 144 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_144_l1926_192681


namespace NUMINAMATH_CALUDE_bag_balls_count_l1926_192661

theorem bag_balls_count (red_balls : ℕ) (white_balls : ℕ) (prob_red : ℚ) : 
  red_balls = 4 → 
  prob_red = 1/4 → 
  prob_red = red_balls / (red_balls + white_balls) →
  white_balls = 12 := by
sorry

end NUMINAMATH_CALUDE_bag_balls_count_l1926_192661


namespace NUMINAMATH_CALUDE_derivative_of_f_at_1_l1926_192657

def f (x : ℝ) := x^2

theorem derivative_of_f_at_1 : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_at_1_l1926_192657


namespace NUMINAMATH_CALUDE_final_amount_is_correct_l1926_192625

-- Define the quantities and prices of fruits
def grapes_quantity : ℝ := 15
def grapes_price : ℝ := 98
def mangoes_quantity : ℝ := 8
def mangoes_price : ℝ := 120
def pineapples_quantity : ℝ := 5
def pineapples_price : ℝ := 75
def oranges_quantity : ℝ := 10
def oranges_price : ℝ := 60

-- Define the discount and tax rates
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.08

-- Define the function to calculate the final amount
def calculate_final_amount : ℝ :=
  let total_cost := grapes_quantity * grapes_price + 
                    mangoes_quantity * mangoes_price + 
                    pineapples_quantity * pineapples_price + 
                    oranges_quantity * oranges_price
  let discounted_total := total_cost * (1 - discount_rate)
  let final_amount := discounted_total * (1 + tax_rate)
  final_amount

-- Theorem statement
theorem final_amount_is_correct : 
  calculate_final_amount = 3309.66 := by sorry

end NUMINAMATH_CALUDE_final_amount_is_correct_l1926_192625


namespace NUMINAMATH_CALUDE_sum_distances_focus_to_points_l1926_192632

/-- The parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (2, 0)

/-- Theorem: Sum of distances from focus to three points on parabola -/
theorem sum_distances_focus_to_points
  (A B C : ℝ × ℝ)
  (hA : A ∈ Parabola)
  (hB : B ∈ Parabola)
  (hC : C ∈ Parabola)
  (h_sum : F.1 * 3 = A.1 + B.1 + C.1) :
  dist F A + dist F B + dist F C = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_distances_focus_to_points_l1926_192632


namespace NUMINAMATH_CALUDE_gcd_of_repeated_five_digit_integers_l1926_192616

theorem gcd_of_repeated_five_digit_integers : 
  ∃ (g : ℕ), 
    (∀ (n : ℕ), 10000 ≤ n ∧ n < 100000 → g ∣ (n * 10000100001)) ∧
    (∀ (d : ℕ), (∀ (n : ℕ), 10000 ≤ n ∧ n < 100000 → d ∣ (n * 10000100001)) → d ∣ g) ∧
    g = 10000100001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_repeated_five_digit_integers_l1926_192616


namespace NUMINAMATH_CALUDE_custom_op_example_l1926_192683

/-- Custom binary operation ※ -/
def custom_op (a b : ℕ) : ℕ := a + 5 + b * 15

/-- Theorem stating that 105 ※ 5 = 185 -/
theorem custom_op_example : custom_op 105 5 = 185 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l1926_192683


namespace NUMINAMATH_CALUDE_halfway_between_one_fourth_and_one_seventh_l1926_192613

theorem halfway_between_one_fourth_and_one_seventh :
  let x : ℚ := 11 / 56
  (x - 1 / 4 : ℚ) = (1 / 7 - x : ℚ) ∧ 
  x = (1 / 4 + 1 / 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_fourth_and_one_seventh_l1926_192613


namespace NUMINAMATH_CALUDE_cold_drink_pitcher_l1926_192629

/-- Represents the recipe for a cold drink -/
structure Recipe where
  iced_tea : Rat
  lemonade : Rat

/-- Calculates the total amount of drink for a given recipe -/
def total_drink (r : Recipe) : Rat :=
  r.iced_tea + r.lemonade

/-- Represents the contents of a pitcher -/
structure Pitcher where
  lemonade : Rat
  total : Rat

/-- The theorem to be proved -/
theorem cold_drink_pitcher (r : Recipe) (p : Pitcher) :
  r.iced_tea = 1/4 →
  r.lemonade = 5/4 →
  p.lemonade = 15 →
  p.total = 18 :=
by sorry

end NUMINAMATH_CALUDE_cold_drink_pitcher_l1926_192629


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1926_192664

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 10) 
  (eq2 : x + 3 * y = 14) : 
  10 * x^2 + 12 * x * y + 10 * y^2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1926_192664


namespace NUMINAMATH_CALUDE_jerrys_shelf_l1926_192617

/-- The number of books on Jerry's shelf -/
def num_books : ℕ := 3

/-- The number of action figures added later -/
def added_figures : ℕ := 2

/-- The difference between action figures and books after adding -/
def difference : ℕ := 3

/-- The initial number of action figures on Jerry's shelf -/
def initial_figures : ℕ := 4

theorem jerrys_shelf :
  initial_figures + added_figures = num_books + difference := by sorry

end NUMINAMATH_CALUDE_jerrys_shelf_l1926_192617


namespace NUMINAMATH_CALUDE_problem_solution_l1926_192689

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 119) : x = 39 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1926_192689


namespace NUMINAMATH_CALUDE_prime_cube_plus_five_prime_l1926_192603

theorem prime_cube_plus_five_prime (p : ℕ) 
  (hp : Nat.Prime p) 
  (hp_cube : Nat.Prime (p^3 + 5)) : 
  p^5 - 7 = 25 := by
sorry

end NUMINAMATH_CALUDE_prime_cube_plus_five_prime_l1926_192603


namespace NUMINAMATH_CALUDE_greatest_valid_integer_l1926_192635

def is_valid (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 18 = 6

theorem greatest_valid_integer : 
  (∀ m, is_valid m → m ≤ 174) ∧ is_valid 174 := by sorry

end NUMINAMATH_CALUDE_greatest_valid_integer_l1926_192635


namespace NUMINAMATH_CALUDE_percentage_increase_men_is_twenty_percent_l1926_192607

/-- Represents the population data and conditions --/
structure PopulationData where
  men_1990 : ℕ
  women_1990 : ℕ
  boys_1990 : ℕ
  total_1994 : ℕ
  boys_1994 : ℕ

/-- Calculates the percentage increase in men given the population data --/
def percentageIncreaseMen (data : PopulationData) : ℚ :=
  let women_1994 := data.women_1990 + data.boys_1990 * data.women_1990 / data.women_1990
  let men_1994 := data.total_1994 - women_1994 - data.boys_1994
  (men_1994 - data.men_1990) * 100 / data.men_1990

/-- Theorem stating that the percentage increase in men is 20% --/
theorem percentage_increase_men_is_twenty_percent (data : PopulationData) 
  (h1 : data.men_1990 = 5000)
  (h2 : data.women_1990 = 3000)
  (h3 : data.boys_1990 = 2000)
  (h4 : data.total_1994 = 13000)
  (h5 : data.boys_1994 = data.boys_1990) :
  percentageIncreaseMen data = 20 := by
  sorry

#eval percentageIncreaseMen {
  men_1990 := 5000,
  women_1990 := 3000,
  boys_1990 := 2000,
  total_1994 := 13000,
  boys_1994 := 2000
}

end NUMINAMATH_CALUDE_percentage_increase_men_is_twenty_percent_l1926_192607


namespace NUMINAMATH_CALUDE_power_plus_one_div_square_int_l1926_192608

theorem power_plus_one_div_square_int (n : ℕ) : n > 1 →
  (∃ k : ℤ, (2^n + 1 : ℤ) = k * n^2) ↔ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_plus_one_div_square_int_l1926_192608


namespace NUMINAMATH_CALUDE_function_decreasing_implies_a_range_l1926_192647

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_function_decreasing_implies_a_range_l1926_192647


namespace NUMINAMATH_CALUDE_ben_savings_problem_l1926_192668

/-- Ben's daily starting amount -/
def daily_start : ℕ := 50

/-- Ben's daily spending -/
def daily_spend : ℕ := 15

/-- Ben's daily savings -/
def daily_savings : ℕ := daily_start - daily_spend

/-- Ben's final amount after mom's doubling and dad's addition -/
def final_amount : ℕ := 500

/-- Additional amount from dad -/
def dad_addition : ℕ := 10

/-- The number of days elapsed -/
def days_elapsed : ℕ := 7

theorem ben_savings_problem :
  final_amount = 2 * (daily_savings * days_elapsed) + dad_addition := by
  sorry

end NUMINAMATH_CALUDE_ben_savings_problem_l1926_192668


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1926_192637

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x ≤ 3}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {x | x < -1 ∨ (2 < x ∧ x ≤ 3)} :=
sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1926_192637


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l1926_192655

theorem vector_subtraction_and_scalar_multiplication :
  let v₁ : Fin 2 → ℝ := ![3, -8]
  let v₂ : Fin 2 → ℝ := ![2, -6]
  v₁ - 5 • v₂ = ![-7, 22] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l1926_192655


namespace NUMINAMATH_CALUDE_john_phone_cost_l1926_192615

theorem john_phone_cost (alan_price : ℝ) (john_percentage : ℝ) : 
  alan_price = 2000 → john_percentage = 0.02 → 
  alan_price * (1 + john_percentage) = 2040 := by
  sorry

end NUMINAMATH_CALUDE_john_phone_cost_l1926_192615


namespace NUMINAMATH_CALUDE_quadratic_monotone_decreasing_condition_l1926_192671

/-- A quadratic function f(x) = x^2 + ax + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

/-- The function is monotonically decreasing in the interval (-∞, 3) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 3 → y < 3 → f a x > f a y

/-- The theorem states that if f is monotonically decreasing in (-∞, 3), then a ∈ (-∞, -6] -/
theorem quadratic_monotone_decreasing_condition (a : ℝ) :
  is_monotone_decreasing a → a ≤ -6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotone_decreasing_condition_l1926_192671


namespace NUMINAMATH_CALUDE_arithmetic_mean_arrangement_l1926_192620

theorem arithmetic_mean_arrangement (n : ℕ+) :
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ (i j k : Fin n), i < k ∧ k < j →
      (p i + p j : ℚ) / 2 ≠ p k := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_arrangement_l1926_192620


namespace NUMINAMATH_CALUDE_five_power_sum_squares_l1926_192696

/-- A function that checks if a number is expressible as a sum of two squares -/
def is_sum_of_two_squares (x : ℕ) : Prop :=
  ∃ a b : ℕ, x = a^2 + b^2

/-- A function that checks if two numbers have the same parity -/
def same_parity (n m : ℕ) : Prop :=
  n % 2 = m % 2

theorem five_power_sum_squares (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  is_sum_of_two_squares (5^m + 5^n) ↔ same_parity n m :=
sorry

end NUMINAMATH_CALUDE_five_power_sum_squares_l1926_192696


namespace NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_complementary_l1926_192667

/-- A bag containing red and black balls -/
structure Bag where
  red : Nat
  black : Nat

/-- The outcome of drawing two balls -/
inductive DrawResult
  | TwoRed
  | OneRedOneBlack
  | TwoBlack

/-- Event representing exactly one black ball drawn -/
def ExactlyOneBlack (result : DrawResult) : Prop :=
  result = DrawResult.OneRedOneBlack

/-- Event representing exactly two black balls drawn -/
def ExactlyTwoBlack (result : DrawResult) : Prop :=
  result = DrawResult.TwoBlack

/-- The sample space of all possible outcomes -/
def SampleSpace (bag : Bag) : Set DrawResult :=
  {DrawResult.TwoRed, DrawResult.OneRedOneBlack, DrawResult.TwoBlack}

/-- Two events are mutually exclusive if their intersection is empty -/
def MutuallyExclusive (E₁ E₂ : Set DrawResult) : Prop :=
  E₁ ∩ E₂ = ∅

/-- Two events are complementary if their union is the entire sample space -/
def Complementary (E₁ E₂ : Set DrawResult) (S : Set DrawResult) : Prop :=
  E₁ ∪ E₂ = S

/-- Main theorem: ExactlyOneBlack and ExactlyTwoBlack are mutually exclusive but not complementary -/
theorem exactly_one_two_black_mutually_exclusive_not_complementary (bag : Bag) :
  let S := SampleSpace bag
  let E₁ := {r : DrawResult | ExactlyOneBlack r}
  let E₂ := {r : DrawResult | ExactlyTwoBlack r}
  MutuallyExclusive E₁ E₂ ∧ ¬Complementary E₁ E₂ S :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_complementary_l1926_192667


namespace NUMINAMATH_CALUDE_max_distance_to_origin_is_three_l1926_192601

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- Calculates the maximum distance from any point on a circle to the origin in polar coordinates -/
def maxDistanceToOrigin (c : PolarCircle) : ℝ :=
  c.center.r + c.radius

theorem max_distance_to_origin_is_three :
  let circle := PolarCircle.mk (PolarPoint.mk 2 (π / 6)) 1
  maxDistanceToOrigin circle = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_to_origin_is_three_l1926_192601


namespace NUMINAMATH_CALUDE_students_per_group_l1926_192682

theorem students_per_group 
  (total_students : ℕ) 
  (unpicked_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 65) 
  (h2 : unpicked_students = 17) 
  (h3 : num_groups = 8) : 
  (total_students - unpicked_students) / num_groups = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_per_group_l1926_192682


namespace NUMINAMATH_CALUDE_expression_simplification_l1926_192643

theorem expression_simplification (a c x y : ℝ) (h : c*x^2 + c*y^2 ≠ 0) :
  (c*x^2*(a^2*x^3 + 3*a^2*y^3 + c^2*y^3) + c*y^2*(a^2*x^3 + 3*c^2*x^3 + c^2*y^3)) / (c*x^2 + c*y^2)
  = a^2*x^3 + 3*c*x^3 + c^2*y^3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1926_192643


namespace NUMINAMATH_CALUDE_polygon_properties_l1926_192693

/-- A polygon with interior angle sum of 1080 degrees has 8 sides and exterior angle sum of 360 degrees -/
theorem polygon_properties (n : ℕ) (interior_sum : ℝ) (h : interior_sum = 1080) :
  (n - 2) * 180 = interior_sum ∧ n = 8 ∧ 360 = (n : ℝ) * (360 / n) := by
  sorry

end NUMINAMATH_CALUDE_polygon_properties_l1926_192693


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l1926_192602

theorem smallest_angle_solution (θ : Real) : 
  (θ > 0) → 
  (∀ φ, φ > 0 → φ < θ → Real.sin (10 * Real.pi / 180) ≠ Real.cos (40 * Real.pi / 180) - Real.cos (φ * Real.pi / 180)) →
  Real.sin (10 * Real.pi / 180) = Real.cos (40 * Real.pi / 180) - Real.cos (θ * Real.pi / 180) →
  θ = 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l1926_192602


namespace NUMINAMATH_CALUDE_f_properties_l1926_192673

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 0 else 1 - 1/x

theorem f_properties :
  (∀ x, x ≥ 0 → f x ≥ 0) ∧
  (f 1 = 0) ∧
  (∀ x, x > 1 → f x > 0) ∧
  (∀ x y, x ≥ 0 → y ≥ 0 → x + y > 0 → f (x * f y) * f y = f (x * y / (x + y))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1926_192673


namespace NUMINAMATH_CALUDE_banana_arrangements_l1926_192641

/-- The number of unique arrangements of letters in a word -/
def uniqueArrangements (totalLetters : Nat) (repetitions : List Nat) : Nat :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- The word "BANANA" has 6 letters -/
def totalLetters : Nat := 6

/-- The repetitions of letters in "BANANA": 3 A's, 2 N's, and 1 B (which we don't need to include) -/
def letterRepetitions : List Nat := [3, 2]

/-- Theorem: The number of unique arrangements of letters in "BANANA" is 60 -/
theorem banana_arrangements :
  uniqueArrangements totalLetters letterRepetitions = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l1926_192641


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l1926_192624

/-- Represents a participant's score in a single day of the competition -/
structure DailyScore where
  scored : ℕ
  attempted : ℕ
  success_ratio : scored ≤ attempted

/-- Represents a participant's total score across two days -/
structure TotalScore where
  day1 : DailyScore
  day2 : DailyScore
  total_attempted : day1.attempted + day2.attempted = 500

/-- Gamma's scores for the two days -/
def gamma : TotalScore := {
  day1 := { scored := 180, attempted := 280, success_ratio := by sorry },
  day2 := { scored := 120, attempted := 220, success_ratio := by sorry },
  total_attempted := by sorry
}

/-- Delta's scores for the two days -/
structure DeltaScore extends TotalScore where
  day1_ratio_less : (day1.scored : ℚ) / day1.attempted < (gamma.day1.scored : ℚ) / gamma.day1.attempted
  day2_ratio_less : (day2.scored : ℚ) / day2.attempted < (gamma.day2.scored : ℚ) / gamma.day2.attempted

theorem delta_max_success_ratio :
  ∀ delta : DeltaScore,
    (delta.day1.scored + delta.day2.scored : ℚ) / 500 ≤ 409 / 500 := by sorry

end NUMINAMATH_CALUDE_delta_max_success_ratio_l1926_192624


namespace NUMINAMATH_CALUDE_rectangle_perimeter_bound_l1926_192692

/-- Given a unit square covered by m^2 rectangles, there exists a rectangle with perimeter at least 4/m -/
theorem rectangle_perimeter_bound (m : ℝ) (h_m : m > 0) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b ≤ 1 / m^2 ∧ 2 * (a + b) ≥ 4 / m := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_bound_l1926_192692


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1926_192662

-- Problem 1
theorem problem_1 : 
  (2 + Real.sqrt 3) ^ 0 + 3 * Real.tan (30 * π / 180) - |Real.sqrt 3 - 2| + (1/2)⁻¹ = 1 + 2 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) (ha : a^2 - 4*a + 3 = 0) (hne : a*(a+3)*(a-3) ≠ 0) : 
  (a^2 - 9) / (a^2 - 3*a) / ((a^2 + 9) / a + 6) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1926_192662


namespace NUMINAMATH_CALUDE_gem_stone_necklaces_sold_l1926_192691

/-- Proves that the number of gem stone necklaces sold is 3 -/
theorem gem_stone_necklaces_sold (bead_necklaces : ℕ) (price_per_necklace : ℕ) (total_earnings : ℕ) :
  bead_necklaces = 4 →
  price_per_necklace = 3 →
  total_earnings = 21 →
  total_earnings = price_per_necklace * (bead_necklaces + 3) :=
by sorry

end NUMINAMATH_CALUDE_gem_stone_necklaces_sold_l1926_192691


namespace NUMINAMATH_CALUDE_savings_account_growth_l1926_192621

/-- Calculates the final amount in a savings account with compound interest. -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The problem statement -/
theorem savings_account_growth :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let years : ℕ := 21
  let final_amount := compound_interest principal rate years
  ∃ ε > 0, |final_amount - 3046.28| < ε :=
by sorry

end NUMINAMATH_CALUDE_savings_account_growth_l1926_192621


namespace NUMINAMATH_CALUDE_train_crossing_time_l1926_192654

/-- Calculates the time for a train to cross a signal pole given its length, 
    the length of a platform it crosses, and the time it takes to cross the platform. -/
theorem train_crossing_time (train_length platform_length : ℝ) (platform_crossing_time : ℝ) 
  (h1 : train_length = 150)
  (h2 : platform_length = 175)
  (h3 : platform_crossing_time = 39) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1926_192654


namespace NUMINAMATH_CALUDE_square_land_area_l1926_192656

/-- The area of a square land plot with side length 32 units is 1024 square units. -/
theorem square_land_area (side_length : ℝ) (h : side_length = 32) : 
  side_length * side_length = 1024 := by
  sorry

end NUMINAMATH_CALUDE_square_land_area_l1926_192656


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1926_192646

def ellipse_equation (x y : ℝ) : Prop := x^2 + y^2/2 = 1

def hyperbola_vertices (h_vertices : ℝ × ℝ) (e_vertices : ℝ × ℝ) : Prop :=
  h_vertices = e_vertices

def eccentricity_product (e_hyperbola e_ellipse : ℝ) : Prop :=
  e_hyperbola * e_ellipse = 1

theorem hyperbola_equation 
  (h_vertices : ℝ × ℝ) 
  (e_vertices : ℝ × ℝ) 
  (e_hyperbola e_ellipse : ℝ) :
  hyperbola_vertices h_vertices e_vertices →
  eccentricity_product e_hyperbola e_ellipse →
  ∃ (x y : ℝ), y^2 - x^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1926_192646


namespace NUMINAMATH_CALUDE_sin_22_5_deg_identity_l1926_192645

theorem sin_22_5_deg_identity : 1 - 2 * (Real.sin (22.5 * π / 180))^2 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_22_5_deg_identity_l1926_192645


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l1926_192630

/-- An ellipse with eccentricity √3/2 and maximum triangle area of 1 -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_max_area : a * b = 2

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : SpecialEllipse) where
  k : ℝ
  m : ℝ
  h_slope_product : (5 : ℝ) / 4 = k^2 + m^2

/-- The theorem statement -/
theorem special_ellipse_properties (E : SpecialEllipse) (L : IntersectingLine E) :
  (E.a = 2 ∧ E.b = 1) ∧
  (∃ (S : ℝ), S = 1 ∧ ∀ (k m : ℝ), (5 : ℝ) / 4 = k^2 + m^2 → S ≥ 
    ((5 - 4*k^2) * (20*k^2 - 1)) / (2 * (4*k^2 + 1))) :=
  sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l1926_192630


namespace NUMINAMATH_CALUDE_prime_power_sum_l1926_192612

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 840 →
  2*w + 3*x + 5*y + 7*z = 21 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1926_192612


namespace NUMINAMATH_CALUDE_smallest_gcd_ef_l1926_192628

theorem smallest_gcd_ef (d e f : ℕ+) (h1 : Nat.gcd d e = 210) (h2 : Nat.gcd d f = 770) :
  ∃ (e' f' : ℕ+), Nat.gcd d e' = 210 ∧ Nat.gcd d f' = 770 ∧ 
  Nat.gcd e' f' = 10 ∧ ∀ (e'' f'' : ℕ+), Nat.gcd d e'' = 210 → Nat.gcd d f'' = 770 → 
  Nat.gcd e'' f'' ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_ef_l1926_192628


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l1926_192605

theorem fraction_zero_implies_x_negative_two (x : ℝ) : 
  (x ≠ 2) → ((|x| - 2) / (x - 2) = 0) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l1926_192605


namespace NUMINAMATH_CALUDE_adams_purchase_cost_l1926_192677

/-- The cost of Adam's purchases of nuts and dried fruits -/
theorem adams_purchase_cost :
  let nuts_quantity : ℝ := 3
  let dried_fruits_quantity : ℝ := 2.5
  let nuts_price_per_kg : ℝ := 12
  let dried_fruits_price_per_kg : ℝ := 8
  let total_cost : ℝ := nuts_quantity * nuts_price_per_kg + dried_fruits_quantity * dried_fruits_price_per_kg
  total_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_adams_purchase_cost_l1926_192677


namespace NUMINAMATH_CALUDE_sixth_term_is_half_l1926_192694

/-- Geometric sequence with first term 16 and common ratio 1/2 -/
def geometricSequence : ℕ → ℚ
  | 0 => 16
  | n + 1 => (geometricSequence n) / 2

/-- The sixth term of the geometric sequence is 1/2 -/
theorem sixth_term_is_half : geometricSequence 5 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_half_l1926_192694


namespace NUMINAMATH_CALUDE_pencils_left_ashtons_pencils_l1926_192675

/-- Given two boxes of pencils with fourteen pencils each, after giving away six pencils, the number of pencils left is 22. -/
theorem pencils_left (boxes : Nat) (pencils_per_box : Nat) (pencils_given_away : Nat) : Nat :=
  boxes * pencils_per_box - pencils_given_away

/-- Ashton's pencil problem -/
theorem ashtons_pencils : pencils_left 2 14 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_pencils_left_ashtons_pencils_l1926_192675


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_l1926_192680

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_remainder (n : ℕ) (h : n ≥ 100) :
  sum_factorials n % 30 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 30 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_l1926_192680


namespace NUMINAMATH_CALUDE_peter_drew_age_difference_l1926_192663

/-- Proves that Peter is 4 years older than Drew given the conditions in the problem --/
theorem peter_drew_age_difference : 
  ∀ (maya drew peter john jacob : ℕ),
  drew = maya + 5 →
  peter > drew →
  john = 30 →
  john = 2 * maya →
  jacob + 2 = (peter + 2) / 2 →
  jacob = 11 →
  peter - drew = 4 := by
  sorry

end NUMINAMATH_CALUDE_peter_drew_age_difference_l1926_192663


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l1926_192684

theorem last_two_digits_sum (n : ℕ) : (8^25 + 12^25) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l1926_192684


namespace NUMINAMATH_CALUDE_previous_height_l1926_192653

theorem previous_height (current_height : ℝ) (growth_rate : ℝ) : 
  current_height = 126 ∧ growth_rate = 0.05 → 
  current_height / (1 + growth_rate) = 120 := by
  sorry

end NUMINAMATH_CALUDE_previous_height_l1926_192653


namespace NUMINAMATH_CALUDE_field_trip_probability_l1926_192674

/-- The number of vehicles available for the field trip -/
def num_vehicles : ℕ := 3

/-- The number of students in the specific group we're considering -/
def group_size : ℕ := 4

/-- The probability that all students in the group ride in the same vehicle -/
def same_vehicle_probability : ℚ := 1 / 27

theorem field_trip_probability :
  (num_vehicles : ℚ) / (num_vehicles ^ group_size) = same_vehicle_probability :=
sorry

end NUMINAMATH_CALUDE_field_trip_probability_l1926_192674


namespace NUMINAMATH_CALUDE_monthly_expenses_calculation_l1926_192649

-- Define the monthly deposit
def monthly_deposit : ℕ := 5000

-- Define the annual savings
def annual_savings : ℕ := 4800

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Theorem to prove
theorem monthly_expenses_calculation :
  (monthly_deposit * months_in_year - annual_savings) / months_in_year = 4600 :=
by sorry

end NUMINAMATH_CALUDE_monthly_expenses_calculation_l1926_192649


namespace NUMINAMATH_CALUDE_product_of_fractions_l1926_192619

theorem product_of_fractions : (1/2 : ℚ) * (3/5 : ℚ) * (5/6 : ℚ) = (1/4 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1926_192619


namespace NUMINAMATH_CALUDE_sugar_amount_proof_l1926_192636

/-- The price of a kilogram of sugar in dollars -/
def sugar_price : ℝ := 1.50

/-- The number of kilograms of sugar bought -/
def sugar_bought : ℝ := 2

/-- The price of a kilogram of salt in dollars -/
def salt_price : ℝ := (5 - 3 * sugar_price)

theorem sugar_amount_proof :
  sugar_bought * sugar_price + 5 * salt_price = 5.50 ∧
  3 * sugar_price + salt_price = 5 →
  sugar_bought = 2 :=
by sorry

end NUMINAMATH_CALUDE_sugar_amount_proof_l1926_192636


namespace NUMINAMATH_CALUDE_reese_savings_problem_l1926_192606

/-- Represents the percentage of savings spent in March -/
def march_spending_percentage : ℝ → Prop := λ M =>
  let initial_savings : ℝ := 11000
  let february_spending : ℝ := 0.2 * initial_savings
  let march_spending : ℝ := M * initial_savings
  let april_spending : ℝ := 1500
  let remaining : ℝ := 2900
  initial_savings - february_spending - march_spending - april_spending = remaining ∧
  M = 0.4

theorem reese_savings_problem :
  ∃ M : ℝ, march_spending_percentage M :=
sorry

end NUMINAMATH_CALUDE_reese_savings_problem_l1926_192606


namespace NUMINAMATH_CALUDE_circle_relations_l1926_192685

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Given three circles P, Q, R with radii p, q, r respectively, where p > q > r,
    and distances between centers d_PQ, d_PR, d_QR, prove that the following
    statements can all be true simultaneously:
    1. p + q can be equal to d_PQ
    2. q + r can be equal to d_QR
    3. p + r can be less than d_PR
    4. p - q can be less than d_PQ -/
theorem circle_relations (P Q R : Circle) 
    (h_p_gt_q : P.radius > Q.radius)
    (h_q_gt_r : Q.radius > R.radius)
    (d_PQ : ℝ) (d_PR : ℝ) (d_QR : ℝ) :
    ∃ (p q r : ℝ),
      p = P.radius ∧ q = Q.radius ∧ r = R.radius ∧
      (p + q = d_PQ ∨ p + q ≠ d_PQ) ∧
      (q + r = d_QR ∨ q + r ≠ d_QR) ∧
      (p + r < d_PR ∨ p + r ≥ d_PR) ∧
      (p - q < d_PQ ∨ p - q ≥ d_PQ) :=
by sorry

end NUMINAMATH_CALUDE_circle_relations_l1926_192685


namespace NUMINAMATH_CALUDE_range_of_a_in_linear_inequality_l1926_192638

/-- The range of values for parameter 'a' in the inequality 2x - y + a > 0,
    given that only one point among (0,0) and (1,1) is inside the region. -/
theorem range_of_a_in_linear_inequality :
  ∃ (a : ℝ), (∀ x y : ℝ, 2*x - y + a > 0 →
    ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) ∧
  (-1 < a ∧ a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_in_linear_inequality_l1926_192638


namespace NUMINAMATH_CALUDE_log_equation_sum_of_squares_l1926_192687

theorem log_equation_sum_of_squares (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 27 = 9 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^2 + y^2 = 189 := by
sorry

end NUMINAMATH_CALUDE_log_equation_sum_of_squares_l1926_192687
