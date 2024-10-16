import Mathlib

namespace NUMINAMATH_CALUDE_no_intersection_l1056_105610

theorem no_intersection : ¬ ∃ x : ℝ, |3 * x + 6| = -|2 * x - 4| := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l1056_105610


namespace NUMINAMATH_CALUDE_remainder_problem_l1056_105609

theorem remainder_problem (P D Q R Q' R' : ℕ) 
  (h1 : P = Q * D + 2 * R)
  (h2 : Q = 2 * D * Q' + R') :
  P % (2 * D^2) = D * R' + 2 * R := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1056_105609


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1056_105656

def U : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0}

def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2}

def B : Set ℤ := {2, 3, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1056_105656


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1056_105633

theorem inequality_solution_set (a b : ℝ) (h : |a - b| > 2) : ∀ x : ℝ, |x - a| + |x - b| > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1056_105633


namespace NUMINAMATH_CALUDE_smallest_sector_angle_l1056_105614

/-- Represents the properties of the circle division problem -/
structure CircleDivision where
  n : ℕ  -- number of sectors
  a₁ : ℕ  -- first term of the arithmetic sequence
  d : ℕ   -- common difference of the arithmetic sequence
  sum : ℕ -- sum of all angles

/-- The circle division satisfies the problem conditions -/
def validCircleDivision (cd : CircleDivision) : Prop :=
  cd.n = 15 ∧
  cd.sum = 360 ∧
  ∀ i : ℕ, i > 0 ∧ i ≤ cd.n → (cd.a₁ + (i - 1) * cd.d) > 0

/-- The theorem stating the smallest possible sector angle -/
theorem smallest_sector_angle (cd : CircleDivision) :
  validCircleDivision cd →
  (∃ cd' : CircleDivision, validCircleDivision cd' ∧ cd'.a₁ < cd.a₁) ∨ cd.a₁ = 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_sector_angle_l1056_105614


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l1056_105642

/-- A point in a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Triangle ABC on a grid -/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : GridPoint) : ℚ :=
  let x1 := p1.x
  let y1 := p1.y
  let x2 := p2.x
  let y2 := p2.y
  let x3 := p3.x
  let y3 := p3.y
  (1/2 : ℚ) * ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) : ℤ)

theorem area_of_triangle_ABC (t : GridTriangle) :
  t.A = ⟨0, 0⟩ →
  t.B = ⟨0, 2⟩ →
  t.C = ⟨3, 0⟩ →
  triangleArea t.A t.B t.C = 1/2 := by
  sorry

#check area_of_triangle_ABC

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l1056_105642


namespace NUMINAMATH_CALUDE_elizabeth_money_l1056_105607

/-- The amount of money Elizabeth has, given the costs of pens and pencils and the relationship between the number of pens and pencils she can buy. -/
theorem elizabeth_money : 
  let pencil_cost : ℚ := 8/5  -- $1.60 expressed as a rational number
  let pen_cost : ℚ := 2       -- $2.00
  let pencil_count : ℕ := 5   -- Number of pencils
  let pen_count : ℕ := 6      -- Number of pens
  (pencil_cost * pencil_count + pen_cost * pen_count : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_money_l1056_105607


namespace NUMINAMATH_CALUDE_min_value_log_quadratic_l1056_105669

theorem min_value_log_quadratic (x : ℝ) (h : x^2 - 2*x + 3 > 0) :
  Real.log (x^2 - 2*x + 3) ≥ Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_log_quadratic_l1056_105669


namespace NUMINAMATH_CALUDE_binary_conversion_l1056_105685

-- Define the binary number
def binary_num : List Nat := [1, 0, 1, 1, 0, 0, 1]

-- Define the function to convert binary to decimal
def binary_to_decimal (bin : List Nat) : Nat :=
  bin.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Define the function to convert decimal to octal
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

-- Theorem statement
theorem binary_conversion :
  binary_to_decimal binary_num = 89 ∧
  decimal_to_octal (binary_to_decimal binary_num) = [1, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_conversion_l1056_105685


namespace NUMINAMATH_CALUDE_irrational_sqrt_3_rational_others_l1056_105698

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

-- Theorem statement
theorem irrational_sqrt_3_rational_others :
  IsIrrational (Real.sqrt 3) ∧
  IsRational 0 ∧
  IsRational (-2) ∧
  IsRational (1/2) := by
  sorry

end NUMINAMATH_CALUDE_irrational_sqrt_3_rational_others_l1056_105698


namespace NUMINAMATH_CALUDE_tom_carrot_consumption_l1056_105651

/-- Proves that Tom ate 1 pound of carrots given the conditions of the problem -/
theorem tom_carrot_consumption (C : ℝ) : 
  C > 0 →  -- Assuming C is positive (implicit in the original problem)
  51 * C + 2 * C * (51 / 3) = 85 →
  C = 1 := by
  sorry

end NUMINAMATH_CALUDE_tom_carrot_consumption_l1056_105651


namespace NUMINAMATH_CALUDE_smallest_number_l1056_105627

theorem smallest_number : ∀ (a b c d : ℝ), 
  a = -2 ∧ b = (1 : ℝ) / 2 ∧ c = 0 ∧ d = -Real.sqrt 2 →
  a < b ∧ a < c ∧ a < d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1056_105627


namespace NUMINAMATH_CALUDE_percent_relation_l1056_105603

theorem percent_relation (a b : ℝ) (h : a = 1.8 * b) : 
  4 * b / a = 20 / 9 := by sorry

end NUMINAMATH_CALUDE_percent_relation_l1056_105603


namespace NUMINAMATH_CALUDE_nine_more_knives_l1056_105634

/-- Represents the number of each type of cutlery -/
structure Cutlery where
  forks : ℕ
  knives : ℕ
  spoons : ℕ
  teaspoons : ℕ

/-- The initial state of the cutlery drawer -/
def initial : Cutlery :=
  { forks := 6
  , knives := 6 + 9  -- We're proving this 9
  , spoons := 2 * (6 + 9)
  , teaspoons := 6 / 2 }

/-- The state after adding 2 of each cutlery -/
def after_adding (c : Cutlery) : Cutlery :=
  { forks := c.forks + 2
  , knives := c.knives + 2
  , spoons := c.spoons + 2
  , teaspoons := c.teaspoons + 2 }

/-- The total number of cutlery pieces -/
def total (c : Cutlery) : ℕ :=
  c.forks + c.knives + c.spoons + c.teaspoons

/-- Main theorem: There are 9 more knives than forks initially -/
theorem nine_more_knives :
  initial.knives = initial.forks + 9 ∧
  initial.spoons = 2 * initial.knives ∧
  initial.teaspoons = initial.forks / 2 ∧
  total (after_adding initial) = 62 :=
by sorry

end NUMINAMATH_CALUDE_nine_more_knives_l1056_105634


namespace NUMINAMATH_CALUDE_tomato_suggestion_count_l1056_105686

theorem tomato_suggestion_count (total students_potatoes students_bacon : ℕ) 
  (h1 : total = 826)
  (h2 : students_potatoes = 324)
  (h3 : students_bacon = 374) :
  total - (students_potatoes + students_bacon) = 128 := by
sorry

end NUMINAMATH_CALUDE_tomato_suggestion_count_l1056_105686


namespace NUMINAMATH_CALUDE_triangle_side_a_value_l1056_105625

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_side_a_value (A B C : Real) (a b c : Real) : 
  -- Given conditions
  (Real.tan A = 2 * Real.tan B) →
  (b = Real.sqrt 2) →
  -- Assuming the area is at its maximum (we can't directly express this in Lean without additional setup)
  -- Conclusion
  (a = Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_a_value_l1056_105625


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1056_105676

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x - 6| = 3*x + 5) ↔ (x = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1056_105676


namespace NUMINAMATH_CALUDE_pyramid_height_l1056_105654

theorem pyramid_height (perimeter : ℝ) (apex_to_vertex : ℝ) (h_perimeter : perimeter = 32) (h_apex : apex_to_vertex = 12) :
  let side := perimeter / 4
  let half_diagonal := side * Real.sqrt 2 / 2
  let height := Real.sqrt (apex_to_vertex ^ 2 - half_diagonal ^ 2)
  height = 4 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_l1056_105654


namespace NUMINAMATH_CALUDE_product_equals_one_l1056_105643

theorem product_equals_one :
  16 * 0.5 * 4 * 0.0625 / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l1056_105643


namespace NUMINAMATH_CALUDE_hundred_with_five_twos_l1056_105660

theorem hundred_with_five_twos :
  (222 / 2) - (22 / 2) = 100 :=
by sorry

end NUMINAMATH_CALUDE_hundred_with_five_twos_l1056_105660


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1056_105629

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  2 * x + 1 / (x + 3) ≥ 2 * Real.sqrt 2 - 6 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > -3 ∧ 2 * x + 1 / (x + 3) = 2 * Real.sqrt 2 - 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1056_105629


namespace NUMINAMATH_CALUDE_marion_ella_score_ratio_l1056_105684

/-- Prove that the ratio of Marion's score to Ella's score is 2:3 -/
theorem marion_ella_score_ratio :
  let total_items : ℕ := 40
  let ella_incorrect : ℕ := 4
  let marion_score : ℕ := 24
  let ella_score : ℕ := total_items - ella_incorrect
  (marion_score : ℚ) / ella_score = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_marion_ella_score_ratio_l1056_105684


namespace NUMINAMATH_CALUDE_gum_sharing_theorem_l1056_105683

def john_gum : ℝ := 54.5
def cole_gum : ℝ := 45.75
def aubrey_gum : ℝ := 37
def maria_gum : ℝ := 70.25
def liam_gum : ℝ := 28.5
def emma_gum : ℝ := 32.5

def total_people : ℕ := 6

def total_gum : ℝ := 2 * (john_gum + cole_gum + aubrey_gum + maria_gum + liam_gum + emma_gum)

theorem gum_sharing_theorem : 
  total_gum / total_people = 89.5 := by
  sorry

end NUMINAMATH_CALUDE_gum_sharing_theorem_l1056_105683


namespace NUMINAMATH_CALUDE_wednesday_distance_l1056_105647

theorem wednesday_distance (monday_distance tuesday_distance : ℕ) 
  (average_distance : ℚ) (total_days : ℕ) :
  monday_distance = 12 →
  tuesday_distance = 18 →
  average_distance = 17 →
  total_days = 3 →
  (monday_distance + tuesday_distance + (average_distance * total_days - monday_distance - tuesday_distance : ℚ)) / total_days = average_distance →
  average_distance * total_days - monday_distance - tuesday_distance = 21 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_distance_l1056_105647


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1056_105690

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 4

/-- The total number of people that can ride the wheel at the same time -/
def total_people : ℕ := 20

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := total_people / num_seats

theorem ferris_wheel_capacity : people_per_seat = 5 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1056_105690


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1056_105626

theorem quadratic_factorization (a b : ℤ) :
  (∀ x : ℝ, 25 * x^2 - 155 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -66 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1056_105626


namespace NUMINAMATH_CALUDE_valid_points_characterization_l1056_105691

def is_in_second_quadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

def satisfies_inequality (x y : ℤ) : Prop := y ≤ x + 4

def is_valid_point (x y : ℤ) : Prop :=
  is_in_second_quadrant x y ∧ satisfies_inequality x y

def valid_points : Set (ℤ × ℤ) :=
  {(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-3, 1)}

theorem valid_points_characterization :
  ∀ x y : ℤ, is_valid_point x y ↔ (x, y) ∈ valid_points := by sorry

end NUMINAMATH_CALUDE_valid_points_characterization_l1056_105691


namespace NUMINAMATH_CALUDE_tshirt_cost_l1056_105695

-- Define the problem parameters
def initial_amount : ℕ := 26
def jumper_cost : ℕ := 9
def heels_cost : ℕ := 5
def remaining_amount : ℕ := 8

-- Define the theorem to prove
theorem tshirt_cost : 
  initial_amount - jumper_cost - heels_cost - remaining_amount = 4 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_l1056_105695


namespace NUMINAMATH_CALUDE_largest_b_for_no_real_roots_l1056_105604

theorem largest_b_for_no_real_roots : ∃ (b : ℤ),
  (∀ (x : ℝ), x^3 + b*x^2 + 15*x + 22 ≠ 0) ∧
  (∀ (b' : ℤ), b' > b → ∃ (x : ℝ), x^3 + b'*x^2 + 15*x + 22 = 0) ∧
  b = 5 := by
  sorry


end NUMINAMATH_CALUDE_largest_b_for_no_real_roots_l1056_105604


namespace NUMINAMATH_CALUDE_least_number_with_remainder_four_l1056_105655

theorem least_number_with_remainder_four (n : ℕ) : n = 184 ↔ 
  (n > 0) ∧ 
  (∀ m : ℕ, 0 < m ∧ m < n → 
    (m % 5 ≠ 4 ∨ m % 6 ≠ 4 ∨ m % 9 ≠ 4 ∨ m % 12 ≠ 4)) ∧
  (n % 5 = 4) ∧ (n % 6 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_four_l1056_105655


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1056_105639

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the circle (x-1)^2 + y^2 = 3 -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 3

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ x y, circle_equation x y ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
                   c.center = (1, 0) ∧
                   c.radius = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1056_105639


namespace NUMINAMATH_CALUDE_oranges_in_basket_l1056_105631

/-- The number of oranges in a fruit basket -/
def num_oranges : ℕ := 6

/-- The number of apples in the fruit basket -/
def num_apples : ℕ := num_oranges - 2

/-- The number of bananas in the fruit basket -/
def num_bananas : ℕ := 3 * num_apples

/-- The number of peaches in the fruit basket -/
def num_peaches : ℕ := num_bananas / 2

/-- Theorem: The number of oranges in the fruit basket is 6 -/
theorem oranges_in_basket : 
  num_oranges + num_apples + num_bananas + num_peaches = 28 → num_oranges = 6 := by
  sorry


end NUMINAMATH_CALUDE_oranges_in_basket_l1056_105631


namespace NUMINAMATH_CALUDE_equation_solutions_l1056_105601

theorem equation_solutions :
  (∀ x : ℝ, (x - 5)^2 = 16 ↔ x = 1 ∨ x = 9) ∧
  (∀ x : ℝ, 2*x^2 - 1 = -4*x ↔ x = -1 + Real.sqrt 6 / 2 ∨ x = -1 - Real.sqrt 6 / 2) ∧
  (∀ x : ℝ, 5*x*(x+1) = 2*(x+1) ↔ x = -1 ∨ x = 2/5) ∧
  (∀ x : ℝ, 2*x^2 - x - 1 = 0 ↔ x = -1/2 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1056_105601


namespace NUMINAMATH_CALUDE_min_disks_needed_l1056_105659

def total_files : ℕ := 40
def disk_capacity : ℚ := 1.44
def large_files : ℕ := 5
def medium_files : ℕ := 15
def small_files : ℕ := total_files - large_files - medium_files
def large_file_size : ℚ := 0.9
def medium_file_size : ℚ := 0.75
def small_file_size : ℚ := 0.5

theorem min_disks_needed :
  let total_size := large_files * large_file_size + medium_files * medium_file_size + small_files * small_file_size
  ∃ (n : ℕ), n * disk_capacity ≥ total_size ∧
             ∀ (m : ℕ), m * disk_capacity ≥ total_size → n ≤ m ∧
             n = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_disks_needed_l1056_105659


namespace NUMINAMATH_CALUDE_power_product_equals_eight_l1056_105635

theorem power_product_equals_eight (m n : ℤ) (h : 2 * m + n - 3 = 0) :
  (4 : ℝ) ^ m * (2 : ℝ) ^ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_eight_l1056_105635


namespace NUMINAMATH_CALUDE_clares_money_l1056_105657

/-- The amount of money Clare's mother gave her --/
def money_from_mother : ℕ := sorry

/-- The number of loaves of bread Clare bought --/
def bread_count : ℕ := 4

/-- The number of cartons of milk Clare bought --/
def milk_count : ℕ := 2

/-- The cost of one loaf of bread in dollars --/
def bread_cost : ℕ := 2

/-- The cost of one carton of milk in dollars --/
def milk_cost : ℕ := 2

/-- The amount of money Clare has left after shopping --/
def money_left : ℕ := 35

/-- Theorem stating that the amount of money Clare's mother gave her is $47 --/
theorem clares_money : money_from_mother = 47 := by
  sorry

end NUMINAMATH_CALUDE_clares_money_l1056_105657


namespace NUMINAMATH_CALUDE_relationship_abc_l1056_105617

theorem relationship_abc : ∀ (a b c : ℕ),
  a = 5^140 ∧ b = 3^210 ∧ c = 2^280 →
  c < a ∧ a < b := by
sorry

end NUMINAMATH_CALUDE_relationship_abc_l1056_105617


namespace NUMINAMATH_CALUDE_irregular_shape_impossible_l1056_105677

/-- Represents a shape formed by two equilateral triangles --/
structure TwoTriangleShape where
  -- Add necessary fields to describe the shape

/-- Predicate to check if a shape is regular (has symmetry or regularity) --/
def is_regular (s : TwoTriangleShape) : Prop :=
  sorry  -- Definition of regularity

/-- Predicate to check if a shape can be formed by two equilateral triangles --/
def can_be_formed_by_triangles (s : TwoTriangleShape) : Prop :=
  sorry  -- Definition based on triangle placement rules

theorem irregular_shape_impossible (s : TwoTriangleShape) :
  ¬(is_regular s) → ¬(can_be_formed_by_triangles s) :=
  sorry  -- The proof would go here

end NUMINAMATH_CALUDE_irregular_shape_impossible_l1056_105677


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l1056_105687

theorem rectangular_garden_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 507 →
  width = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l1056_105687


namespace NUMINAMATH_CALUDE_remainder_7325_div_11_l1056_105688

theorem remainder_7325_div_11 : 7325 % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7325_div_11_l1056_105688


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l1056_105645

def binary_to_decimal (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n |>.reverse

def a : List Bool := [true, true, false]
def b : List Bool := [true, false, true]
def c : List Bool := [true, false, true, true]
def d : List Bool := [true, false, false, true, true]

theorem binary_sum_theorem :
  decimal_to_binary (binary_to_decimal a + binary_to_decimal b + 
                     binary_to_decimal c + binary_to_decimal d) =
  [true, false, true, false, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_theorem_l1056_105645


namespace NUMINAMATH_CALUDE_darrel_nickels_l1056_105667

def quarters : ℕ := 76
def dimes : ℕ := 85
def pennies : ℕ := 150
def fee_percentage : ℚ := 10 / 100
def amount_after_fee : ℚ := 27

def quarter_value : ℚ := 25 / 100
def dime_value : ℚ := 10 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100

theorem darrel_nickels :
  let total_before_fee := amount_after_fee / (1 - fee_percentage)
  let known_coins_value := quarters * quarter_value + dimes * dime_value + pennies * penny_value
  let nickel_value_sum := total_before_fee - known_coins_value
  (nickel_value_sum / nickel_value : ℚ) = 20 := by sorry

end NUMINAMATH_CALUDE_darrel_nickels_l1056_105667


namespace NUMINAMATH_CALUDE_map_to_actual_distance_l1056_105637

/-- Given a map scale and a road length on the map, calculate the actual road length in kilometers. -/
theorem map_to_actual_distance (scale : ℚ) (map_length : ℚ) (actual_length : ℚ) : 
  scale = 1 / 50000 →
  map_length = 15 →
  actual_length = 7.5 →
  scale * actual_length = map_length := by
  sorry

#check map_to_actual_distance

end NUMINAMATH_CALUDE_map_to_actual_distance_l1056_105637


namespace NUMINAMATH_CALUDE_early_movie_savings_l1056_105611

/-- Calculates the savings for going to an earlier movie given ticket and food combo prices and discounts --/
theorem early_movie_savings 
  (evening_ticket_price : ℚ)
  (evening_combo_price : ℚ)
  (ticket_discount_percent : ℚ)
  (combo_discount_percent : ℚ)
  (h1 : evening_ticket_price = 10)
  (h2 : evening_combo_price = 10)
  (h3 : ticket_discount_percent = 20 / 100)
  (h4 : combo_discount_percent = 50 / 100)
  : evening_ticket_price * ticket_discount_percent + 
    evening_combo_price * combo_discount_percent = 7 := by
  sorry

end NUMINAMATH_CALUDE_early_movie_savings_l1056_105611


namespace NUMINAMATH_CALUDE_largest_factorable_n_l1056_105662

/-- The largest value of n for which 3x^2 + nx + 90 can be factored as the product of two linear factors with integer coefficients -/
def largest_n : ℕ := 271

/-- A function that checks if a quadratic expression 3x^2 + nx + 90 can be factored as the product of two linear factors with integer coefficients -/
def is_factorable (n : ℤ) : Prop :=
  ∃ (a b : ℤ), (3 * a + b = n) ∧ (a * b = 90)

theorem largest_factorable_n :
  (is_factorable largest_n) ∧ 
  (∀ m : ℤ, m > largest_n → ¬(is_factorable m)) :=
sorry

end NUMINAMATH_CALUDE_largest_factorable_n_l1056_105662


namespace NUMINAMATH_CALUDE_circle_C_properties_l1056_105602

def circle_C (ρ θ : ℝ) : Prop :=
  ρ^2 = 4*ρ*(Real.cos θ + Real.sin θ) - 3

def point_on_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = 2 + Real.sqrt 5 * Real.cos θ ∧ y = 2 + Real.sqrt 5 * Real.sin θ

theorem circle_C_properties :
  -- 1. Parametric equations
  (∀ x y θ : ℝ, point_on_C x y ↔ 
    x = 2 + Real.sqrt 5 * Real.cos θ ∧ 
    y = 2 + Real.sqrt 5 * Real.sin θ) ∧
  -- 2. Maximum value of x + 2y
  (∀ x y : ℝ, point_on_C x y → x + 2*y ≤ 11) ∧
  -- 3. Coordinates at maximum
  (point_on_C 3 4 ∧ 3 + 2*4 = 11) :=
by sorry

end NUMINAMATH_CALUDE_circle_C_properties_l1056_105602


namespace NUMINAMATH_CALUDE_correct_set_for_60_deg_terminal_side_l1056_105624

/-- The set of angles with the same terminal side as a 60° angle -/
def SameTerminalSideAs60Deg : Set ℝ :=
  {α | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3}

/-- Theorem stating that SameTerminalSideAs60Deg is the correct set -/
theorem correct_set_for_60_deg_terminal_side :
  SameTerminalSideAs60Deg = {α | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3} := by
  sorry

end NUMINAMATH_CALUDE_correct_set_for_60_deg_terminal_side_l1056_105624


namespace NUMINAMATH_CALUDE_after_school_program_l1056_105644

/-- Represents the number of students in each class combination --/
structure ClassCombinations where
  drawing_only : ℕ
  chess_only : ℕ
  music_only : ℕ
  drawing_chess : ℕ
  drawing_music : ℕ
  chess_music : ℕ
  all_three : ℕ

/-- The after-school program problem --/
theorem after_school_program 
  (total_students : ℕ) 
  (drawing_students : ℕ) 
  (chess_students : ℕ) 
  (music_students : ℕ) 
  (multi_class_students : ℕ) 
  (h1 : total_students = 30)
  (h2 : drawing_students = 15)
  (h3 : chess_students = 17)
  (h4 : music_students = 12)
  (h5 : multi_class_students = 14)
  : ∃ (c : ClassCombinations), 
    c.drawing_only + c.chess_only + c.music_only + 
    c.drawing_chess + c.drawing_music + c.chess_music + c.all_three = total_students ∧
    c.drawing_only + c.drawing_chess + c.drawing_music + c.all_three = drawing_students ∧
    c.chess_only + c.drawing_chess + c.chess_music + c.all_three = chess_students ∧
    c.music_only + c.drawing_music + c.chess_music + c.all_three = music_students ∧
    c.drawing_chess + c.drawing_music + c.chess_music + c.all_three = multi_class_students ∧
    c.all_three = 2 := by
  sorry

end NUMINAMATH_CALUDE_after_school_program_l1056_105644


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1056_105600

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (5 + n) = 7 → n = 44 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1056_105600


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_expression_l1056_105674

theorem smallest_prime_factor_of_expression : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (12^3 + 15^4 - 6^6) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (12^3 + 15^4 - 6^6) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_expression_l1056_105674


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1056_105672

theorem intersection_of_sets : 
  let A : Set ℤ := {1, 2, 3}
  let B : Set ℤ := {-2, 2}
  A ∩ B = {2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1056_105672


namespace NUMINAMATH_CALUDE_range_of_m_given_quadratic_inequality_l1056_105628

theorem range_of_m_given_quadratic_inequality (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*m*x + m + 2 ≥ 0) ↔ m ∈ Set.Icc (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_given_quadratic_inequality_l1056_105628


namespace NUMINAMATH_CALUDE_super_bowl_probability_sum_l1056_105682

theorem super_bowl_probability_sum :
  ∀ (p_play p_not_play : ℝ),
  p_play = 9 * p_not_play →
  p_play ≥ 0 →
  p_not_play ≥ 0 →
  p_play + p_not_play = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_super_bowl_probability_sum_l1056_105682


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l1056_105632

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) :
  total_length = 35 →
  ratio = 2 / 5 →
  ∃ shorter_length longer_length : ℝ,
    shorter_length + longer_length = total_length ∧
    shorter_length = ratio * longer_length ∧
    shorter_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l1056_105632


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1056_105636

theorem arithmetic_mean_problem (a b c : ℝ) 
  (h1 : (a + b) / 2 = 80) 
  (h2 : (b + c) / 2 = 180) : 
  a - c = -200 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1056_105636


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1056_105646

theorem fraction_multiplication : (2 : ℚ) / 3 * 5 / 7 * 8 / 9 = 80 / 189 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1056_105646


namespace NUMINAMATH_CALUDE_binary_11001_is_25_l1056_105678

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11001_is_25 :
  binary_to_decimal [true, false, false, true, true] = 25 := by
  sorry

end NUMINAMATH_CALUDE_binary_11001_is_25_l1056_105678


namespace NUMINAMATH_CALUDE_units_digit_problem_l1056_105648

theorem units_digit_problem : ∃ n : ℕ, n % 10 = 7 ∧ 
  (((2008^2 + 2^2008)^2 + 2^(2008^2 + 2^2008)) % 10 = n % 10) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1056_105648


namespace NUMINAMATH_CALUDE_always_sum_21_l1056_105666

theorem always_sum_21 (selection : Finset ℕ) :
  selection ⊆ Finset.range 20 →
  selection.card = 11 →
  ∃ x y, x ∈ selection ∧ y ∈ selection ∧ x ≠ y ∧ x + y = 21 :=
sorry

end NUMINAMATH_CALUDE_always_sum_21_l1056_105666


namespace NUMINAMATH_CALUDE_complex_magnitude_calculation_l1056_105613

theorem complex_magnitude_calculation : 
  Complex.abs (6 - 3 * Complex.I) * Complex.abs (6 + 3 * Complex.I) - 2 * Complex.abs (5 - Complex.I) = 45 - 2 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_calculation_l1056_105613


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l1056_105692

/-- Given that R varies directly as S and inversely as T, 
    prove that S = 20/3 when R = 5 and T = 1/3, 
    given that R = 2, T = 1/2, and S = 4 in another case. -/
theorem direct_inverse_variation (R S T : ℚ) : 
  (∃ k : ℚ, ∀ R S T, R = k * S / T) →  -- R varies directly as S and inversely as T
  (2 : ℚ) = (4 : ℚ) / (1/2 : ℚ) →      -- When R = 2, S = 4, and T = 1/2
  (5 : ℚ) = S / (1/3 : ℚ) →            -- When R = 5 and T = 1/3
  S = 20/3 := by
sorry

end NUMINAMATH_CALUDE_direct_inverse_variation_l1056_105692


namespace NUMINAMATH_CALUDE_distance_between_locations_l1056_105653

/-- The distance between two locations given the speeds of two vehicles traveling towards each other and the time they take to meet. -/
theorem distance_between_locations (car_speed truck_speed : ℝ) (time : ℝ) : 
  car_speed > 0 → truck_speed > 0 → time > 0 →
  (car_speed + truck_speed) * time = 1925 → car_speed = 100 → truck_speed = 75 → time = 11 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_locations_l1056_105653


namespace NUMINAMATH_CALUDE_limit_at_neg_seven_l1056_105620

/-- The limit of (2x^2 + 15x + 7)/(x + 7) as x approaches -7 is -13 -/
theorem limit_at_neg_seven (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, x ≠ -7 →
    |x - (-7)| < δ → |(2*x^2 + 15*x + 7)/(x + 7) - (-13)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_at_neg_seven_l1056_105620


namespace NUMINAMATH_CALUDE_expansion_equality_l1056_105616

theorem expansion_equality (x y z : ℝ) :
  (x + 10*z + 5) * (2*y + 15) = 2*x*y + 20*y*z + 15*x + 10*y + 150*z + 75 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l1056_105616


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1056_105665

theorem min_value_of_expression (x y z : ℝ) : (x^2*y - 1)^2 + (x + y + z)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1056_105665


namespace NUMINAMATH_CALUDE_inverse_statement_is_false_l1056_105621

-- Define the set S of all non-zero real numbers
def S : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 3 * a * b

-- Define what it means for an element to be an inverse under *
def is_inverse (a b : ℝ) : Prop := star a b = 1/3 ∧ star b a = 1/3

-- The theorem to be proved
theorem inverse_statement_is_false :
  ∀ a ∈ S, ¬(is_inverse a (1/(3*a))) := by
  sorry

end NUMINAMATH_CALUDE_inverse_statement_is_false_l1056_105621


namespace NUMINAMATH_CALUDE_square_gt_necessary_not_sufficient_l1056_105618

theorem square_gt_necessary_not_sufficient (a : ℝ) :
  (∀ a, a > 1 → a^2 > a) ∧ 
  (∃ a, a^2 > a ∧ ¬(a > 1)) :=
sorry

end NUMINAMATH_CALUDE_square_gt_necessary_not_sufficient_l1056_105618


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1056_105641

theorem sufficient_not_necessary (p q : Prop) :
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1056_105641


namespace NUMINAMATH_CALUDE_range_of_3a_minus_b_l1056_105615

theorem range_of_3a_minus_b (a b : ℝ) (ha : -5 < a ∧ a < 2) (hb : 1 < b ∧ b < 4) :
  -19 < 3 * a - b ∧ 3 * a - b < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_b_l1056_105615


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_5_over_32_l1056_105697

theorem ten_thousandths_place_of_5_over_32 : 
  ∃ (a b c d : ℕ), (5 : ℚ) / 32 = (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + (6 : ℚ) / 10000 + (d : ℚ) / 100000 :=
by sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_5_over_32_l1056_105697


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l1056_105623

theorem complex_sum_of_parts (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  (z * Complex.mk 1 2 = 5) → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l1056_105623


namespace NUMINAMATH_CALUDE_alex_sandwiches_l1056_105649

/-- The number of different sandwiches Alex can make -/
def num_sandwiches (num_meats : ℕ) (num_cheeses : ℕ) (num_breads : ℕ) : ℕ :=
  num_meats * (1 + num_cheeses + (num_cheeses.choose 2)) * num_breads

/-- Theorem stating the number of different sandwiches Alex can make -/
theorem alex_sandwiches :
  num_sandwiches 12 11 3 = 2412 := by sorry

end NUMINAMATH_CALUDE_alex_sandwiches_l1056_105649


namespace NUMINAMATH_CALUDE_sin_cos_identity_tan_fraction_value_l1056_105619

-- Part 1
theorem sin_cos_identity (α : Real) :
  (Real.sin (3 * α) / Real.sin α) - (Real.cos (3 * α) / Real.cos α) = 2 := by
  sorry

-- Part 2
theorem tan_fraction_value (α : Real) (h : Real.tan (α / 2) = 2) :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_tan_fraction_value_l1056_105619


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l1056_105693

/-- Given a cubic function f(x) = ax³ + 3x² + 2, prove that if f'(-1) = 4, then a = 10/3 -/
theorem cubic_function_derivative (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + 3 * x^2 + 2
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l1056_105693


namespace NUMINAMATH_CALUDE_direct_variation_theorem_y_value_at_negative_ten_l1056_105663

/-- A function representing direct variation --/
def DirectVariation (k : ℝ) : ℝ → ℝ := fun x ↦ k * x

theorem direct_variation_theorem (k : ℝ) :
  (DirectVariation k 5 = 15) → (DirectVariation k (-10) = -30) := by
  sorry

/-- Main theorem proving the relationship between y and x --/
theorem y_value_at_negative_ten :
  ∃ k : ℝ, (DirectVariation k 5 = 15) ∧ (DirectVariation k (-10) = -30) := by
  sorry

end NUMINAMATH_CALUDE_direct_variation_theorem_y_value_at_negative_ten_l1056_105663


namespace NUMINAMATH_CALUDE_baker_sold_cakes_l1056_105608

/-- Given that Baker bought 31 cakes and sold 47 more cakes than he bought,
    prove that Baker sold 78 cakes. -/
theorem baker_sold_cakes : ℕ → Prop :=
  fun cakes_bought : ℕ =>
    cakes_bought = 31 →
    ∃ cakes_sold : ℕ,
      cakes_sold = cakes_bought + 47 ∧
      cakes_sold = 78

/-- Proof of the theorem -/
lemma prove_baker_sold_cakes : baker_sold_cakes 31 := by
  sorry

end NUMINAMATH_CALUDE_baker_sold_cakes_l1056_105608


namespace NUMINAMATH_CALUDE_x_less_than_one_necessary_not_sufficient_for_x_squared_less_than_one_l1056_105661

theorem x_less_than_one_necessary_not_sufficient_for_x_squared_less_than_one :
  ∀ x : ℝ, (x^2 < 1 → x < 1) ∧ ¬(x < 1 → x^2 < 1) := by sorry

end NUMINAMATH_CALUDE_x_less_than_one_necessary_not_sufficient_for_x_squared_less_than_one_l1056_105661


namespace NUMINAMATH_CALUDE_even_expression_l1056_105681

theorem even_expression (x : ℤ) (h : x = 3) : 
  ∃ k : ℤ, 2 * (x^2 + 9) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_even_expression_l1056_105681


namespace NUMINAMATH_CALUDE_no_solution_for_a_l1056_105694

theorem no_solution_for_a (x : ℝ) (h : x = 4) :
  ¬∃ a : ℝ, a / (x + 4) + a / (x - 4) = a / (x - 4) :=
sorry

end NUMINAMATH_CALUDE_no_solution_for_a_l1056_105694


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_division_remainder_l1056_105630

def f (x : ℝ) : ℝ := 5 * x^3 - 10 * x^2 + 15 * x - 20

def divisor (x : ℝ) : ℝ := 5 * x - 10

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := by sorry

theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x, f x = divisor x * q x + 10 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_division_remainder_l1056_105630


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l1056_105673

theorem snow_leopard_arrangement (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 3) :
  (k.factorial) * ((n - k).factorial) = 4320 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l1056_105673


namespace NUMINAMATH_CALUDE_wang_elevator_journey_l1056_105612

def floor_movements : List Int := [6, -3, 10, -8, 12, -7, -10]
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.2

theorem wang_elevator_journey :
  (List.sum floor_movements = 0) ∧
  (List.sum (List.map Int.natAbs floor_movements) * floor_height * electricity_per_meter = 33.6) := by
  sorry

end NUMINAMATH_CALUDE_wang_elevator_journey_l1056_105612


namespace NUMINAMATH_CALUDE_discount_reduction_l1056_105699

theorem discount_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.2
  let remaining_after_first := 1 - first_discount
  let remaining_after_second := 1 - second_discount
  let final_price := original_price * remaining_after_first * remaining_after_second
  (original_price - final_price) / original_price = 0.44 :=
by
  sorry

end NUMINAMATH_CALUDE_discount_reduction_l1056_105699


namespace NUMINAMATH_CALUDE_range_of_H_l1056_105658

/-- The function H defined as the difference of absolute values -/
def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

/-- Theorem stating the range of function H -/
theorem range_of_H :
  (∀ x : ℝ, H x ≥ -4 ∧ H x ≤ 4) ∧
  (∃ x : ℝ, H x = -4) ∧
  (∃ x : ℝ, H x = 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_H_l1056_105658


namespace NUMINAMATH_CALUDE_donald_oranges_l1056_105640

/-- Given that Donald has 4 oranges initially and finds 5 more,
    prove that he has 9 oranges in total. -/
theorem donald_oranges (initial : Nat) (found : Nat) (total : Nat) 
    (h1 : initial = 4) 
    (h2 : found = 5) 
    (h3 : total = initial + found) : 
  total = 9 := by
  sorry

end NUMINAMATH_CALUDE_donald_oranges_l1056_105640


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l1056_105671

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n α) : 
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l1056_105671


namespace NUMINAMATH_CALUDE_sale_price_lower_than_original_l1056_105696

theorem sale_price_lower_than_original (x : ℝ) (h : x > 0) :
  0.75 * (1.30 * x) < x := by
  sorry

end NUMINAMATH_CALUDE_sale_price_lower_than_original_l1056_105696


namespace NUMINAMATH_CALUDE_triangle_problem_l1056_105680

theorem triangle_problem (A B C : Real) (a b c : Real) : 
  -- Given conditions
  (a = 2) →
  (b = Real.sqrt 6) →
  (B = 60 * π / 180) →
  -- Triangle properties
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Conclusions
  (A = 45 * π / 180 ∧ 
   C = 75 * π / 180 ∧ 
   c = 1 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1056_105680


namespace NUMINAMATH_CALUDE_square_area_ratio_after_doubling_l1056_105679

theorem square_area_ratio_after_doubling (s : ℝ) (h : s > 0) :
  (s^2) / ((2*s)^2) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_after_doubling_l1056_105679


namespace NUMINAMATH_CALUDE_range_of_a_l1056_105652

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  (9 - 5*x) / 4 > 1 ∧ x < a

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 1

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x, inequality_system x a ↔ solution_set x) → a ≥ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l1056_105652


namespace NUMINAMATH_CALUDE_total_trees_planted_l1056_105689

/-- The total number of trees planted by a family in spring -/
theorem total_trees_planted (apricot peach cherry : ℕ) : 
  apricot = 58 →
  peach = 3 * apricot →
  cherry = 5 * peach →
  apricot + peach + cherry = 1102 := by
  sorry

end NUMINAMATH_CALUDE_total_trees_planted_l1056_105689


namespace NUMINAMATH_CALUDE_union_and_intersection_when_a_eq_4_subset_condition_l1056_105605

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x ≤ 4}

-- Define set B with parameter a
def B (a : ℝ) : Set ℝ := {x | 6 - a < x ∧ x < 2*a - 1}

-- Theorem for question 1
theorem union_and_intersection_when_a_eq_4 :
  (A ∪ B 4) = {x | 1 < x ∧ x < 7} ∧
  (B 4 ∩ (U \ A)) = {x | 4 < x ∧ x < 7} :=
sorry

-- Theorem for question 2
theorem subset_condition :
  ∀ a : ℝ, A ⊆ B a ↔ a ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_union_and_intersection_when_a_eq_4_subset_condition_l1056_105605


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equal_l1056_105622

theorem polygon_interior_exterior_angles_equal (n : ℕ) : 
  n ≥ 3 → (n - 2) * 180 = 360 → n = 4 := by sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equal_l1056_105622


namespace NUMINAMATH_CALUDE_square_area_not_covered_by_circle_l1056_105638

theorem square_area_not_covered_by_circle (d : ℝ) (h : d = 8) :
  let r := d / 2
  let square_area := d^2
  let circle_area := π * r^2
  square_area - circle_area = 64 - 16 * π := by
  sorry

end NUMINAMATH_CALUDE_square_area_not_covered_by_circle_l1056_105638


namespace NUMINAMATH_CALUDE_jim_out_of_pocket_l1056_105675

def out_of_pocket (first_ring_cost second_ring_cost first_ring_sale_price : ℕ) : ℕ :=
  first_ring_cost + second_ring_cost - first_ring_sale_price

theorem jim_out_of_pocket :
  let first_ring_cost : ℕ := 10000
  let second_ring_cost : ℕ := 2 * first_ring_cost
  let first_ring_sale_price : ℕ := first_ring_cost / 2
  out_of_pocket first_ring_cost second_ring_cost first_ring_sale_price = 25000 := by
  sorry

end NUMINAMATH_CALUDE_jim_out_of_pocket_l1056_105675


namespace NUMINAMATH_CALUDE_solution_set_m_zero_solution_set_real_l1056_105668

-- Define the inequality
def inequality (m : ℝ) (x : ℝ) : Prop :=
  (m - 1) * x^2 + (m - 1) * x + 2 > 0

-- Part 1: Solution set when m = 0
theorem solution_set_m_zero :
  {x : ℝ | inequality 0 x} = Set.Ioo (-2) 1 := by sorry

-- Part 2: Range of m for solution set = ℝ
theorem solution_set_real :
  ∀ m : ℝ, (∀ x : ℝ, inequality m x) ↔ 1 ≤ m ∧ m < 9 := by sorry

end NUMINAMATH_CALUDE_solution_set_m_zero_solution_set_real_l1056_105668


namespace NUMINAMATH_CALUDE_min_value_A_min_value_A_equality_l1056_105606

theorem min_value_A (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  let A := (Real.sqrt (3 * x^4 + y) + Real.sqrt (3 * y^4 + z) + Real.sqrt (3 * z^4 + x) - 3) / (x * y + y * z + z * x)
  A ≥ 1 := by
  sorry

theorem min_value_A_equality (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  let A := (Real.sqrt (3 * x^4 + y) + Real.sqrt (3 * y^4 + z) + Real.sqrt (3 * z^4 + x) - 3) / (x * y + y * z + z * x)
  (A = 1) ↔ (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_A_min_value_A_equality_l1056_105606


namespace NUMINAMATH_CALUDE_maintenance_team_schedule_l1056_105650

theorem maintenance_team_schedule : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 9 11)) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_team_schedule_l1056_105650


namespace NUMINAMATH_CALUDE_polynomial_value_at_minus_one_l1056_105670

/-- Given real numbers a, b, and c, define polynomials g and f -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 5
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 50*x + c

/-- The main theorem to prove -/
theorem polynomial_value_at_minus_one 
  (a b c : ℝ) 
  (h1 : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
    g a r1 = 0 ∧ g a r2 = 0 ∧ g a r3 = 0)
  (h2 : ∀ x : ℝ, g a x = 0 → f b c x = 0) :
  f b c (-1) = -1804 := by
  sorry

#check polynomial_value_at_minus_one

end NUMINAMATH_CALUDE_polynomial_value_at_minus_one_l1056_105670


namespace NUMINAMATH_CALUDE_two_x_is_equal_mean_value_function_l1056_105664

/-- A function is an "equal mean value function" if it satisfies two conditions:
    1) For any x in its domain, f(x) + f(-x) = 0
    2) For any x₁ in its domain, there exists x₂ such that (f(x₁) + f(x₂))/2 = (x₁ + x₂)/2 -/
def is_equal_mean_value_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x₁, ∃ x₂, (f x₁ + f x₂) / 2 = (x₁ + x₂) / 2)

/-- The function f(x) = 2x is an "equal mean value function" -/
theorem two_x_is_equal_mean_value_function :
  is_equal_mean_value_function (λ x ↦ 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_two_x_is_equal_mean_value_function_l1056_105664
