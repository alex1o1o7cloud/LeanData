import Mathlib

namespace NUMINAMATH_CALUDE_arrangements_with_A_B_at_ends_arrangements_with_A_B_not_adjacent_adjustment_methods_l1925_192516

-- Define the number of instructors and students
def num_instructors : ℕ := 3
def num_students : ℕ := 7

-- Define the theorem for part (1)
theorem arrangements_with_A_B_at_ends :
  (2 * Nat.factorial 5 * Nat.factorial num_instructors : ℕ) = 1440 := by sorry

-- Define the theorem for part (2)
theorem arrangements_with_A_B_not_adjacent :
  (Nat.factorial 5 * Nat.choose 6 2 * 2 * Nat.factorial num_instructors : ℕ) = 21600 := by sorry

-- Define the theorem for part (3)
theorem adjustment_methods :
  (Nat.choose num_students 2 * (Nat.factorial 5 / Nat.factorial 3) : ℕ) = 420 := by sorry

end NUMINAMATH_CALUDE_arrangements_with_A_B_at_ends_arrangements_with_A_B_not_adjacent_adjustment_methods_l1925_192516


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1925_192579

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1925_192579


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l1925_192557

/-- A cuboid with three distinct side areas -/
structure Cuboid where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ

/-- The total surface area of a cuboid -/
def surface_area (c : Cuboid) : ℝ := 2 * (c.area1 + c.area2 + c.area3)

/-- Theorem: The surface area of a cuboid with side areas 4, 3, and 6 is 26 -/
theorem cuboid_surface_area :
  let c : Cuboid := { area1 := 4, area2 := 3, area3 := 6 }
  surface_area c = 26 := by
  sorry

#check cuboid_surface_area

end NUMINAMATH_CALUDE_cuboid_surface_area_l1925_192557


namespace NUMINAMATH_CALUDE_two_digit_factorizations_of_2210_l1925_192585

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ a * b = 2210

def distinct_factorizations (f1 f2 : ℕ × ℕ) : Prop :=
  f1.1 ≠ f2.1 ∧ f1.1 ≠ f2.2

theorem two_digit_factorizations_of_2210 :
  ∃ (f1 f2 : ℕ × ℕ),
    valid_factorization f1.1 f1.2 ∧
    valid_factorization f2.1 f2.2 ∧
    distinct_factorizations f1 f2 ∧
    ∀ (f : ℕ × ℕ), valid_factorization f.1 f.2 →
      (f = f1 ∨ f = f2 ∨ f = (f1.2, f1.1) ∨ f = (f2.2, f2.1)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_factorizations_of_2210_l1925_192585


namespace NUMINAMATH_CALUDE_volume_of_specific_pyramid_l1925_192530

/-- A right pyramid with a square base and an equilateral triangular face --/
structure RightPyramid where
  -- The side length of the equilateral triangular face
  side_length : ℝ
  -- Assumption that the side length is positive
  side_length_pos : side_length > 0

/-- The volume of the right pyramid --/
noncomputable def volume (p : RightPyramid) : ℝ :=
  (500 * Real.sqrt 3) / 3

/-- Theorem stating the volume of the specific pyramid --/
theorem volume_of_specific_pyramid :
  ∃ (p : RightPyramid), p.side_length = 10 ∧ volume p = (500 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_pyramid_l1925_192530


namespace NUMINAMATH_CALUDE_unique_number_l1925_192541

theorem unique_number : ∃! (n : ℕ), n > 0 ∧ n^2 + n = 217 ∧ 3 ∣ n ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_l1925_192541


namespace NUMINAMATH_CALUDE_red_balls_in_box_l1925_192564

/-- Given a box with an initial number of red balls and a number of red balls added,
    calculate the final number of red balls in the box. -/
def final_red_balls (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The final number of red balls in the box is 7 when starting with 5 and adding 2. -/
theorem red_balls_in_box : final_red_balls 5 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_in_box_l1925_192564


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_sum_power_l1925_192572

theorem sqrt_plus_square_zero_implies_sum_power (x y : ℝ) :
  Real.sqrt (x - 1) + (y + 2)^2 = 0 → (x + y)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_sum_power_l1925_192572


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_three_halves_l1925_192543

/-- Parametric equation of the first line -/
def line1 (r : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (2 + r, -1 - 2*k*r, 3 + k*r)

/-- Parametric equation of the second line -/
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (1 + 3*t, 2 - t, 1 + 2*t)

/-- Direction vector of the first line -/
def dir1 (k : ℝ) : ℝ × ℝ × ℝ := (1, -2*k, k)

/-- Direction vector of the second line -/
def dir2 : ℝ × ℝ × ℝ := (3, -1, 2)

/-- Two lines are coplanar if their direction vectors are proportional -/
def coplanar (k : ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ dir1 k = (c • dir2)

theorem lines_coplanar_iff_k_eq_three_halves :
  ∃ (k : ℝ), coplanar k ↔ k = 3/2 :=
sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_three_halves_l1925_192543


namespace NUMINAMATH_CALUDE_sum_of_erased_numbers_l1925_192526

/-- Represents a sequence of odd numbers -/
def OddSequence (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * i + 1)

/-- The sum of the first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n ^ 2

/-- Theorem: Sum of erased numbers in the sequence -/
theorem sum_of_erased_numbers
  (n : ℕ) -- Length of the first part
  (h1 : sumOddNumbers (n + 2) = 4147) -- Sum of third part is 4147
  (h2 : n > 0) -- Ensure non-empty sequence
  : ∃ (a b : ℕ), a ∈ OddSequence (4 * n + 6) ∧ 
                 b ∈ OddSequence (4 * n + 6) ∧ 
                 a + b = 168 :=
sorry

end NUMINAMATH_CALUDE_sum_of_erased_numbers_l1925_192526


namespace NUMINAMATH_CALUDE_broken_line_endpoint_characterization_l1925_192590

/-- A broken line from O to M -/
structure BrokenLine where
  segments : List (ℝ × ℝ)
  start_at_origin : segments.foldl (λ acc (x, y) => (acc.1 + x, acc.2 + y)) (0, 0) = (0, 0)
  unit_length : segments.foldl (λ acc (x, y) => acc + x^2 + y^2) 0 = 1

/-- Predicate to check if a broken line satisfies the intersection condition -/
def satisfies_intersection_condition (l : BrokenLine) : Prop :=
  ∀ (a b : ℝ), (∀ (x y : ℝ), (x, y) ∈ l.segments → (a * x + b * y ≠ 0 ∨ a * x + b * y ≠ 1))

theorem broken_line_endpoint_characterization (x y : ℝ) :
  (∃ (l : BrokenLine), satisfies_intersection_condition l ∧ 
   l.segments.foldl (λ acc (dx, dy) => (acc.1 + dx, acc.2 + dy)) (0, 0) = (x, y)) →
  x^2 + y^2 ≤ 1 ∧ |x| + |y| ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_endpoint_characterization_l1925_192590


namespace NUMINAMATH_CALUDE_punch_water_calculation_l1925_192505

/-- Calculates the amount of water needed for a punch mixture -/
def water_needed (total_volume : ℚ) (water_parts : ℕ) (juice_parts : ℕ) : ℚ :=
  (total_volume * water_parts) / (water_parts + juice_parts)

/-- Theorem stating the amount of water needed for the specific punch recipe -/
theorem punch_water_calculation :
  water_needed 3 5 3 = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_punch_water_calculation_l1925_192505


namespace NUMINAMATH_CALUDE_triangle_centroid_inequality_locus_is_circle_l1925_192554

open Real

-- Define a triangle with vertices A, B, C and centroid G
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ
  is_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Define distance squared between two points
def dist_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define the theorem
theorem triangle_centroid_inequality (t : Triangle) (M : ℝ × ℝ) :
  dist_sq M t.A + dist_sq M t.B + dist_sq M t.C ≥ 
  dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C ∧
  (dist_sq M t.A + dist_sq M t.B + dist_sq M t.C = 
   dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C ↔ M = t.G) :=
sorry

-- Define the locus of points
def locus (t : Triangle) (k : ℝ) : Set (ℝ × ℝ) :=
  {M | dist_sq M t.A + dist_sq M t.B + dist_sq M t.C = k}

-- Define the theorem for the locus
theorem locus_is_circle (t : Triangle) (k : ℝ) 
  (h : k > dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C) :
  ∃ (r : ℝ), r > 0 ∧ locus t k = {M | dist_sq M t.G = r^2} ∧
  r^2 = (k - (dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C)) / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_centroid_inequality_locus_is_circle_l1925_192554


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1925_192587

theorem polygon_sides_count (n : ℕ) : 
  (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l1925_192587


namespace NUMINAMATH_CALUDE_fraction_halfway_between_one_sixth_and_one_fourth_l1925_192577

theorem fraction_halfway_between_one_sixth_and_one_fourth :
  let a := (1 : ℚ) / 6
  let b := (1 : ℚ) / 4
  (a + b) / 2 = 5 / 24 := by sorry

end NUMINAMATH_CALUDE_fraction_halfway_between_one_sixth_and_one_fourth_l1925_192577


namespace NUMINAMATH_CALUDE_two_car_efficiency_l1925_192578

/-- Two-car family fuel efficiency problem -/
theorem two_car_efficiency (mpg1 : ℝ) (total_miles : ℝ) (total_gallons : ℝ) (gallons1 : ℝ) :
  mpg1 = 25 →
  total_miles = 1825 →
  total_gallons = 55 →
  gallons1 = 30 →
  (total_miles - mpg1 * gallons1) / (total_gallons - gallons1) = 43 := by
sorry

end NUMINAMATH_CALUDE_two_car_efficiency_l1925_192578


namespace NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l1925_192593

theorem students_in_both_band_and_chorus 
  (total_students : ℕ) 
  (band_students : ℕ) 
  (chorus_students : ℕ) 
  (either_band_or_chorus : ℕ) 
  (h1 : total_students = 300) 
  (h2 : band_students = 150) 
  (h3 : chorus_students = 180) 
  (h4 : either_band_or_chorus = 250) : 
  band_students + chorus_students - either_band_or_chorus = 80 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l1925_192593


namespace NUMINAMATH_CALUDE_power_division_sum_equality_l1925_192594

theorem power_division_sum_equality : (-6)^5 / 6^2 + 4^3 - 7^2 = -201 := by
  sorry

end NUMINAMATH_CALUDE_power_division_sum_equality_l1925_192594


namespace NUMINAMATH_CALUDE_height_difference_climbing_l1925_192506

/-- Proves that the difference in height climbed between two people with different climbing rates over a given time is equal to the product of the time and the difference in their climbing rates. -/
theorem height_difference_climbing (matt_rate jason_rate : ℝ) (time : ℝ) 
  (h1 : matt_rate = 6)
  (h2 : jason_rate = 12)
  (h3 : time = 7) :
  jason_rate * time - matt_rate * time = (jason_rate - matt_rate) * time :=
by sorry

/-- Calculates the actual height difference between Jason and Matt after 7 minutes of climbing. -/
def actual_height_difference (matt_rate jason_rate : ℝ) (time : ℝ) 
  (h1 : matt_rate = 6)
  (h2 : jason_rate = 12)
  (h3 : time = 7) : ℝ :=
jason_rate * time - matt_rate * time

#eval actual_height_difference 6 12 7 rfl rfl rfl

end NUMINAMATH_CALUDE_height_difference_climbing_l1925_192506


namespace NUMINAMATH_CALUDE_folder_cost_l1925_192598

/-- The cost of office supplies problem -/
theorem folder_cost (pencil_cost : ℚ) (pencil_count : ℕ) (folder_count : ℕ) (total_cost : ℚ) : 
  pencil_cost = 1/2 →
  pencil_count = 24 →
  folder_count = 20 →
  total_cost = 30 →
  (total_cost - pencil_cost * pencil_count) / folder_count = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_folder_cost_l1925_192598


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l1925_192519

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 5

def contains_2_and_5 (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 5 ∈ n.digits 10

theorem smallest_valid_number_last_four_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 6 = 0 ∧
    m % 5 = 0 ∧
    is_valid_number m ∧
    contains_2_and_5 m ∧
    (∀ n : ℕ, n > 0 → n % 6 = 0 → n % 5 = 0 → is_valid_number n → contains_2_and_5 n → m ≤ n) ∧
    m % 10000 = 5220 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l1925_192519


namespace NUMINAMATH_CALUDE_largest_four_digit_square_base7_l1925_192596

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem largest_four_digit_square_base7 :
  (M * M ≥ 7^3) ∧ 
  (M * M < 7^4) ∧ 
  (∀ n : ℕ, n > M → n * n ≥ 7^4) ∧
  (toBase7 M = [6, 6]) := by sorry

end NUMINAMATH_CALUDE_largest_four_digit_square_base7_l1925_192596


namespace NUMINAMATH_CALUDE_opposite_silver_is_orange_l1925_192529

/-- Represents the colors of the cube faces -/
inductive Color
  | Blue
  | Orange
  | Black
  | Yellow
  | Silver
  | Violet

/-- Represents the positions of the cube faces -/
inductive Position
  | Top
  | Bottom
  | Front
  | Back
  | Left
  | Right

/-- Represents a view of the cube -/
structure View where
  top : Color
  front : Color
  right : Color

/-- The cube with its colored faces -/
structure Cube where
  faces : Position → Color

def first_view : View :=
  { top := Color.Blue, front := Color.Yellow, right := Color.Violet }

def second_view : View :=
  { top := Color.Blue, front := Color.Silver, right := Color.Violet }

def third_view : View :=
  { top := Color.Blue, front := Color.Black, right := Color.Violet }

theorem opposite_silver_is_orange (c : Cube) :
  (c.faces Position.Front = Color.Silver) →
  (c.faces Position.Top = Color.Blue) →
  (c.faces Position.Right = Color.Violet) →
  (c.faces Position.Back = Color.Orange) :=
by sorry

end NUMINAMATH_CALUDE_opposite_silver_is_orange_l1925_192529


namespace NUMINAMATH_CALUDE_crayon_ratio_l1925_192513

def billies_crayons : ℕ := 18
def bobbies_crayons : ℕ := 3 * billies_crayons
def lizzies_crayons : ℕ := 27

theorem crayon_ratio : 
  (lizzies_crayons : ℚ) / (bobbies_crayons : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_crayon_ratio_l1925_192513


namespace NUMINAMATH_CALUDE_height_of_column_G_l1925_192552

-- Define the regular octagon vertices
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (0, -8)
def D : ℝ × ℝ := (-8, 0)
def G : ℝ × ℝ := (0, 8)

-- Define the heights of columns A, B, C, D
def height_A : ℝ := 15
def height_B : ℝ := 12
def height_C : ℝ := 14
def height_D : ℝ := 13

-- Theorem statement
theorem height_of_column_G : 
  ∃ (height_G : ℝ), height_G = 15.5 :=
by
  sorry

end NUMINAMATH_CALUDE_height_of_column_G_l1925_192552


namespace NUMINAMATH_CALUDE_factorial_ratio_l1925_192592

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1925_192592


namespace NUMINAMATH_CALUDE_bob_first_six_probability_l1925_192581

/-- The probability of tossing a six on a fair die -/
def probSix : ℚ := 1 / 6

/-- The probability of not tossing a six on a fair die -/
def probNotSix : ℚ := 1 - probSix

/-- The order of players: Alice, Charlie, Bob -/
inductive Player : Type
| Alice : Player
| Charlie : Player
| Bob : Player

/-- The probability that Bob is the first to toss a six in the die-tossing game -/
def probBobFirstSix : ℚ := 25 / 91

theorem bob_first_six_probability :
  probBobFirstSix = (probNotSix * probNotSix * probSix) / (1 - probNotSix * probNotSix * probNotSix) :=
by sorry

end NUMINAMATH_CALUDE_bob_first_six_probability_l1925_192581


namespace NUMINAMATH_CALUDE_equation_solution_l1925_192589

theorem equation_solution : 
  {x : ℝ | x * (x - 14) = 0} = {0, 14} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1925_192589


namespace NUMINAMATH_CALUDE_piggy_bank_coins_l1925_192535

theorem piggy_bank_coins (coins : Fin 6 → ℕ) : 
  coins 2 = 81 ∧ 
  coins 3 = 90 ∧ 
  coins 4 = 99 ∧ 
  coins 5 = 108 ∧ 
  coins 6 = 117 ∧ 
  (∃ d : ℕ, ∀ i : Fin 5, coins (i + 1) = coins i + d) → 
  coins 1 = 72 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_coins_l1925_192535


namespace NUMINAMATH_CALUDE_campers_rowing_difference_l1925_192555

theorem campers_rowing_difference (morning afternoon evening : ℕ) 
  (h1 : morning = 33) 
  (h2 : afternoon = 34) 
  (h3 : evening = 10) : 
  afternoon - evening = 24 := by
sorry

end NUMINAMATH_CALUDE_campers_rowing_difference_l1925_192555


namespace NUMINAMATH_CALUDE_line_exists_l1925_192544

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The line l1: x + 5y - 5 = 0 -/
def line_l1 (x y : ℝ) : Prop := x + 5*y - 5 = 0

/-- The line l: 25x - 5y - 21 = 0 -/
def line_l (x y : ℝ) : Prop := 25*x - 5*y - 21 = 0

/-- Two points are distinct -/
def distinct (x1 y1 x2 y2 : ℝ) : Prop := x1 ≠ x2 ∨ y1 ≠ y2

/-- A line perpendicularly bisects a segment -/
def perpendicularly_bisects (x1 y1 x2 y2 : ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (xm ym : ℝ), line xm ym ∧ 
    xm = (x1 + x2) / 2 ∧ 
    ym = (y1 + y2) / 2 ∧
    (y2 - y1) * (x2 - xm) = (x2 - x1) * (y2 - ym)

theorem line_exists : ∃ (x1 y1 x2 y2 : ℝ),
  parabola x1 y1 ∧ parabola x2 y2 ∧
  line_l x1 y1 ∧ line_l x2 y2 ∧
  distinct x1 y1 x2 y2 ∧
  perpendicularly_bisects x1 y1 x2 y2 line_l1 :=
sorry

end NUMINAMATH_CALUDE_line_exists_l1925_192544


namespace NUMINAMATH_CALUDE_colored_paper_purchase_l1925_192539

theorem colored_paper_purchase (total_money : ℝ) (pencil_cost : ℝ) (paper_cost : ℝ) (pencils_bought : ℕ) :
  total_money = 10 →
  pencil_cost = 1.2 →
  paper_cost = 0.2 →
  pencils_bought = 5 →
  (total_money - pencil_cost * (pencils_bought : ℝ)) / paper_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_colored_paper_purchase_l1925_192539


namespace NUMINAMATH_CALUDE_nine_digit_divisible_by_101_l1925_192546

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ | 100 ≤ n ∧ n < 1000 }

/-- Converts a three-digit number to a nine-digit number by repeating it three times -/
def toNineDigitNumber (n : ThreeDigitNumber) : ℕ :=
  1000000 * n + 1000 * n + n

/-- Theorem: Any nine-digit number formed by repeating a three-digit number three times is divisible by 101 -/
theorem nine_digit_divisible_by_101 (n : ThreeDigitNumber) :
  ∃ k : ℕ, toNineDigitNumber n = 101 * k := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_divisible_by_101_l1925_192546


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l1925_192523

theorem wire_length_around_square_field (area : ℝ) (n : ℕ) :
  area = 24336 ∧ n = 13 →
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  n * perimeter = 8112 :=
by sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l1925_192523


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1925_192524

def polynomial (x : ℤ) : ℤ := x^3 + 2*x^2 - 3*x - 17

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = {-17, -1, 1, 17} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1925_192524


namespace NUMINAMATH_CALUDE_clay_transformation_in_two_operations_l1925_192520

/-- Represents a collection of clay pieces -/
structure ClayCollection where
  pieces : List Nat
  deriving Repr

/-- Represents an operation on clay pieces -/
def combine_operation (c : ClayCollection) (group_size : Nat) : ClayCollection :=
  sorry

/-- The initial state of clay pieces -/
def initial_state : ClayCollection :=
  { pieces := List.replicate 111 1 }

/-- The desired final state of clay pieces -/
def final_state : ClayCollection :=
  { pieces := [1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16] }

/-- Theorem stating that the transformation is possible in 2 operations -/
theorem clay_transformation_in_two_operations :
  ∃ (op1 op2 : Nat),
    (combine_operation (combine_operation initial_state op1) op2) = final_state :=
  sorry

end NUMINAMATH_CALUDE_clay_transformation_in_two_operations_l1925_192520


namespace NUMINAMATH_CALUDE_f_odd_f_increasing_f_odd_and_increasing_l1925_192556

/-- The function f(x) = x|x| -/
def f (x : ℝ) : ℝ := x * abs x

/-- f is an odd function -/
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

/-- f is an increasing function -/
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

/-- f is both odd and increasing -/
theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_f_odd_f_increasing_f_odd_and_increasing_l1925_192556


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l1925_192503

theorem sum_of_solutions_squared_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 8)^2 = 64 ∧ (b - 8)^2 = 64 ∧ a + b = 16) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l1925_192503


namespace NUMINAMATH_CALUDE_rectangle_area_l1925_192559

/-- Theorem: For a rectangle EFGH with vertices E(0, 0), F(0, y), G(y, 3y), and H(y, 0),
    where y > 0 and the area of the rectangle is 45 square units, the value of y is √15. -/
theorem rectangle_area (y : ℝ) (h1 : y > 0) (h2 : y * 3 * y = 45) : y = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1925_192559


namespace NUMINAMATH_CALUDE_paperclip_production_l1925_192549

theorem paperclip_production (machines_base : ℕ) (paperclips_per_minute : ℕ) (machines : ℕ) (minutes : ℕ) :
  machines_base = 8 →
  paperclips_per_minute = 560 →
  machines = 18 →
  minutes = 6 →
  (machines * paperclips_per_minute * minutes) / machines_base = 7560 := by
  sorry

end NUMINAMATH_CALUDE_paperclip_production_l1925_192549


namespace NUMINAMATH_CALUDE_chloes_candies_l1925_192521

/-- Given that Linda has 34 candies and the total number of candies is 62,
    prove that Chloe has 28 candies. -/
theorem chloes_candies (linda_candies : ℕ) (total_candies : ℕ) (chloe_candies : ℕ) : 
  linda_candies = 34 → total_candies = 62 → chloe_candies = total_candies - linda_candies →
  chloe_candies = 28 := by
  sorry

end NUMINAMATH_CALUDE_chloes_candies_l1925_192521


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1925_192573

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x * f y - f (x * y)) / 5 = x + y + 3

/-- The main theorem stating that the function f(x) = x + 4 satisfies the functional equation -/
theorem functional_equation_solution :
  ∃ f : ℝ → ℝ, FunctionalEquation f ∧ ∀ x : ℝ, f x = x + 4 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1925_192573


namespace NUMINAMATH_CALUDE_min_club_members_l1925_192528

theorem min_club_members (N : ℕ) : N < 80 ∧ 
  ((N - 5) % 8 = 0 ∨ (N - 5) % 7 = 0) ∧ 
  N % 9 = 7 → 
  N ≥ 61 :=
by sorry

end NUMINAMATH_CALUDE_min_club_members_l1925_192528


namespace NUMINAMATH_CALUDE_mosaic_completion_time_l1925_192575

-- Define the start time
def start_time : ℕ := 9 * 60  -- 9:00 AM in minutes since midnight

-- Define the time when 1/4 of the mosaic is completed
def quarter_time : ℕ := (12 + 12) * 60 + 45  -- 12:45 PM in minutes since midnight

-- Define the fraction of work completed
def fraction_completed : ℚ := 1/4

-- Define the duration to complete 1/4 of the mosaic
def quarter_duration : ℕ := quarter_time - start_time

-- Theorem to prove
theorem mosaic_completion_time :
  let total_duration : ℕ := (quarter_duration * 4)
  let finish_time : ℕ := (start_time + total_duration) % (24 * 60)
  finish_time = 0  -- 0 minutes past midnight (12:00 AM)
  := by sorry

end NUMINAMATH_CALUDE_mosaic_completion_time_l1925_192575


namespace NUMINAMATH_CALUDE_set_union_problem_l1925_192538

theorem set_union_problem (a b : ℝ) :
  let A : Set ℝ := {-1, a}
  let B : Set ℝ := {2^a, b}
  A ∩ B = {1} → A ∪ B = {-1, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l1925_192538


namespace NUMINAMATH_CALUDE_sqrt_fraction_equivalence_l1925_192537

theorem sqrt_fraction_equivalence (x : ℝ) (h : x < -2) :
  Real.sqrt (x / (1 + (x + 1) / (x + 2))) = -x := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equivalence_l1925_192537


namespace NUMINAMATH_CALUDE_certain_number_proof_l1925_192576

theorem certain_number_proof (N x : ℝ) (h1 : x = 0.1) (h2 : N / x * 2 = 12) : N = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1925_192576


namespace NUMINAMATH_CALUDE_fraction_subtraction_property_l1925_192518

theorem fraction_subtraction_property (a b n : ℕ) (h1 : b > a) (h2 : a > 0) 
  (h3 : ∀ k : ℕ, k > 0 → (1 : ℚ) / k ≤ a / b → k ≥ n) : a * n - b < a := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_property_l1925_192518


namespace NUMINAMATH_CALUDE_y_intercept_of_specific_line_l1925_192547

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ := sorry

/-- Given a line with slope 3 and x-intercept (-3, 0), its y-intercept is (0, 9). -/
theorem y_intercept_of_specific_line :
  let l : Line := { slope := 3, x_intercept := (-3, 0) }
  y_intercept l = (0, 9) := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_specific_line_l1925_192547


namespace NUMINAMATH_CALUDE_factorization_problems_l1925_192562

variable (m a b : ℝ)

theorem factorization_problems :
  (ma^2 - mb^2 = m*(a+b)*(a-b)) ∧
  ((a+b) - 2*a*(a+b) + a^2*(a+b) = (a+b)*(a-1)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l1925_192562


namespace NUMINAMATH_CALUDE_point_division_and_linear_combination_l1925_192510

/-- Given a line segment AB and a point P on it, prove that P divides AB in the ratio 4:1 
    and can be expressed as a linear combination of A and B -/
theorem point_division_and_linear_combination (A B P : ℝ × ℝ) : 
  A = (1, 2) →
  B = (4, 3) →
  (P.1 - A.1) / (B.1 - P.1) = 4 →
  (P.2 - A.2) / (B.2 - P.2) = 4 →
  ∃ (t u : ℝ), P = (t * A.1 + u * B.1, t * A.2 + u * B.2) ∧ t = 1/5 ∧ u = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_point_division_and_linear_combination_l1925_192510


namespace NUMINAMATH_CALUDE_vector_form_equiv_line_equation_l1925_192567

/-- The line equation y = 2x + 5 -/
def line_equation (x y : ℝ) : Prop := y = 2 * x + 5

/-- The vector form of the line -/
def vector_form (r k t x y : ℝ) : Prop :=
  x = r + 3 * t ∧ y = -3 + k * t

/-- Theorem stating that the vector form represents the line y = 2x + 5 
    if and only if r = -4 and k = 6 -/
theorem vector_form_equiv_line_equation :
  ∀ r k : ℝ, (∀ t x y : ℝ, vector_form r k t x y → line_equation x y) ∧
             (∀ x y : ℝ, line_equation x y → ∃ t : ℝ, vector_form r k t x y) ↔
  r = -4 ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_form_equiv_line_equation_l1925_192567


namespace NUMINAMATH_CALUDE_angle_sum_less_than_three_halves_pi_l1925_192583

theorem angle_sum_less_than_three_halves_pi
  (α β : Real)
  (h1 : π / 2 < α ∧ α < π)
  (h2 : π / 2 < β ∧ β < π)
  (h3 : Real.tan α < Real.tan (π / 2 - β)) :
  α + β < 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_less_than_three_halves_pi_l1925_192583


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1925_192548

theorem final_sum_after_operations (S a b : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l1925_192548


namespace NUMINAMATH_CALUDE_linear_function_quadrant_l1925_192515

theorem linear_function_quadrant (m : ℤ) : 
  (∀ x y : ℝ, y = (m + 4) * x + (m + 2) → ¬(x < 0 ∧ y > 0)) →
  (m = -3 ∨ m = -2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrant_l1925_192515


namespace NUMINAMATH_CALUDE_goods_train_length_l1925_192565

/-- Calculate the length of a goods train given the speeds of two trains moving in opposite directions and the time taken for the goods train to pass an observer in the other train. -/
theorem goods_train_length (man_train_speed goods_train_speed : ℝ) (passing_time : ℝ) :
  man_train_speed = 30 →
  goods_train_speed = 82 →
  passing_time = 9 →
  ∃ (length : ℝ), abs (length - 279.99) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_goods_train_length_l1925_192565


namespace NUMINAMATH_CALUDE_dans_age_l1925_192588

theorem dans_age (x : ℕ) : (x + 16 = 4 * (x - 8)) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_dans_age_l1925_192588


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l1925_192507

/-- The wood measurement problem from "The Mathematical Classic of Sunzi" -/
theorem sunzi_wood_measurement_problem (x y : ℝ) : 
  (y - x = 4.5 ∧ y / 2 = x - 1) ↔ 
  (∃ (rope_length wood_length : ℝ),
    rope_length > wood_length ∧
    rope_length - wood_length = 4.5 ∧
    rope_length / 2 > wood_length - 1 ∧
    rope_length / 2 < wood_length) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l1925_192507


namespace NUMINAMATH_CALUDE_scaling_transformation_correct_l1925_192584

/-- Scaling transformation function -/
def scale (sx sy : ℚ) (p : ℚ × ℚ) : ℚ × ℚ :=
  (sx * p.1, sy * p.2)

/-- The initial point -/
def initial_point : ℚ × ℚ := (1, 2)

/-- The scaling factors -/
def sx : ℚ := 1/2
def sy : ℚ := 1/3

/-- The expected result after transformation -/
def expected_result : ℚ × ℚ := (1/2, 2/3)

theorem scaling_transformation_correct :
  scale sx sy initial_point = expected_result := by
  sorry

end NUMINAMATH_CALUDE_scaling_transformation_correct_l1925_192584


namespace NUMINAMATH_CALUDE_incorrect_average_theorem_l1925_192508

def incorrect_average (n : ℕ) (correct_avg : ℚ) (correct_num wrong_num : ℚ) : ℚ :=
  (n * correct_avg - correct_num + wrong_num) / n

theorem incorrect_average_theorem :
  incorrect_average 10 24 76 26 = 19 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_average_theorem_l1925_192508


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l1925_192553

theorem circle_equation_k_value :
  ∀ k : ℝ,
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 10*y - k = 0 ↔ (x + 4)^2 + (y + 5)^2 = 100) →
  k = 59 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l1925_192553


namespace NUMINAMATH_CALUDE_inverse_proportionality_l1925_192509

theorem inverse_proportionality (x y : ℝ) (P : ℝ) : 
  (x + y = 30) → (x - y = 12) → (x * y = P) → (3 * (P / 3) = 63) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l1925_192509


namespace NUMINAMATH_CALUDE_no_two_digit_integer_satisfies_conditions_l1925_192591

theorem no_two_digit_integer_satisfies_conditions : 
  ∀ n : ℕ, 10 ≤ n → n < 100 → 
  ¬(∃ (a b : ℕ), n = 10 * a + b ∧ a < 10 ∧ b < 10 ∧ 
    (n % (a + b) = 0) ∧ (n % (a^2 * b) = 0)) := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_integer_satisfies_conditions_l1925_192591


namespace NUMINAMATH_CALUDE_most_suitable_student_l1925_192517

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the average score and variances
def average_score : ℝ := 180

def variance (s : Student) : ℝ :=
  match s with
  | Student.A => 65
  | Student.B => 56.5
  | Student.C => 53
  | Student.D => 50.5

-- Define the suitability criterion
def more_suitable (s1 s2 : Student) : Prop :=
  variance s1 < variance s2

-- Theorem statement
theorem most_suitable_student :
  ∀ s : Student, s ≠ Student.D → more_suitable Student.D s :=
sorry

end NUMINAMATH_CALUDE_most_suitable_student_l1925_192517


namespace NUMINAMATH_CALUDE_ratio_problem_l1925_192525

theorem ratio_problem (first_part : ℝ) (percent : ℝ) (second_part : ℝ) : 
  first_part = 4 →
  percent = 20 →
  first_part / second_part = percent / 100 →
  second_part = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1925_192525


namespace NUMINAMATH_CALUDE_binomial_50_2_l1925_192514

theorem binomial_50_2 : Nat.choose 50 2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_binomial_50_2_l1925_192514


namespace NUMINAMATH_CALUDE_problem_statement_l1925_192532

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 3 * a * (b^2 - 1) = b * (1 - a^2)) : 
  (1 / a + 3 / b = a + 3 * b) ∧ 
  (a^(3/2) * b^(1/2) + 3 * a^(1/2) * b^(3/2) ≥ 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1925_192532


namespace NUMINAMATH_CALUDE_color_change_probability_l1925_192574

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of 5-second intervals where a color change occurs -/
def colorChangeIntervals (cycle : TrafficLightCycle) : ℕ := 3

/-- Represents the duration of the observation interval -/
def observationInterval : ℕ := 5

/-- Theorem: The probability of observing a color change is 3/20 -/
theorem color_change_probability (cycle : TrafficLightCycle)
    (h1 : cycle.green = 45)
    (h2 : cycle.yellow = 5)
    (h3 : cycle.red = 50) :
    (colorChangeIntervals cycle : ℚ) * observationInterval / (cycleDuration cycle) = 3 / 20 := by
  sorry


end NUMINAMATH_CALUDE_color_change_probability_l1925_192574


namespace NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l1925_192512

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_tangent_line 
  (a b : ℝ) 
  (h1 : f' a b 1 = 0) 
  (h2 : f' a b 2 = 0) :
  (a = -3 ∧ b = 4) ∧ 
  (∃ (k m : ℝ), k = 12 ∧ m = 8 ∧ ∀ (x y : ℝ), y = k * x + m ↔ y = (f' (-3) 4 0) * x + f (-3) 4 0) := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l1925_192512


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1925_192563

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point (given_line : Line) (p : Point) :
  ∃ (result_line : Line),
    pointOnLine p result_line ∧
    parallel result_line given_line ∧
    result_line.a = 2 ∧
    result_line.b = 1 ∧
    result_line.c = -1 :=
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1925_192563


namespace NUMINAMATH_CALUDE_steve_total_cost_is_23_56_l1925_192536

/-- Calculates the total cost of Steve's DVD purchase --/
def steveTotalCost (mikeDVDPrice baseShippingRate salesTaxRate discountRate : ℚ) : ℚ :=
  let steveDVDPrice := 2 * mikeDVDPrice
  let otherDVDPrice := 7
  let subtotalBeforePromo := otherDVDPrice + otherDVDPrice
  let shippingCost := baseShippingRate * subtotalBeforePromo
  let subtotalWithShipping := subtotalBeforePromo + shippingCost
  let salesTax := salesTaxRate * subtotalWithShipping
  let subtotalWithTax := subtotalWithShipping + salesTax
  let discount := discountRate * subtotalWithTax
  subtotalWithTax - discount

/-- Theorem stating that Steve's total cost is $23.56 --/
theorem steve_total_cost_is_23_56 :
  steveTotalCost 5 0.8 0.1 0.15 = 23.56 := by
  sorry

end NUMINAMATH_CALUDE_steve_total_cost_is_23_56_l1925_192536


namespace NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l1925_192561

theorem tangent_line_to_logarithmic_curve (a : ℝ) :
  (∃ x₀ y₀ : ℝ, 
    y₀ = x₀ + 1 ∧ 
    y₀ = Real.log (x₀ + a) ∧
    (1 : ℝ) = 1 / (x₀ + a)) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l1925_192561


namespace NUMINAMATH_CALUDE_power_of_square_l1925_192569

theorem power_of_square (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l1925_192569


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_equals_one_max_a_for_positive_f_main_result_l1925_192586

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - a * x / (x + 1)

-- Theorem for part (I)
theorem local_minimum_implies_a_equals_one (a : ℝ) :
  (∀ x, x > -1 → f a x ≥ f a 0) → a = 1 := by sorry

-- Theorem for part (II)
theorem max_a_for_positive_f (a : ℝ) :
  (∀ x, x > 0 → f a x > 0) → a ≤ 1 := by sorry

-- Theorem combining both parts
theorem main_result :
  (∃ a : ℝ, (∀ x, x > -1 → f a x ≥ f a 0) ∧ 
   (∀ a', (∀ x, x > 0 → f a' x > 0) → a' ≤ a)) ∧
  (∃ a : ℝ, a = 1 ∧ (∀ x, x > -1 → f a x ≥ f a 0) ∧ 
   (∀ a', (∀ x, x > 0 → f a' x > 0) → a' ≤ a)) := by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_a_equals_one_max_a_for_positive_f_main_result_l1925_192586


namespace NUMINAMATH_CALUDE_sqrt_simplification_l1925_192545

theorem sqrt_simplification (x : ℝ) :
  1 + x ≥ 0 → -1 - x ≥ 0 → Real.sqrt (1 + x) - Real.sqrt (-1 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l1925_192545


namespace NUMINAMATH_CALUDE_prop_P_implies_t_range_prop_P_sufficient_not_necessary_implies_a_range_l1925_192502

/-- Represents the curve equation -/
def curve_equation (x y t : ℝ) : Prop :=
  x^2 / (4 - t) + y^2 / (t - 1) = 1

/-- Predicate for the curve being an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (t : ℝ) : Prop :=
  ∃ x y : ℝ, curve_equation x y t

/-- The inequality involving t and a -/
def inequality (t a : ℝ) : Prop :=
  t^2 - (a + 3) * t + (a + 2) < 0

/-- Proposition P implies the range of t -/
theorem prop_P_implies_t_range :
  ∀ t : ℝ, is_ellipse_x_foci t → 1 < t ∧ t < 5/2 :=
sorry

/-- Proposition P is sufficient but not necessary for Q implies the range of a -/
theorem prop_P_sufficient_not_necessary_implies_a_range :
  (∀ t a : ℝ, is_ellipse_x_foci t → inequality t a) ∧
  (∃ t a : ℝ, inequality t a ∧ ¬is_ellipse_x_foci t) →
  ∀ a : ℝ, a > 1/2 :=
sorry

end NUMINAMATH_CALUDE_prop_P_implies_t_range_prop_P_sufficient_not_necessary_implies_a_range_l1925_192502


namespace NUMINAMATH_CALUDE_chess_group_age_sum_l1925_192522

theorem chess_group_age_sum : 
  ∀ (a b : ℕ),
  (a^2 + (a+2)^2 + (a+4)^2 + (a+6)^2 + b^2 + (b+2)^2 = 2796) →
  (a + (a+2) + (a+4) + (a+6) + b + (b+2) = 94) :=
by sorry

end NUMINAMATH_CALUDE_chess_group_age_sum_l1925_192522


namespace NUMINAMATH_CALUDE_bird_watching_ratio_l1925_192542

theorem bird_watching_ratio (cardinals robins blue_jays sparrows : ℕ) : 
  cardinals = 3 →
  robins = 4 * cardinals →
  sparrows = 3 * cardinals + 1 →
  cardinals + robins + blue_jays + sparrows = 31 →
  blue_jays = 2 * cardinals :=
by
  sorry

end NUMINAMATH_CALUDE_bird_watching_ratio_l1925_192542


namespace NUMINAMATH_CALUDE_linear_regression_intercept_l1925_192511

/-- Linear regression model parameters -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Mean values of x and y -/
structure MeanValues where
  x_mean : ℝ
  y_mean : ℝ

/-- Theorem: Given a linear regression model and mean values, prove the intercept -/
theorem linear_regression_intercept 
  (model : LinearRegression) 
  (means : MeanValues) 
  (h_slope : model.slope = -12/5) 
  (h_x_mean : means.x_mean = -4) 
  (h_y_mean : means.y_mean = 25) : 
  model.intercept = 77/5 := by
  sorry

#check linear_regression_intercept

end NUMINAMATH_CALUDE_linear_regression_intercept_l1925_192511


namespace NUMINAMATH_CALUDE_circle_equation_l1925_192534

-- Define the point P
def P : ℝ × ℝ := (3, 1)

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x + 2 * y + 3 = 0
def l₂ (x y : ℝ) : Prop := x + 2 * y - 7 = 0

-- Define the two possible circle equations
def circle₁ (x y : ℝ) : Prop := (x - 4/5)^2 + (y - 3/5)^2 = 5
def circle₂ (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 5

-- Theorem statement
theorem circle_equation (x y : ℝ) :
  (∃ (c : ℝ × ℝ → Prop), c P ∧
    (∀ (x y : ℝ), l₁ x y → (∃ (t : ℝ), c (x, y) ∧ (∀ (ε : ℝ), ε ≠ 0 → ¬ c (x + ε, y + 2 * ε)))) ∧
    (∀ (x y : ℝ), l₂ x y → (∃ (t : ℝ), c (x, y) ∧ (∀ (ε : ℝ), ε ≠ 0 → ¬ c (x + ε, y + 2 * ε))))) →
  circle₁ x y ∨ circle₂ x y :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1925_192534


namespace NUMINAMATH_CALUDE_min_a_over_x_l1925_192501

theorem min_a_over_x (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100)
  (h : y^2 - 1 = a^2 * (x^2 - 1)) :
  ∀ k : ℚ, (k : ℝ) = a / x → k ≥ 2 ∧ ∃ a₀ x₀ y₀ : ℕ,
    a₀ > 100 ∧ x₀ > 100 ∧ y₀ > 100 ∧
    y₀^2 - 1 = a₀^2 * (x₀^2 - 1) ∧
    (a₀ : ℝ) / x₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_a_over_x_l1925_192501


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l1925_192597

theorem rectangular_prism_width 
  (l h d : ℝ) 
  (hl : l = 6) 
  (hh : h = 8) 
  (hd : d = 15) 
  (h_diagonal : d^2 = l^2 + w^2 + h^2) : 
  w = 5 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l1925_192597


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1925_192551

theorem arithmetic_calculation : 5 * 7 + 6 * 9 + 8 * 4 + 7 * 6 = 163 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1925_192551


namespace NUMINAMATH_CALUDE_wage_ratio_is_two_to_one_l1925_192504

/-- The ratio of a man's daily wage to a woman's daily wage -/
def wage_ratio (men_wage women_wage : ℚ) : ℚ := men_wage / women_wage

/-- The total earnings of a group of workers over a period -/
def total_earnings (num_workers : ℕ) (num_days : ℕ) (daily_wage : ℚ) : ℚ :=
  (num_workers : ℚ) * (num_days : ℚ) * daily_wage

theorem wage_ratio_is_two_to_one :
  ∃ (men_wage women_wage : ℚ),
    total_earnings 40 10 men_wage = 14400 ∧
    total_earnings 40 30 women_wage = 21600 ∧
    wage_ratio men_wage women_wage = 2 := by
  sorry

end NUMINAMATH_CALUDE_wage_ratio_is_two_to_one_l1925_192504


namespace NUMINAMATH_CALUDE_blackboard_numbers_l1925_192566

/-- The sum of reciprocals of the initial numbers on the blackboard -/
def initial_sum (m : ℕ) : ℚ := (2 * m) / (2 * m + 1)

/-- The operation performed in each move -/
def move (a b c : ℚ) : ℚ := (a * b * c) / (a * b + b * c + c * a)

theorem blackboard_numbers (m : ℕ) (h1 : m ≥ 2) :
  ∀ x : ℚ, 
    (∃ (nums : List ℚ), 
      (nums.length = 2) ∧ 
      (4/3 ∈ nums) ∧ 
      (x ∈ nums) ∧ 
      (1 / (4/3) + 1 / x = initial_sum m)) →
    x > 4 := by sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l1925_192566


namespace NUMINAMATH_CALUDE_expression_simplification_l1925_192560

theorem expression_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a + b ≠ 0) (h3 : a^3 - b^3 ≠ 0) :
  (3 * a^2 + 3 * a * b + 3 * b^2) / (4 * (a + b)) * (2 * a^2 - 2 * b^2) / (9 * (a^3 - b^3)) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1925_192560


namespace NUMINAMATH_CALUDE_f_919_equals_6_l1925_192599

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_six (f : ℝ → ℝ) : Prop := ∀ x, f (x + 6) = f x

theorem f_919_equals_6 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 4) = f (x - 2))
  (h3 : ∀ x ∈ Set.Icc (-3) 0, f x = (6 : ℝ) ^ (-x)) :
  f 919 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_919_equals_6_l1925_192599


namespace NUMINAMATH_CALUDE_micrometer_conversion_l1925_192558

-- Define the conversion factor from micrometers to meters
def micrometer_to_meter : ℝ := 1e-6

-- State the theorem
theorem micrometer_conversion :
  0.01 * micrometer_to_meter = 1e-8 := by
  sorry

end NUMINAMATH_CALUDE_micrometer_conversion_l1925_192558


namespace NUMINAMATH_CALUDE_log_inequality_l1925_192570

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x^2) < x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1925_192570


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1925_192527

/-- Given two lines m and n that intersect at (1, 6), where
    m has equation y = 4x + 2 and n has equation y = kx + 3,
    prove that k = 3. -/
theorem intersection_of_lines (k : ℝ) : 
  (∀ x y : ℝ, y = 4*x + 2 → (x = 1 ∧ y = 6)) →  -- line m passes through (1, 6)
  (∀ x y : ℝ, y = k*x + 3 → (x = 1 ∧ y = 6)) →  -- line n passes through (1, 6)
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1925_192527


namespace NUMINAMATH_CALUDE_certain_number_proof_l1925_192500

theorem certain_number_proof (x y : ℕ) : 
  x + y = 24 → 
  x = 11 → 
  x ≤ y → 
  7 * x + 5 * y = 142 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1925_192500


namespace NUMINAMATH_CALUDE_shaded_square_area_ratio_l1925_192540

theorem shaded_square_area_ratio : 
  let shaded_square_side : ℝ := Real.sqrt 2
  let grid_side : ℝ := 6
  (shaded_square_side ^ 2) / (grid_side ^ 2) = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_shaded_square_area_ratio_l1925_192540


namespace NUMINAMATH_CALUDE_remainder_sum_l1925_192550

theorem remainder_sum (n : ℤ) : n % 12 = 5 → (n % 3 + n % 4 = 3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1925_192550


namespace NUMINAMATH_CALUDE_f_is_even_l1925_192568

-- Define g as an even function
def g_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^2)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_even g) : ∀ x, f g (-x) = f g x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1925_192568


namespace NUMINAMATH_CALUDE_tim_tasks_per_day_l1925_192571

/-- The number of tasks Tim does per day -/
def tasks_per_day (days_per_week : ℕ) (weekly_earnings : ℚ) (pay_per_task : ℚ) : ℚ :=
  (weekly_earnings / days_per_week) / pay_per_task

/-- Proof that Tim does 100 tasks per day -/
theorem tim_tasks_per_day :
  tasks_per_day 6 720 1.2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_tim_tasks_per_day_l1925_192571


namespace NUMINAMATH_CALUDE_change_eight_dollars_theorem_l1925_192595

theorem change_eight_dollars_theorem :
  ∃ (n : ℕ), n > 0 ∧
  (∃ (combinations : List (ℕ × ℕ × ℕ)),
    combinations.length = n ∧
    ∀ (c : ℕ × ℕ × ℕ), c ∈ combinations →
      let (nickels, dimes, quarters) := c
      nickels > 0 ∧ dimes > 0 ∧ quarters > 0 ∧
      5 * nickels + 10 * dimes + 25 * quarters = 800) :=
by sorry

end NUMINAMATH_CALUDE_change_eight_dollars_theorem_l1925_192595


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_common_foci_l1925_192531

/-- The value of m for which the given ellipse and hyperbola share common foci -/
theorem ellipse_hyperbola_common_foci : ∃ m : ℝ,
  (∀ x y : ℝ, x^2 / 25 + y^2 / 16 = 1 → x^2 / m - y^2 / 5 = 1) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 →
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →
      ∃ c : ℝ, c^2 = a^2 - b^2 ∧ (x = c ∨ x = -c) ∧ y = 0)) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 →
    (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 →
      ∃ c : ℝ, c^2 = a^2 + b^2 ∧ (x = c ∨ x = -c) ∧ y = 0)) →
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_common_foci_l1925_192531


namespace NUMINAMATH_CALUDE_lasso_probability_l1925_192533

theorem lasso_probability (p : ℝ) (n : ℕ) (hp : p = 1 / 2) (hn : n = 4) :
  1 - (1 - p) ^ n = 15 / 16 :=
by sorry

end NUMINAMATH_CALUDE_lasso_probability_l1925_192533


namespace NUMINAMATH_CALUDE_circle_equation_l1925_192582

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y + 1 = 0
def y_axis (x : ℝ) : Prop := x = 0

-- State the theorem
theorem circle_equation (C : Circle) : 
  (∃ x y : ℝ, line1 x y ∧ y_axis x ∧ C.center = (x, y)) →  -- Center condition
  (∃ x y : ℝ, line2 x y ∧ (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2) →  -- Tangent condition
  ∀ x y : ℝ, (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2 ↔ x^2 + (y-1)^2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_circle_equation_l1925_192582


namespace NUMINAMATH_CALUDE_win_sector_area_l1925_192580

/-- The area of a WIN sector on a circular spinner --/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1925_192580
