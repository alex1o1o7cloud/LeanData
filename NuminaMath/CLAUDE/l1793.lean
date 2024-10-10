import Mathlib

namespace terminating_decimal_count_l1793_179342

/-- A fraction a/b has a terminating decimal representation if and only if
    the denominator b can be factored as 2^m * 5^n * d, where d is coprime to 10 -/
def has_terminating_decimal (a b : ℕ) : Prop := sorry

/-- Count of integers in a given range satisfying a property -/
def count_satisfying (lower upper : ℕ) (P : ℕ → Prop) : ℕ := sorry

theorem terminating_decimal_count :
  count_satisfying 1 508 (λ k => has_terminating_decimal k 425) = 29 := by sorry

end terminating_decimal_count_l1793_179342


namespace smallest_n_not_divisible_by_ten_l1793_179341

theorem smallest_n_not_divisible_by_ten (n : ℕ) :
  (n > 2016 ∧ ¬(10 ∣ (1^n + 2^n + 3^n + 4^n)) ∧
   ∀ m, 2016 < m ∧ m < n → (10 ∣ (1^m + 2^m + 3^m + 4^m))) →
  n = 2020 := by
sorry

end smallest_n_not_divisible_by_ten_l1793_179341


namespace total_yellow_marbles_l1793_179374

theorem total_yellow_marbles (mary joan peter : ℕ) 
  (h1 : mary = 9) 
  (h2 : joan = 3) 
  (h3 : peter = 7) : 
  mary + joan + peter = 19 := by
sorry

end total_yellow_marbles_l1793_179374


namespace specific_polyhedron_volume_l1793_179303

/-- A polyhedron formed by folding a flat figure -/
structure Polyhedron where
  /-- The number of equilateral triangles in the flat figure -/
  num_triangles : ℕ
  /-- The number of squares in the flat figure -/
  num_squares : ℕ
  /-- The side length of the squares -/
  square_side : ℝ
  /-- The number of regular hexagons in the flat figure -/
  num_hexagons : ℕ

/-- Calculate the volume of the polyhedron -/
def calculate_volume (p : Polyhedron) : ℝ :=
  sorry

/-- The theorem stating the volume of the specific polyhedron -/
theorem specific_polyhedron_volume :
  let p : Polyhedron := {
    num_triangles := 3,
    num_squares := 3,
    square_side := 2,
    num_hexagons := 1
  }
  calculate_volume p = 11 :=
sorry

end specific_polyhedron_volume_l1793_179303


namespace sufficient_but_not_necessary_condition_l1793_179353

theorem sufficient_but_not_necessary_condition (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m = n → m^2 = n^2) ∧ ¬(m^2 = n^2 → m = n) :=
by sorry

end sufficient_but_not_necessary_condition_l1793_179353


namespace min_value_theorem_l1793_179365

def vector_a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (4, y)

def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem min_value_theorem (x y : ℝ) :
  perpendicular (vector_a x) (vector_b y) →
  ∃ (min : ℝ), min = 6 ∧ ∀ (z : ℝ), 9^x + 3^y ≥ z := by
  sorry

end min_value_theorem_l1793_179365


namespace rosa_flower_count_l1793_179310

/-- Rosa's initial number of flowers -/
def initial_flowers : ℝ := 67.0

/-- Number of flowers Andre gave to Rosa -/
def additional_flowers : ℝ := 90.0

/-- Rosa's total number of flowers -/
def total_flowers : ℝ := initial_flowers + additional_flowers

theorem rosa_flower_count : total_flowers = 157.0 := by
  sorry

end rosa_flower_count_l1793_179310


namespace collinear_points_theorem_l1793_179339

/-- Four points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three vectors are collinear -/
def are_collinear (v1 v2 v3 : Point3D) : Prop :=
  ∃ (t1 t2 : ℝ), v2.x - v1.x = t1 * (v3.x - v1.x) ∧
                 v2.y - v1.y = t1 * (v3.y - v1.y) ∧
                 v2.z - v1.z = t1 * (v3.z - v1.z) ∧
                 t2 ≠ 0 ∧ t2 ≠ 1 ∧
                 v3.x - v1.x = t2 * (v2.x - v1.x) ∧
                 v3.y - v1.y = t2 * (v2.y - v1.y) ∧
                 v3.z - v1.z = t2 * (v2.z - v1.z)

theorem collinear_points_theorem (a c d : ℝ) :
  let p1 : Point3D := ⟨2, 0, a⟩
  let p2 : Point3D := ⟨2*a, 2, 0⟩
  let p3 : Point3D := ⟨0, c, 1⟩
  let p4 : Point3D := ⟨9*d, 9*d, -d⟩
  (are_collinear p1 p2 p3 ∧ are_collinear p1 p2 p4 ∧ are_collinear p1 p3 p4) →
  d = 1/9 :=
by sorry

end collinear_points_theorem_l1793_179339


namespace right_triangle_cosine_l1793_179327

/-- In a right triangle DEF where angle D is 90 degrees and sin E = 3/5, cos F = 3/5 -/
theorem right_triangle_cosine (D E F : ℝ) : 
  D = Real.pi / 2 → 
  Real.sin E = 3 / 5 → 
  Real.cos F = 3 / 5 := by
  sorry

end right_triangle_cosine_l1793_179327


namespace always_positive_expression_l1793_179384

theorem always_positive_expression (a : ℝ) : |a| + 2 > 0 := by
  sorry

end always_positive_expression_l1793_179384


namespace weight_of_b_l1793_179375

theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 47 →
  b = 39 := by
sorry

end weight_of_b_l1793_179375


namespace partition_of_naturals_l1793_179350

/-- The set of natural numbers starting from 1 -/
def ℕ' : Set ℕ := {n : ℕ | n ≥ 1}

/-- The set S(x, y) for real x and y -/
def S (x y : ℝ) : Set ℕ := {s : ℕ | ∃ n : ℕ, n ∈ ℕ' ∧ s = ⌊n * x + y⌋}

/-- The main theorem -/
theorem partition_of_naturals (r : ℚ) (hr : r > 1) :
  ∃ u v : ℝ, (S r 0 ∩ S u v = ∅) ∧ (S r 0 ∪ S u v = ℕ') := by
  sorry

end partition_of_naturals_l1793_179350


namespace polygon_sides_l1793_179358

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 360 * 3) : n = 8 := by
  sorry

end polygon_sides_l1793_179358


namespace cube_of_102_l1793_179334

theorem cube_of_102 : (100 + 2)^3 = 1061208 := by
  sorry

end cube_of_102_l1793_179334


namespace no_base_for_square_202_l1793_179328

-- Define the base-b representation of 202_b
def base_b_representation (b : ℕ) : ℕ := 2 * b^2 + 2

-- Define the property of being a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

-- Theorem statement
theorem no_base_for_square_202 :
  ∀ b : ℕ, b > 2 → ¬(is_perfect_square (base_b_representation b)) := by
  sorry

end no_base_for_square_202_l1793_179328


namespace smallest_integer_c_l1793_179372

theorem smallest_integer_c (x : ℕ) (h : x = 8 * 3) : 
  (∃ c : ℕ, 27 ^ c > 3 ^ x ∧ ∀ k : ℕ, 27 ^ k > 3 ^ x → c ≤ k) → 
  (∃ c : ℕ, 27 ^ c > 3 ^ x ∧ ∀ k : ℕ, 27 ^ k > 3 ^ x → c ≤ k) ∧ 
  (∀ c : ℕ, 27 ^ c > 3 ^ x ∧ (∀ k : ℕ, 27 ^ k > 3 ^ x → c ≤ k) → c = 9) :=
by sorry

end smallest_integer_c_l1793_179372


namespace largest_result_is_630_l1793_179344

-- Define the set of available digits
def Digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the allowed operations
inductive Operation
| Add : Operation
| Sub : Operation
| Mul : Operation
| Div : Operation

-- Define a sequence of operations
def OperationSequence := List (Operation × Nat)

-- Function to apply a sequence of operations
def applyOperations (seq : OperationSequence) : Nat :=
  sorry

-- Theorem stating that 630 is the largest possible result
theorem largest_result_is_630 :
  ∀ (seq : OperationSequence),
    (∀ n ∈ Digits, (seq.map Prod.snd).count n = 1) →
    applyOperations seq ≤ 630 :=
  sorry

end largest_result_is_630_l1793_179344


namespace number_equation_l1793_179396

theorem number_equation (x : ℝ) : 43 + 3 * x = 58 ↔ x = 5 := by
  sorry

end number_equation_l1793_179396


namespace chocolate_candy_difference_l1793_179302

-- Define the cost of chocolate and candy bar
def chocolate_cost : ℕ := 3
def candy_bar_cost : ℕ := 2

-- Theorem statement
theorem chocolate_candy_difference :
  chocolate_cost - candy_bar_cost = 1 := by
  sorry

end chocolate_candy_difference_l1793_179302


namespace divisibility_condition_l1793_179340

theorem divisibility_condition (n : ℕ) (hn : n ≥ 1) :
  (3^(n-1) + 5^(n-1)) ∣ (3^n + 5^n) ↔ n = 1 := by
  sorry

end divisibility_condition_l1793_179340


namespace horner_method_proof_l1793_179364

def horner_polynomial (x : ℝ) : ℝ := x * (x * (x * (x * (2 * x + 0) + 4) + 3) + 1)

theorem horner_method_proof :
  let f (x : ℝ) := 3 * x^2 + 2 * x^5 + 4 * x^3 + x
  f 3 = horner_polynomial 3 ∧ horner_polynomial 3 = 624 :=
by sorry

end horner_method_proof_l1793_179364


namespace factorization_1_factorization_2_l1793_179331

-- Part 1
theorem factorization_1 (x y : ℝ) : 
  (x - y)^2 - 4*(x - y) + 4 = (x - y - 2)^2 := by sorry

-- Part 2
theorem factorization_2 (a b : ℝ) : 
  (a^2 + b^2)^2 - 4*a^2*b^2 = (a - b)^2 * (a + b)^2 := by sorry

end factorization_1_factorization_2_l1793_179331


namespace ratio_from_linear_equation_l1793_179309

theorem ratio_from_linear_equation (x y : ℝ) (h : 2 * y - 5 * x = 0) :
  ∃ (k : ℝ), k > 0 ∧ x = 2 * k ∧ y = 5 * k :=
sorry

end ratio_from_linear_equation_l1793_179309


namespace tangent_line_at_origin_l1793_179347

/-- The equation of the tangent line to y = x^2 + x + 1/2 at (0, 1/2) is y = x + 1/2 -/
theorem tangent_line_at_origin (x : ℝ) :
  let f (x : ℝ) := x^2 + x + 1/2
  let f' (x : ℝ) := 2*x + 1
  let tangent_line (x : ℝ) := x + 1/2
  (∀ x, deriv f x = f' x) →
  (f 0 = 1/2) →
  (tangent_line 0 = 1/2) →
  (f' 0 = 1) →
  ∀ x, tangent_line x = f 0 + f' 0 * x :=
by sorry

end tangent_line_at_origin_l1793_179347


namespace f_composition_value_l1793_179346

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (-x) else Real.log x

theorem f_composition_value : f (f (1/3)) = 3 := by
  sorry

end f_composition_value_l1793_179346


namespace min_perimeter_nine_square_rectangle_l1793_179395

/-- Represents a rectangle divided into nine squares with integer side lengths -/
structure NineSquareRectangle where
  a : ℕ
  b : ℕ

/-- The perimeter of a NineSquareRectangle -/
def perimeter (r : NineSquareRectangle) : ℕ :=
  2 * ((3 * r.a + 8 * r.a) + (2 * r.a + 12 * r.a))

/-- Theorem stating the minimum perimeter of a NineSquareRectangle is 52 -/
theorem min_perimeter_nine_square_rectangle :
  ∃ (r : NineSquareRectangle), perimeter r = 52 ∧ ∀ (s : NineSquareRectangle), perimeter s ≥ 52 :=
sorry

end min_perimeter_nine_square_rectangle_l1793_179395


namespace tangent_line_equation_l1793_179366

/-- A line that passes through (3, 4) and is tangent to the circle x^2 + y^2 = 25 has the equation 3x + 4y - 25 = 0 -/
theorem tangent_line_equation :
  ∃! (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y + c = 0 → a * 3 + b * 4 + c = 0) ∧ 
    (∀ x y : ℝ, x^2 + y^2 = 25 → (a * x + b * y + c)^2 = (a^2 + b^2) * 25) ∧
    a = 3 ∧ b = 4 ∧ c = -25 :=
sorry

end tangent_line_equation_l1793_179366


namespace fraction_relationship_l1793_179326

theorem fraction_relationship (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 := by
  sorry

end fraction_relationship_l1793_179326


namespace age_difference_l1793_179352

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a = c + 18 := by
  sorry

end age_difference_l1793_179352


namespace base8_52_equals_base10_42_l1793_179322

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ :=
  let ones := n % 10
  let eights := n / 10
  eights * 8 + ones

/-- The base-8 number 52 is equal to the base-10 number 42 --/
theorem base8_52_equals_base10_42 : base8ToBase10 52 = 42 := by
  sorry

end base8_52_equals_base10_42_l1793_179322


namespace jose_initial_caps_l1793_179354

/-- The number of bottle caps Jose gave to Rebecca -/
def given_caps : ℕ := 2

/-- The number of bottle caps Jose has left -/
def remaining_caps : ℕ := 5

/-- The initial number of bottle caps Jose had -/
def initial_caps : ℕ := given_caps + remaining_caps

theorem jose_initial_caps : initial_caps = 7 := by
  sorry

end jose_initial_caps_l1793_179354


namespace bryan_bookshelves_l1793_179349

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 38 / 2

/-- The number of books per bookshelf -/
def books_per_shelf : ℕ := 2

/-- The total number of books -/
def total_books : ℕ := 38

theorem bryan_bookshelves : 
  (num_bookshelves * books_per_shelf = total_books) ∧ (num_bookshelves = 19) :=
by sorry

end bryan_bookshelves_l1793_179349


namespace quadratic_root_property_l1793_179304

theorem quadratic_root_property (m : ℝ) : m^2 - m - 3 = 0 → 2023 - m^2 + m = 2020 := by
  sorry

end quadratic_root_property_l1793_179304


namespace point_to_line_distance_l1793_179323

/-- The distance from a point to a line in 2D space -/
theorem point_to_line_distance
  (x₀ y₀ a b c : ℝ) (h : a^2 + b^2 ≠ 0) :
  let d := |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)
  ∀ x y, a * x + b * y + c = 0 → 
    d ≤ Real.sqrt ((x - x₀)^2 + (y - y₀)^2) :=
by sorry

end point_to_line_distance_l1793_179323


namespace g_sum_property_l1793_179311

/-- Given a function g(x) = ax^6 + bx^4 - cx^3 - cx^2 + 3, 
    if g(2) = 5, then g(2) + g(-2) = 10 -/
theorem g_sum_property (a b c : ℝ) : 
  let g := λ x : ℝ => a * x^6 + b * x^4 - c * x^3 - c * x^2 + 3
  (g 2 = 5) → (g 2 + g (-2) = 10) := by
  sorry

end g_sum_property_l1793_179311


namespace impossibility_theorem_l1793_179390

/-- Represents a pile of chips -/
structure Pile :=
  (chips : ℕ)

/-- Represents the state of all piles -/
def State := List Pile

/-- The i-th prime number -/
def ithPrime (i : ℕ) : ℕ := sorry

/-- Initial state with 2018 piles, where the i-th pile has p_i chips (p_i is the i-th prime) -/
def initialState : State := 
  List.range 2018 |>.map (fun i => Pile.mk (ithPrime (i + 1)))

/-- Splits a pile into two piles and adds one chip to one of the new piles -/
def splitPile (s : State) (i : ℕ) (j k : ℕ) : State := sorry

/-- Merges two piles and adds one chip to the resulting pile -/
def mergePiles (s : State) (i j : ℕ) : State := sorry

/-- The target state with 2018 piles, each containing 2018 chips -/
def targetState : State := 
  List.replicate 2018 (Pile.mk 2018)

/-- Predicate to check if a given state is reachable from the initial state -/
def isReachable (s : State) : Prop := sorry

theorem impossibility_theorem : ¬ isReachable targetState := by
  sorry

end impossibility_theorem_l1793_179390


namespace constant_avg_speed_not_imply_uniform_motion_l1793_179356

/-- A snail's motion over a time interval -/
structure SnailMotion where
  /-- The time interval in minutes -/
  interval : ℝ
  /-- The distance traveled in meters -/
  distance : ℝ
  /-- The average speed in meters per minute -/
  avg_speed : ℝ
  /-- Condition: The average speed is constant -/
  constant_avg_speed : avg_speed = distance / interval

/-- Definition of uniform motion -/
def is_uniform_motion (motion : SnailMotion) : Prop :=
  ∀ t : ℝ, 0 ≤ t → t ≤ motion.interval →
    motion.distance * (t / motion.interval) = motion.avg_speed * t

/-- Theorem: Constant average speed does not imply uniform motion -/
theorem constant_avg_speed_not_imply_uniform_motion :
  ∃ (motion : SnailMotion), ¬(is_uniform_motion motion) :=
sorry

end constant_avg_speed_not_imply_uniform_motion_l1793_179356


namespace fifteen_sided_polygon_diagonals_l1793_179362

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by sorry

end fifteen_sided_polygon_diagonals_l1793_179362


namespace equation_solutions_l1793_179394

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1^2 - 5*x1 - 6 = 0 ∧ x2^2 - 5*x2 - 6 = 0 ∧ x1 = 6 ∧ x2 = -1) ∧
  (∃ y1 y2 : ℝ, (y1 + 1)*(y1 - 1) + y1*(y1 + 2) = 7 + 6*y1 ∧
                (y2 + 1)*(y2 - 1) + y2*(y2 + 2) = 7 + 6*y2 ∧
                y1 = Real.sqrt 5 + 1 ∧ y2 = 1 - Real.sqrt 5) :=
by sorry

end equation_solutions_l1793_179394


namespace min_staff_members_theorem_l1793_179377

/-- Represents the seating arrangement in a school hall --/
structure SchoolHall where
  male_students : ℕ
  female_students : ℕ
  benches_3_seats : ℕ
  benches_4_seats : ℕ

/-- Calculates the minimum number of staff members required --/
def min_staff_members (hall : SchoolHall) : ℕ :=
  let total_students := hall.male_students + hall.female_students
  let total_seats := 3 * hall.benches_3_seats + 4 * hall.benches_4_seats
  max (total_students - total_seats) 0

/-- Theorem stating the minimum number of staff members required --/
theorem min_staff_members_theorem (hall : SchoolHall) : 
  hall.male_students = 29 ∧ 
  hall.female_students = 4 * hall.male_students ∧ 
  hall.benches_3_seats = 15 ∧ 
  hall.benches_4_seats = 14 →
  min_staff_members hall = 44 := by
  sorry

#eval min_staff_members {
  male_students := 29,
  female_students := 116,
  benches_3_seats := 15,
  benches_4_seats := 14
}

end min_staff_members_theorem_l1793_179377


namespace greatest_x_value_l1793_179343

theorem greatest_x_value (x : ℤ) (h : 2.134 * (10 : ℝ) ^ (x : ℝ) < 240000) :
  x ≤ 5 ∧ ∃ y : ℤ, y > 5 → 2.134 * (10 : ℝ) ^ (y : ℝ) ≥ 240000 :=
by sorry

end greatest_x_value_l1793_179343


namespace john_pens_difference_l1793_179338

theorem john_pens_difference (total_pens blue_pens : ℕ) 
  (h_total : total_pens = 31)
  (h_blue : blue_pens = 18)
  (h_black_twice_red : ∃ (black_pens red_pens : ℕ), 
    total_pens = blue_pens + black_pens + red_pens ∧
    blue_pens = 2 * black_pens ∧
    black_pens > red_pens) :
  ∃ (black_pens red_pens : ℕ), black_pens - red_pens = 5 := by
sorry

end john_pens_difference_l1793_179338


namespace ellipse_equation_l1793_179370

/-- An ellipse with foci on the x-axis, focal distance 2√6, passing through (√3, √2) -/
structure Ellipse where
  /-- Half the distance between the foci -/
  c : ℝ
  /-- Semi-major axis -/
  a : ℝ
  /-- Semi-minor axis -/
  b : ℝ
  /-- Focal distance is 2√6 -/
  h_focal_distance : c = Real.sqrt 6
  /-- a > b > 0 -/
  h_a_gt_b : a > b ∧ b > 0
  /-- c² = a² - b² -/
  h_c_squared : c^2 = a^2 - b^2
  /-- The ellipse passes through (√3, √2) -/
  h_point : 3 / a^2 + 2 / b^2 = 1

/-- The standard equation of the ellipse is x²/9 + y²/3 = 1 -/
theorem ellipse_equation (e : Ellipse) : e.a^2 = 9 ∧ e.b^2 = 3 := by
  sorry

#check ellipse_equation

end ellipse_equation_l1793_179370


namespace tesseract_simplex_ratio_l1793_179316

-- Define the vertices of the 4-simplex
def v₀ : Fin 4 → ℝ := λ _ => 0
def v₁ : Fin 4 → ℝ := λ i => if i.val < 2 then 1 else 0
def v₂ : Fin 4 → ℝ := λ i => if i.val = 0 ∨ i.val = 2 then 1 else 0
def v₃ : Fin 4 → ℝ := λ i => if i.val = 0 ∨ i.val = 3 then 1 else 0
def v₄ : Fin 4 → ℝ := λ i => if i.val > 0 then 1 else 0

-- Define the 4-simplex
def simplex : Fin 5 → (Fin 4 → ℝ) := λ i =>
  match i with
  | 0 => v₀
  | 1 => v₁
  | 2 => v₂
  | 3 => v₃
  | 4 => v₄

-- Define the hypervolume of a unit tesseract
def tesseract_hypervolume : ℝ := 1

-- Define the function to calculate the hypervolume of the 4-simplex
noncomputable def simplex_hypervolume : ℝ := sorry

-- State the theorem
theorem tesseract_simplex_ratio :
  tesseract_hypervolume / simplex_hypervolume = 24 / Real.sqrt 5 := by sorry

end tesseract_simplex_ratio_l1793_179316


namespace rose_orchid_difference_l1793_179330

/-- Given the initial and final counts of roses and orchids in a vase, 
    prove that there are 10 more roses than orchids in the final state. -/
theorem rose_orchid_difference :
  let initial_roses : ℕ := 5
  let initial_orchids : ℕ := 3
  let final_roses : ℕ := 12
  let final_orchids : ℕ := 2
  final_roses - final_orchids = 10 := by
  sorry

end rose_orchid_difference_l1793_179330


namespace simplified_win_ratio_l1793_179381

def chloe_wins : ℕ := 24
def max_wins : ℕ := 9

theorem simplified_win_ratio : 
  ∃ (a b : ℕ), a = 8 ∧ b = 3 ∧ chloe_wins * b = max_wins * a := by
  sorry

end simplified_win_ratio_l1793_179381


namespace married_couples_children_l1793_179386

/-- The fraction of married couples with more than one child -/
def fraction_more_than_one_child : ℚ := 3/5

/-- The fraction of married couples with more than 3 children -/
def fraction_more_than_three_children : ℚ := 2/5

/-- The fraction of married couples with 2 or 3 children -/
def fraction_two_or_three_children : ℚ := 1/5

theorem married_couples_children :
  fraction_more_than_one_child = 
    fraction_more_than_three_children + fraction_two_or_three_children :=
by sorry

end married_couples_children_l1793_179386


namespace kyles_money_after_snowboarding_l1793_179383

/-- Calculates Kyle's remaining money after snowboarding -/
def kyles_remaining_money (daves_money : ℕ) : ℕ :=
  let kyles_initial_money := 3 * daves_money - 12
  let snowboarding_cost := kyles_initial_money / 3
  kyles_initial_money - snowboarding_cost

/-- Proves that Kyle has $84 left after snowboarding -/
theorem kyles_money_after_snowboarding :
  kyles_remaining_money 46 = 84 := by
  sorry

#eval kyles_remaining_money 46

end kyles_money_after_snowboarding_l1793_179383


namespace vector_subtraction_l1793_179305

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_subtraction :
  (3 • b - a) = ![(-3 : ℝ), -5] := by sorry

end vector_subtraction_l1793_179305


namespace smallest_undefined_value_l1793_179389

theorem smallest_undefined_value (y : ℝ) : 
  (∀ z : ℝ, z < y → (z - 3) / (6 * z^2 - 37 * z + 6) ≠ 0) ∧ 
  ((y - 3) / (6 * y^2 - 37 * y + 6) = 0) → 
  y = 1/6 := by sorry

end smallest_undefined_value_l1793_179389


namespace longer_side_length_l1793_179382

/-- A rectangular plot with fence poles -/
structure FencedPlot where
  width : ℝ
  length : ℝ
  pole_distance : ℝ
  pole_count : ℕ

/-- The perimeter of a rectangle -/
def perimeter (plot : FencedPlot) : ℝ :=
  2 * (plot.width + plot.length)

/-- The total length of fencing -/
def fencing_length (plot : FencedPlot) : ℝ :=
  (plot.pole_count - 1 : ℝ) * plot.pole_distance

theorem longer_side_length (plot : FencedPlot) 
  (h1 : plot.width = 15)
  (h2 : plot.pole_distance = 5)
  (h3 : plot.pole_count = 26)
  (h4 : plot.width < plot.length)
  (h5 : perimeter plot = fencing_length plot) :
  plot.length = 47.5 := by
  sorry

end longer_side_length_l1793_179382


namespace open_box_volume_l1793_179369

/-- The volume of an open box formed by cutting squares from corners of a rectangular sheet -/
theorem open_box_volume (sheet_length sheet_width cut_length : ℝ) :
  sheet_length = 48 ∧ 
  sheet_width = 36 ∧ 
  cut_length = 8 →
  let box_length := sheet_length - 2 * cut_length
  let box_width := sheet_width - 2 * cut_length
  let box_height := cut_length
  box_length * box_width * box_height = 5120 := by
sorry

end open_box_volume_l1793_179369


namespace jury_deliberation_theorem_l1793_179335

/-- Calculates the equivalent full days spent in jury deliberation --/
def jury_deliberation_days (total_days : ℕ) (selection_days : ℕ) (trial_multiplier : ℕ) 
  (deliberation_hours_per_day : ℕ) (hours_per_day : ℕ) : ℕ :=
  let trial_days := selection_days * trial_multiplier
  let deliberation_days := total_days - selection_days - trial_days
  let total_deliberation_hours := deliberation_days * deliberation_hours_per_day
  total_deliberation_hours / hours_per_day

theorem jury_deliberation_theorem :
  jury_deliberation_days 19 2 4 16 24 = 6 := by
  sorry

end jury_deliberation_theorem_l1793_179335


namespace no_three_digit_even_sum_27_l1793_179329

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_three_digit_even_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ Even n :=
sorry

end no_three_digit_even_sum_27_l1793_179329


namespace smallest_satisfying_number_l1793_179398

theorem smallest_satisfying_number : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 10 → n % k = k - 1) ∧
  (∀ (m : ℕ), m > 0 ∧ 
    (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 10 → m % k = k - 1) → m ≥ 2519) ∧
  (2519 % 10 = 9 ∧ 
   2519 % 9 = 8 ∧ 
   2519 % 8 = 7 ∧ 
   2519 % 7 = 6 ∧ 
   2519 % 6 = 5 ∧ 
   2519 % 5 = 4 ∧ 
   2519 % 4 = 3 ∧ 
   2519 % 3 = 2 ∧ 
   2519 % 2 = 1) :=
by sorry

end smallest_satisfying_number_l1793_179398


namespace equation_equality_l1793_179324

theorem equation_equality : 5 + (-6) - (-7) = 5 - 6 + 7 := by
  sorry

end equation_equality_l1793_179324


namespace rabbit_speed_l1793_179368

def rabbit_speed_equation (x : ℝ) : Prop :=
  2 * (2 * x + 4) = 188

theorem rabbit_speed : ∃ x : ℝ, rabbit_speed_equation x ∧ x = 45 := by
  sorry

end rabbit_speed_l1793_179368


namespace forty_fifth_even_positive_integer_l1793_179317

theorem forty_fifth_even_positive_integer :
  (fun n : ℕ => 2 * n) 45 = 90 := by sorry

end forty_fifth_even_positive_integer_l1793_179317


namespace circular_center_ratio_l1793_179397

/-- Represents a square flag with a symmetric cross design -/
structure SymmetricCrossFlag where
  side : ℝ
  cross_area_ratio : ℝ
  (cross_area_valid : cross_area_ratio = 1/4)

/-- The area of the circular center of the cross -/
noncomputable def circular_center_area (flag : SymmetricCrossFlag) : ℝ :=
  (flag.cross_area_ratio * flag.side^2) / 4

theorem circular_center_ratio (flag : SymmetricCrossFlag) :
  circular_center_area flag / flag.side^2 = 1/4 :=
by sorry

end circular_center_ratio_l1793_179397


namespace chocolate_solution_l1793_179313

def chocolate_problem (n : ℕ) (c s : ℝ) : Prop :=
  -- Condition 1: The cost price of n chocolates equals the selling price of 150 chocolates
  n * c = 150 * s ∧
  -- Condition 2: The gain percent is 10
  (s - c) / c = 0.1

theorem chocolate_solution :
  ∃ (n : ℕ) (c s : ℝ), chocolate_problem n c s ∧ n = 165 :=
by sorry

end chocolate_solution_l1793_179313


namespace sequence_transformations_l1793_179399

def Sequence (α : Type) := ℕ → α

def is_obtainable (s t : Sequence ℝ) : Prop :=
  ∃ (operations : List (Sequence ℝ → Sequence ℝ)),
    (operations.foldl (λ acc op => op acc) s) = t

theorem sequence_transformations (a b c : Sequence ℝ) :
  (∀ n, a n = n^2) ∧
  (∀ n, b n = n + Real.sqrt 2) ∧
  (∀ n, c n = (n^2000 + 1) / n) →
  (is_obtainable a (λ n => n)) ∧
  (¬ is_obtainable b (λ n => n)) ∧
  (is_obtainable c (λ n => n)) := by
  sorry

end sequence_transformations_l1793_179399


namespace bill_donut_combinations_l1793_179393

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 2 identical objects into 4 distinct boxes --/
def bill_combinations : ℕ := distribute 2 4

/-- Theorem: Bill's donut combinations equal 10 --/
theorem bill_donut_combinations : bill_combinations = 10 := by sorry

end bill_donut_combinations_l1793_179393


namespace units_digit_problem_l1793_179380

theorem units_digit_problem :
  ∃ n : ℕ, (15 + Real.sqrt 221)^19 + 3 * (15 + Real.sqrt 221)^83 = 10 * n + 0 := by
  sorry

end units_digit_problem_l1793_179380


namespace correct_mean_calculation_l1793_179314

def incorrect_mean : ℝ := 120
def num_values : ℕ := 40
def original_values : List ℝ := [-50, 350, 100, 25, -80]
def incorrect_values : List ℝ := [-30, 320, 120, 60, -100]

theorem correct_mean_calculation :
  let incorrect_sum := incorrect_mean * num_values
  let difference := (List.sum original_values) - (List.sum incorrect_values)
  let correct_sum := incorrect_sum + difference
  correct_sum / num_values = 119.375 := by
sorry

end correct_mean_calculation_l1793_179314


namespace bracelets_lost_l1793_179312

theorem bracelets_lost (initial_bracelets : ℕ) (remaining_bracelets : ℕ) 
  (h1 : initial_bracelets = 9) 
  (h2 : remaining_bracelets = 7) : 
  initial_bracelets - remaining_bracelets = 2 := by
  sorry

end bracelets_lost_l1793_179312


namespace cody_discount_l1793_179359

/-- The discount Cody got after taxes --/
def discount_after_taxes (initial_price tax_rate discount final_price : ℝ) : ℝ :=
  initial_price * (1 + tax_rate) - final_price

/-- Theorem stating the discount Cody got after taxes --/
theorem cody_discount :
  ∃ (discount : ℝ),
    discount_after_taxes 40 0.05 discount (2 * 17) = 8 :=
by
  sorry

end cody_discount_l1793_179359


namespace brownies_remaining_l1793_179351

/-- The number of brownies left after Tina, her husband, and guests eat some. -/
def brownies_left (total : ℕ) (tina_daily : ℕ) (tina_days : ℕ) (husband_daily : ℕ) (husband_days : ℕ) (shared : ℕ) : ℕ :=
  total - (tina_daily * tina_days + husband_daily * husband_days + shared)

/-- Theorem stating that 5 brownies are left under the given conditions. -/
theorem brownies_remaining : brownies_left 24 2 5 1 5 4 = 5 := by
  sorry

end brownies_remaining_l1793_179351


namespace millet_majority_on_fourth_day_l1793_179360

/-- Represents the proportion of millet remaining after birds consume 40% --/
def milletRemainingRatio : ℝ := 0.6

/-- Represents the proportion of millet in the daily seed addition --/
def dailyMilletAddition : ℝ := 0.4

/-- Calculates the total proportion of millet in the feeder after n days --/
def milletProportion (n : ℕ) : ℝ :=
  1 - milletRemainingRatio ^ n

/-- Theorem stating that on the fourth day, the proportion of millet exceeds 50% for the first time --/
theorem millet_majority_on_fourth_day :
  (milletProportion 4 > 1/2) ∧ 
  (∀ k : ℕ, k < 4 → milletProportion k ≤ 1/2) := by
  sorry


end millet_majority_on_fourth_day_l1793_179360


namespace reciprocal_nonexistence_l1793_179348

theorem reciprocal_nonexistence (a : ℝ) : (¬∃x : ℝ, x * a = 1) → a = 0 := by
  sorry

end reciprocal_nonexistence_l1793_179348


namespace inequality_proof_l1793_179391

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := by
  sorry

end inequality_proof_l1793_179391


namespace resettlement_threshold_year_consecutive_equal_proportion_l1793_179379

/-- The area of new housing constructed in the first year (2015) in millions of square meters. -/
def initial_new_housing : ℝ := 5

/-- The area of resettlement housing in the first year (2015) in millions of square meters. -/
def initial_resettlement : ℝ := 2

/-- The annual growth rate of new housing area. -/
def new_housing_growth_rate : ℝ := 0.1

/-- The annual increase in resettlement housing area in millions of square meters. -/
def resettlement_increase : ℝ := 0.5

/-- The cumulative area of resettlement housing after n years. -/
def cumulative_resettlement (n : ℕ) : ℝ :=
  25 * n^2 + 175 * n

/-- The area of new housing in the nth year. -/
def new_housing (n : ℕ) : ℝ :=
  initial_new_housing * (1 + new_housing_growth_rate)^(n - 1)

/-- The area of resettlement housing in the nth year. -/
def resettlement (n : ℕ) : ℝ :=
  initial_resettlement + resettlement_increase * (n - 1)

theorem resettlement_threshold_year :
  ∃ n : ℕ, cumulative_resettlement n ≥ 30 ∧ ∀ m < n, cumulative_resettlement m < 30 :=
sorry

theorem consecutive_equal_proportion :
  ∃ n : ℕ, resettlement n / new_housing n = resettlement (n + 1) / new_housing (n + 1) :=
sorry

end resettlement_threshold_year_consecutive_equal_proportion_l1793_179379


namespace bridge_length_l1793_179363

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 225 := by
  sorry

end bridge_length_l1793_179363


namespace x_minus_y_equals_three_l1793_179306

theorem x_minus_y_equals_three (x y : ℝ) 
  (h1 : x + y = 8) 
  (h2 : x^2 - y^2 = 24) : 
  x - y = 3 := by
sorry

end x_minus_y_equals_three_l1793_179306


namespace smallest_sum_of_sequence_l1793_179320

/-- Given positive integers A, B, C, D satisfying the conditions:
    1. A, B, C form an arithmetic sequence
    2. B, C, D form a geometric sequence
    3. C/B = 4/3
    The smallest possible value of A + B + C + D is 43. -/
theorem smallest_sum_of_sequence (A B C D : ℕ+) : 
  (∃ r : ℚ, C = A + r ∧ B = A + 2*r) →  -- A, B, C form an arithmetic sequence
  (∃ q : ℚ, C = B * q ∧ D = C * q) →   -- B, C, D form a geometric sequence
  (C : ℚ) / B = 4 / 3 →                -- The ratio of the geometric sequence
  A + B + C + D ≥ 43 ∧ 
  (∃ A' B' C' D' : ℕ+, A' + B' + C' + D' = 43 ∧ 
    (∃ r : ℚ, C' = A' + r ∧ B' = A' + 2*r) ∧
    (∃ q : ℚ, C' = B' * q ∧ D' = C' * q) ∧
    (C' : ℚ) / B' = 4 / 3) :=
by sorry

end smallest_sum_of_sequence_l1793_179320


namespace base_r_is_10_l1793_179319

/-- Converts a number from base r to base 10 -/
def toBase10 (digits : List Nat) (r : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * r ^ i) 0

/-- The problem statement -/
theorem base_r_is_10 (r : Nat) : r > 0 → 
  toBase10 [1, 3, 5] r + toBase10 [1, 5, 4] r = toBase10 [0, 0, 1, 1] r → 
  r = 10 := by
  sorry

end base_r_is_10_l1793_179319


namespace rectangle_cylinder_max_volume_l1793_179367

theorem rectangle_cylinder_max_volume (x y : Real) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 9) :
  let V := π * x * y^2
  (∀ x' y' : Real, x' > 0 → y' > 0 → x' + y' = 9 → π * x' * y'^2 ≤ π * x * y^2) →
  x = 6 ∧ V = 108 * π :=
by sorry

end rectangle_cylinder_max_volume_l1793_179367


namespace min_value_of_function_equality_condition_l1793_179355

theorem min_value_of_function (x : ℝ) (h : x > 0) : (x^2 + 1) / x ≥ 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) : (x^2 + 1) / x = 2 ↔ x = 1 :=
by sorry

end min_value_of_function_equality_condition_l1793_179355


namespace smallest_dual_palindrome_correct_l1793_179337

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

def smallest_dual_palindrome : ℕ := 33

theorem smallest_dual_palindrome_correct :
  (smallest_dual_palindrome > 10) ∧
  (is_palindrome smallest_dual_palindrome 3) ∧
  (is_palindrome smallest_dual_palindrome 5) ∧
  (∀ m : ℕ, m > 10 ∧ m < smallest_dual_palindrome →
    ¬(is_palindrome m 3 ∧ is_palindrome m 5)) :=
by sorry

end smallest_dual_palindrome_correct_l1793_179337


namespace min_value_of_f_l1793_179315

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- The theorem stating that the minimum value of f(x) is -1 -/
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -1 := by
  sorry

end min_value_of_f_l1793_179315


namespace book_pages_count_l1793_179321

/-- The number of pages Frank reads per day -/
def pages_per_day : ℕ := 22

/-- The number of days it took Frank to finish the book -/
def days_to_finish : ℕ := 569

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_day * days_to_finish

theorem book_pages_count : total_pages = 12518 := by
  sorry

end book_pages_count_l1793_179321


namespace hyperbola_foci_coordinates_l1793_179373

/-- The coordinates of the foci for the hyperbola x^2 - 4y^2 = 4 are (±√5, 0) -/
theorem hyperbola_foci_coordinates :
  let h : ℝ → ℝ → Prop := λ x y => x^2 - 4*y^2 = 4
  ∃ c : ℝ, c^2 = 5 ∧ 
    (∀ x y, h x y ↔ (x/2)^2 - y^2 = 1) ∧
    (∀ x y, h x y → (x = c ∨ x = -c) ∧ y = 0) :=
by sorry

end hyperbola_foci_coordinates_l1793_179373


namespace current_velocity_is_two_l1793_179325

-- Define the rowing speed in still water
def still_water_speed : ℝ := 10

-- Define the total time for the round trip
def total_time : ℝ := 15

-- Define the distance to the place
def distance : ℝ := 72

-- Define the velocity of the current as a variable
def current_velocity : ℝ → ℝ := λ v => v

-- Define the equation for the total time of the round trip
def time_equation (v : ℝ) : Prop :=
  distance / (still_water_speed - v) + distance / (still_water_speed + v) = total_time

-- Theorem statement
theorem current_velocity_is_two :
  ∃ v : ℝ, time_equation v ∧ current_velocity v = 2 :=
sorry

end current_velocity_is_two_l1793_179325


namespace marie_message_clearing_l1793_179301

/-- Calculate the number of days required to clear all unread messages -/
def days_to_clear_messages (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) : ℕ :=
  if read_per_day > new_per_day then
    (initial_messages + (read_per_day - new_per_day - 1)) / (read_per_day - new_per_day)
  else
    0

theorem marie_message_clearing :
  days_to_clear_messages 98 20 6 = 7 := by
sorry

end marie_message_clearing_l1793_179301


namespace parabola_intersection_sum_l1793_179378

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

/-- The point R -/
def R : ℝ × ℝ := (10, -6)

/-- The line through R with slope n -/
def line_through_R (n : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 + 6 = n * (p.1 - 10)}

/-- The condition for non-intersection -/
def no_intersection (n : ℝ) : Prop :=
  line_through_R n ∩ P = ∅

theorem parabola_intersection_sum (a b : ℝ) :
  (∀ n, no_intersection n ↔ a < n ∧ n < b) →
  a + b = 40 := by sorry

end parabola_intersection_sum_l1793_179378


namespace negation_of_exp_inequality_l1793_179300

theorem negation_of_exp_inequality (p : Prop) : 
  (p ↔ ∀ x : ℝ, x > 0 → Real.exp x ≥ 1) → 
  (¬p ↔ ∃ x : ℝ, x > 0 ∧ Real.exp x < 1) :=
sorry

end negation_of_exp_inequality_l1793_179300


namespace max_sum_products_l1793_179392

theorem max_sum_products (a b c d : ℕ) : 
  a ∈ ({2, 3, 4, 5} : Set ℕ) → 
  b ∈ ({2, 3, 4, 5} : Set ℕ) → 
  c ∈ ({2, 3, 4, 5} : Set ℕ) → 
  d ∈ ({2, 3, 4, 5} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (∀ x y z w, x ∈ ({2, 3, 4, 5} : Set ℕ) → 
              y ∈ ({2, 3, 4, 5} : Set ℕ) → 
              z ∈ ({2, 3, 4, 5} : Set ℕ) → 
              w ∈ ({2, 3, 4, 5} : Set ℕ) → 
              x ≠ y → x ≠ z → x ≠ w → y ≠ z → y ≠ w → z ≠ w →
              x * y + x * z + x * w + y * z ≤ a * b + a * c + a * d + b * c) →
  a * b + a * c + a * d + b * c = 39 :=
by sorry

end max_sum_products_l1793_179392


namespace min_area_circle_through_intersections_l1793_179332

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x + y + 4 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the minimum area circle
def min_area_circle (x y : ℝ) : Prop := (x + 13/5)^2 + (y - 6/5)^2 = 4/5

-- Theorem statement
theorem min_area_circle_through_intersections :
  ∀ x y : ℝ, 
  (∃ x1 y1 x2 y2 : ℝ, 
    line_l x1 y1 ∧ circle_C x1 y1 ∧
    line_l x2 y2 ∧ circle_C x2 y2 ∧
    x1 ≠ x2 ∧ y1 ≠ y2 ∧
    min_area_circle x1 y1 ∧
    min_area_circle x2 y2) →
  (∀ r : ℝ, ∀ a b : ℝ,
    ((x - a)^2 + (y - b)^2 = r^2 ∧
     (∃ x1 y1 x2 y2 : ℝ, 
       line_l x1 y1 ∧ circle_C x1 y1 ∧
       line_l x2 y2 ∧ circle_C x2 y2 ∧
       (x1 - a)^2 + (y1 - b)^2 = r^2 ∧
       (x2 - a)^2 + (y2 - b)^2 = r^2)) →
    r^2 ≥ 4/5) :=
sorry

end min_area_circle_through_intersections_l1793_179332


namespace school_demographics_l1793_179388

theorem school_demographics (total_students : ℕ) (boys_avg_age girls_avg_age school_avg_age : ℚ) : 
  total_students = 632 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  school_avg_age = 47/4 →
  ∃ (num_girls : ℕ), num_girls = 156 ∧ num_girls ≤ total_students := by
  sorry

end school_demographics_l1793_179388


namespace roots_of_equation_l1793_179308

theorem roots_of_equation : ∀ x : ℝ, 
  (x^2 - 5*x + 6)*(x - 1)*(x - 6) = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 := by
  sorry

end roots_of_equation_l1793_179308


namespace distinct_paintings_l1793_179376

/-- The number of disks in the circle -/
def n : ℕ := 12

/-- The number of disks to be painted blue -/
def blue : ℕ := 4

/-- The number of disks to be painted red -/
def red : ℕ := 3

/-- The number of disks to be painted green -/
def green : ℕ := 2

/-- The total number of disks to be painted -/
def painted : ℕ := blue + red + green

/-- The number of rotational symmetries of the circle -/
def symmetries : ℕ := n

/-- The number of ways to color the disks without considering symmetry -/
def total_colorings : ℕ := Nat.choose n blue * Nat.choose (n - blue) red * Nat.choose (n - blue - red) green

/-- The number of distinct paintings considering rotational symmetry -/
theorem distinct_paintings : (total_colorings / symmetries : ℚ) = 23100 := by
  sorry

end distinct_paintings_l1793_179376


namespace range_of_a_l1793_179371

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2*a - 4}
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A a ∩ B = A a → a < 5 := by
  sorry

end range_of_a_l1793_179371


namespace expression_value_when_x_is_two_l1793_179345

theorem expression_value_when_x_is_two :
  let x : ℝ := 2
  (x + 2 - x) * (2 - x - 2) = -4 :=
by sorry

end expression_value_when_x_is_two_l1793_179345


namespace smallest_a_value_l1793_179385

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.cos (a * ↑x + b) = Real.sin (36 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, ∃ b' ≥ 0, Real.cos (a' * ↑x + b') = Real.sin (36 * ↑x)) → a' ≥ 36 :=
sorry

end smallest_a_value_l1793_179385


namespace xy_value_l1793_179307

theorem xy_value (x y : ℝ) : 
  |x - y + 1| + (y + 5)^2010 = 0 → x * y = 30 := by sorry

end xy_value_l1793_179307


namespace smallest_append_digits_for_2014_l1793_179336

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10 → k > 0 → n % k = 0

def append_digits (base n digits : ℕ) : ℕ :=
  base * (10 ^ digits) + n

theorem smallest_append_digits_for_2014 :
  (∃ n : ℕ, n < 10000 ∧ is_divisible_by_all_less_than_10 (append_digits 2014 4 n)) ∧
  (∀ d : ℕ, d < 4 → ∀ n : ℕ, n < 10^d → ¬is_divisible_by_all_less_than_10 (append_digits 2014 d n)) :=
sorry

end smallest_append_digits_for_2014_l1793_179336


namespace least_three_digit_with_digit_product_8_l1793_179387

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem least_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 118 ≤ n :=
by sorry

end least_three_digit_with_digit_product_8_l1793_179387


namespace tangent_line_equation_l1793_179357

/-- The circle C with equation x^2 + y^2 = 10 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 10}

/-- The point P(1, 3) -/
def P : ℝ × ℝ := (1, 3)

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A line is tangent to a circle if it intersects the circle at exactly one point -/
def IsTangentTo (l : Line) (s : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ s ∧ l.a * p.1 + l.b * p.2 + l.c = 0

theorem tangent_line_equation :
  IsTangentTo (Line.mk 1 3 (-10)) C ∧ (Line.mk 1 3 (-10)).a * P.1 + (Line.mk 1 3 (-10)).b * P.2 + (Line.mk 1 3 (-10)).c = 0 :=
by sorry

end tangent_line_equation_l1793_179357


namespace nest_distance_building_materials_distance_l1793_179333

/-- Given two birds making round trips to collect building materials, 
    calculate the distance from the nest to the materials. -/
theorem nest_distance (num_birds : ℕ) (num_trips : ℕ) (total_distance : ℝ) : ℝ :=
  let distance_per_bird := total_distance / num_birds
  let distance_per_trip := distance_per_bird / num_trips
  distance_per_trip / 4

/-- Prove that for two birds making 10 round trips each, 
    with a total distance of 8000 miles, 
    the building materials are 100 miles from the nest. -/
theorem building_materials_distance : 
  nest_distance 2 10 8000 = 100 := by
  sorry

end nest_distance_building_materials_distance_l1793_179333


namespace last_person_coins_l1793_179361

/-- Represents the amount of coins each person receives in an arithmetic sequence. -/
structure CoinDistribution where
  a : ℚ
  d : ℚ

/-- Calculates the total number of coins distributed. -/
def totalCoins (dist : CoinDistribution) : ℚ :=
  5 * dist.a

/-- Checks if the sum of the first two equals the sum of the last three. -/
def sumCondition (dist : CoinDistribution) : Prop :=
  (dist.a - 2*dist.d) + (dist.a - dist.d) = dist.a + (dist.a + dist.d) + (dist.a + 2*dist.d)

/-- The main theorem stating the amount the last person receives. -/
theorem last_person_coins (dist : CoinDistribution) 
  (h1 : totalCoins dist = 5)
  (h2 : sumCondition dist) :
  dist.a + 2*dist.d = 2/3 := by
  sorry

end last_person_coins_l1793_179361


namespace parabola_line_intersection_l1793_179318

/-- Given a parabola and a moving line with common points, prove the range of t and minimum value of c -/
theorem parabola_line_intersection (t c x₁ x₂ y₁ y₂ : ℝ) : 
  (∀ x, y₁ = x^2 ∧ y₁ = (2*t - 1)*x - c) →  -- Parabola and line equations
  (∀ x, y₂ = x^2 ∧ y₂ = (2*t - 1)*x - c) →  -- Parabola and line equations
  x₁^2 + x₂^2 = t^2 + 2*t - 3 →             -- Given condition
  (2 - Real.sqrt 2 ≤ t ∧ t ≤ 2 + Real.sqrt 2 ∧ t ≠ 1/2) ∧  -- Range of t
  (c ≥ (11 - 6*Real.sqrt 2) / 4) ∧                        -- Minimum value of c
  (c = (11 - 6*Real.sqrt 2) / 4 ↔ t = 2 - Real.sqrt 2)    -- When minimum occurs
  := by sorry


end parabola_line_intersection_l1793_179318
