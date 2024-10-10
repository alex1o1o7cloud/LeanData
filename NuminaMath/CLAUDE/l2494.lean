import Mathlib

namespace plane_equation_correct_l2494_249484

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
structure ParametricPlane where
  origin : Point3D
  direction1 : Point3D
  direction2 : Point3D

/-- Represents the equation of a plane in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point satisfies a plane equation -/
def satisfiesPlaneEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- The given parametric equation of the plane -/
def givenPlane : ParametricPlane :=
  { origin := { x := 2, y := 4, z := 1 }
  , direction1 := { x := 2, y := 1, z := -3 }
  , direction2 := { x := -3, y := 0, z := 1 }
  }

/-- The equation of the plane to be proven -/
def planeEquation : PlaneEquation :=
  { A := 1, B := 8, C := 3, D := -37 }

theorem plane_equation_correct :
  (∀ s t : ℝ, satisfiesPlaneEquation
    { x := 2 + 2*s - 3*t
    , y := 4 + s
    , z := 1 - 3*s + t
    } planeEquation) ∧
  planeEquation.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs planeEquation.A) (Int.natAbs planeEquation.B))
          (Nat.gcd (Int.natAbs planeEquation.C) (Int.natAbs planeEquation.D)) = 1 :=
by sorry

end plane_equation_correct_l2494_249484


namespace square_side_length_l2494_249475

/-- The perimeter of an equilateral triangle with side length s -/
def triangle_perimeter (s : ℝ) : ℝ := 3 * s

/-- The perimeter of a square with side length s -/
def square_perimeter (s : ℝ) : ℝ := 4 * s

/-- The side length of the equilateral triangle -/
def triangle_side : ℝ := 12

theorem square_side_length :
  ∃ (s : ℝ), s = 9 ∧ square_perimeter s = triangle_perimeter triangle_side :=
by sorry

end square_side_length_l2494_249475


namespace garden_plants_l2494_249430

/-- The total number of plants in a rectangular garden -/
def total_plants (rows : ℕ) (columns : ℕ) : ℕ := rows * columns

/-- Theorem: A garden with 52 rows and 15 columns has 780 plants in total -/
theorem garden_plants : total_plants 52 15 = 780 := by
  sorry

end garden_plants_l2494_249430


namespace prob_eight_rolls_prime_odd_l2494_249436

/-- A function representing the probability of rolling either 3 or 5 on a standard die -/
def prob_prime_odd_roll : ℚ := 1 / 3

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The probability of getting a product of all rolls that is odd and consists only of prime numbers -/
def prob_all_prime_odd : ℚ := (prob_prime_odd_roll) ^ num_rolls

theorem prob_eight_rolls_prime_odd :
  prob_all_prime_odd = 1 / 6561 := by sorry

end prob_eight_rolls_prime_odd_l2494_249436


namespace senate_committee_seating_l2494_249461

/-- The number of unique circular arrangements of n distinguishable objects -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

theorem senate_committee_seating :
  circularArrangements 10 = 362880 := by
  sorry

end senate_committee_seating_l2494_249461


namespace solution_to_equation_l2494_249438

theorem solution_to_equation (x : ℝ) (h : (9 : ℝ) / x^2 = x / 81) : x = 9 := by
  sorry

end solution_to_equation_l2494_249438


namespace percentage_problem_l2494_249423

theorem percentage_problem (x : ℝ) (h : x = 942.8571428571427) :
  ∃ P : ℝ, (P / 100) * x = (1 / 3) * x + 110 ∧ P = 45 := by
  sorry

end percentage_problem_l2494_249423


namespace jackson_vacation_savings_l2494_249498

/-- Calculates the total savings for a vacation given the number of months,
    paychecks per month, and amount saved per paycheck. -/
def vacation_savings (months : ℕ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℕ) : ℕ :=
  months * paychecks_per_month * savings_per_paycheck

/-- Proves that Jackson's vacation savings equal $3000 given the problem conditions. -/
theorem jackson_vacation_savings :
  vacation_savings 15 2 100 = 3000 := by
  sorry

end jackson_vacation_savings_l2494_249498


namespace zoom_video_glitch_duration_l2494_249420

theorem zoom_video_glitch_duration :
  let mac_download_time : ℕ := 10
  let windows_download_time : ℕ := 3 * mac_download_time
  let total_download_time : ℕ := mac_download_time + windows_download_time
  let total_time : ℕ := 82
  let call_time : ℕ := total_time - total_download_time
  let audio_glitch_time : ℕ := 2 * 4
  let video_glitch_time : ℕ := call_time - (audio_glitch_time + 2 * (audio_glitch_time + video_glitch_time))
  video_glitch_time = 6 := by
  sorry

end zoom_video_glitch_duration_l2494_249420


namespace sum_of_roots_l2494_249408

theorem sum_of_roots (a b : ℝ) : 
  (a^2 - 4*a - 2023 = 0) → (b^2 - 4*b - 2023 = 0) → a + b = 4 := by
  sorry

end sum_of_roots_l2494_249408


namespace backpack_cost_l2494_249435

/-- Calculates the total cost of personalized backpacks for grandchildren --/
def totalCost (originalPrice taxRates : List ℝ) (discount monogrammingCost coupon : ℝ) : ℝ :=
  let discountedPrice := originalPrice.map (λ p => p * (1 - discount))
  let priceWithMonogram := discountedPrice.map (λ p => p + monogrammingCost)
  let priceWithTax := List.zipWith (λ p r => p * (1 + r)) priceWithMonogram taxRates
  priceWithTax.sum - coupon

/-- Theorem stating the total cost of backpacks for grandchildren --/
theorem backpack_cost :
  let originalPrice := [20, 20, 20, 20, 20]
  let taxRates := [0.06, 0.08, 0.055, 0.0725, 0.04]
  let discount := 0.2
  let monogrammingCost := 12
  let coupon := 5
  totalCost originalPrice taxRates discount monogrammingCost coupon = 143.61 := by
  sorry

#eval totalCost [20, 20, 20, 20, 20] [0.06, 0.08, 0.055, 0.0725, 0.04] 0.2 12 5

end backpack_cost_l2494_249435


namespace dot_product_range_l2494_249474

theorem dot_product_range (a b : ℝ × ℝ) : 
  let norm_a := Real.sqrt (a.1^2 + a.2^2)
  let angle := Real.arccos ((b.1 * (a.1 - b.1) + b.2 * (a.2 - b.2)) / 
    (Real.sqrt (b.1^2 + b.2^2) * Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)))
  norm_a = 2 ∧ angle = 2 * Real.pi / 3 →
  2 - 4 * Real.sqrt 3 / 3 ≤ a.1 * b.1 + a.2 * b.2 ∧ 
  a.1 * b.1 + a.2 * b.2 ≤ 2 + 4 * Real.sqrt 3 / 3 :=
by sorry

end dot_product_range_l2494_249474


namespace mini_van_tank_capacity_l2494_249490

/-- Proves that the capacity of a mini-van's tank is 65 liters given the specified conditions -/
theorem mini_van_tank_capacity :
  let service_cost : ℝ := 2.20
  let fuel_cost_per_liter : ℝ := 0.70
  let num_mini_vans : ℕ := 4
  let num_trucks : ℕ := 2
  let total_cost : ℝ := 395.4
  let truck_tank_ratio : ℝ := 2.2  -- 120% bigger means 2.2 times the size

  ∃ (mini_van_capacity : ℝ),
    mini_van_capacity > 0 ∧
    (service_cost * (num_mini_vans + num_trucks) +
     fuel_cost_per_liter * (num_mini_vans * mini_van_capacity + num_trucks * (truck_tank_ratio * mini_van_capacity)) = total_cost) ∧
    mini_van_capacity = 65 :=
by
  sorry

end mini_van_tank_capacity_l2494_249490


namespace same_color_probability_l2494_249426

/-- The number of green balls in the bag -/
def green_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 7

/-- The total number of balls in the bag -/
def total_balls : ℕ := green_balls + red_balls

/-- The probability of drawing two balls of the same color with replacement -/
theorem same_color_probability : 
  (green_balls / total_balls) ^ 2 + (red_balls / total_balls) ^ 2 = 113 / 225 := by
  sorry

end same_color_probability_l2494_249426


namespace vector_subtraction_l2494_249414

/-- Given two vectors OA and OB in ℝ², prove that the vector AB is their difference. -/
theorem vector_subtraction (OA OB : ℝ × ℝ) (h1 : OA = (1, -2)) (h2 : OB = (-3, 1)) :
  OB - OA = (-4, 3) := by
  sorry

end vector_subtraction_l2494_249414


namespace sqrt_equation_solution_l2494_249454

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (x - 3) = 5 → x = 28 := by sorry

end sqrt_equation_solution_l2494_249454


namespace rectangle_with_equal_sides_is_square_inverse_proposition_inverse_proposition_is_true_l2494_249401

-- Define what a rectangle is
def is_rectangle (shape : Type) : Prop := sorry

-- Define what a square is
def is_square (shape : Type) : Prop := sorry

-- Define what it means for a shape to have equal adjacent sides
def has_equal_adjacent_sides (shape : Type) : Prop := sorry

-- The original proposition
theorem rectangle_with_equal_sides_is_square (shape : Type) :
  is_rectangle shape → has_equal_adjacent_sides shape → is_square shape := sorry

-- The inverse proposition
theorem inverse_proposition (shape : Type) :
  is_square shape → is_rectangle shape ∧ has_equal_adjacent_sides shape := sorry

-- The main theorem: proving that the inverse proposition is true
theorem inverse_proposition_is_true :
  (∀ shape, is_square shape → is_rectangle shape ∧ has_equal_adjacent_sides shape) := sorry

end rectangle_with_equal_sides_is_square_inverse_proposition_inverse_proposition_is_true_l2494_249401


namespace parabola_focus_distance_l2494_249449

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a point on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

-- Define the y-axis intersection point
def y_axis_intersection (N : ℝ × ℝ) : Prop :=
  N.1 = 0

-- Define the midpoint condition
def is_midpoint (F M N : ℝ × ℝ) : Prop :=
  M.1 = (F.1 + N.1) / 2 ∧ M.2 = (F.2 + N.2) / 2

-- Main theorem
theorem parabola_focus_distance (M N : ℝ × ℝ) :
  point_on_parabola M →
  y_axis_intersection N →
  is_midpoint focus M N →
  (focus.1 - N.1)^2 + (focus.2 - N.2)^2 = 36 := by
  sorry

end parabola_focus_distance_l2494_249449


namespace trapezoid_area_is_2198_l2494_249497

-- Define the trapezoid
structure Trapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ

-- Define the properties of our specific trapezoid
def my_trapezoid : Trapezoid := {
  leg := 40
  diagonal := 50
  longer_base := 60
}

-- Function to calculate the area of the trapezoid
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

-- Theorem statement
theorem trapezoid_area_is_2198 : 
  trapezoid_area my_trapezoid = 2198 := by
  sorry

end trapezoid_area_is_2198_l2494_249497


namespace cylinder_not_identical_views_l2494_249444

-- Define the basic shapes
structure Shape :=
  (name : String)

-- Define the views
inductive View
  | Top
  | Front
  | Side

-- Define a function to get the shape of a view
def getViewShape (object : Shape) (view : View) : Shape :=
  sorry

-- Define the property of having identical views
def hasIdenticalViews (object : Shape) : Prop :=
  ∀ v1 v2 : View, getViewShape object v1 = getViewShape object v2

-- Define specific shapes
def cylinder : Shape :=
  { name := "Cylinder" }

def cube : Shape :=
  { name := "Cube" }

-- State the theorem
theorem cylinder_not_identical_views :
  ¬(hasIdenticalViews cylinder) ∧ hasIdenticalViews cube :=
sorry

end cylinder_not_identical_views_l2494_249444


namespace sum_of_cubes_of_five_l2494_249492

theorem sum_of_cubes_of_five : 5^3 + 5^3 + 5^3 + 5^3 = 625 := by
  sorry

end sum_of_cubes_of_five_l2494_249492


namespace sum_and_reciprocal_sum_zero_l2494_249487

theorem sum_and_reciprocal_sum_zero (a b c d : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d)
  (h4 : a + b + c + d = 0)
  (h5 : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 := by
sorry

end sum_and_reciprocal_sum_zero_l2494_249487


namespace nilpotent_matrix_cube_zero_l2494_249495

theorem nilpotent_matrix_cube_zero
  (A : Matrix (Fin 3) (Fin 3) ℝ)
  (h : A ^ 4 = 0) :
  A ^ 3 = 0 := by
sorry

end nilpotent_matrix_cube_zero_l2494_249495


namespace shaded_area_fraction_l2494_249437

theorem shaded_area_fraction (n : ℕ) (h : n = 18) :
  let total_rectangles := n
  let shaded_rectangles := n / 2
  (shaded_rectangles : ℚ) / total_rectangles = 1 / 4 :=
by sorry

end shaded_area_fraction_l2494_249437


namespace room_width_calculation_l2494_249455

theorem room_width_calculation (length area : ℝ) (h1 : length = 12) (h2 : area = 96) :
  area / length = 8 := by
  sorry

end room_width_calculation_l2494_249455


namespace mary_earnings_l2494_249470

/-- Mary's earnings from cleaning homes -/
theorem mary_earnings (total_earnings : ℕ) (homes_cleaned : ℕ) 
  (h1 : total_earnings = 276)
  (h2 : homes_cleaned = 6) :
  total_earnings / homes_cleaned = 46 := by
  sorry

end mary_earnings_l2494_249470


namespace bottom_right_not_divisible_by_2011_l2494_249480

/-- Represents a cell on the board -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the board configuration -/
structure Board where
  size : Nat
  markedCells : List Cell

/-- Checks if a cell is on the main diagonal -/
def isOnMainDiagonal (c : Cell) : Prop := c.row + c.col = 2011

/-- Checks if a cell is in a corner -/
def isCorner (c : Cell) (n : Nat) : Prop :=
  (c.row = 0 ∧ c.col = 0) ∨ (c.row = 0 ∧ c.col = n - 1) ∨
  (c.row = n - 1 ∧ c.col = 0) ∨ (c.row = n - 1 ∧ c.col = n - 1)

/-- The value in the bottom-right corner of the board -/
def bottomRightValue (b : Board) : Nat :=
  sorry  -- Implementation not required for the statement

theorem bottom_right_not_divisible_by_2011 (b : Board) :
  b.size = 2012 →
  (∀ c ∈ b.markedCells, isOnMainDiagonal c ∧ ¬isCorner c b.size) →
  bottomRightValue b % 2011 = 2 :=
sorry

end bottom_right_not_divisible_by_2011_l2494_249480


namespace third_tea_price_is_175_5_l2494_249400

/-- The price of the third variety of tea -/
def third_tea_price (price1 price2 mixture_price : ℚ) : ℚ :=
  2 * mixture_price - (price1 + price2) / 2

/-- Theorem stating that the price of the third variety of tea is 175.5 given the conditions -/
theorem third_tea_price_is_175_5 :
  third_tea_price 126 135 153 = 175.5 := by
  sorry

end third_tea_price_is_175_5_l2494_249400


namespace absolute_value_equality_l2494_249412

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x + 1| → x = 1 := by
  sorry

end absolute_value_equality_l2494_249412


namespace pure_imaginary_condition_l2494_249427

theorem pure_imaginary_condition (x y : ℝ) : 
  (∀ z : ℂ, z.re = x ∧ z.im = y → (z.re = 0 ↔ z.im ≠ 0)) ↔
  (x = 0 → ∃ y : ℝ, y ≠ 0) ∧ (∃ x y : ℝ, x = 0 ∧ y = 0) :=
sorry

end pure_imaginary_condition_l2494_249427


namespace quadratic_root_square_relation_l2494_249473

theorem quadratic_root_square_relation (c : ℝ) : 
  (c > 0) →
  (∃ x₁ x₂ : ℝ, (8 * x₁^2 - 6 * x₁ + 9 * c^2 = 0) ∧ 
                (8 * x₂^2 - 6 * x₂ + 9 * c^2 = 0) ∧ 
                (x₂ = x₁^2)) →
  (c = 1/3) := by
sorry

end quadratic_root_square_relation_l2494_249473


namespace camel_count_theorem_l2494_249486

/-- Represents the number of humps on a camel -/
inductive CamelType
  | dromedary : CamelType  -- one hump
  | bactrian : CamelType   -- two humps

/-- Calculate the number of humps for a given camel type -/
def humps (c : CamelType) : Nat :=
  match c with
  | .dromedary => 1
  | .bactrian => 2

/-- A group of camels -/
structure CamelGroup where
  dromedaryCount : Nat
  bactrianCount : Nat

/-- Calculate the total number of humps in a camel group -/
def totalHumps (g : CamelGroup) : Nat :=
  g.dromedaryCount * humps CamelType.dromedary + g.bactrianCount * humps CamelType.bactrian

/-- Calculate the total number of feet in a camel group -/
def totalFeet (g : CamelGroup) : Nat :=
  (g.dromedaryCount + g.bactrianCount) * 4

/-- Calculate the total number of camels in a group -/
def totalCamels (g : CamelGroup) : Nat :=
  g.dromedaryCount + g.bactrianCount

theorem camel_count_theorem (g : CamelGroup) :
  totalHumps g = 23 → totalFeet g = 60 → totalCamels g = 15 := by
  sorry

end camel_count_theorem_l2494_249486


namespace fraction_equality_l2494_249419

theorem fraction_equality (x y : ℚ) (hx : x = 4/7) (hy : y = 5/11) :
  (7*x + 11*y) / (63*x*y) = 11/20 := by
  sorry

end fraction_equality_l2494_249419


namespace geometric_sequence_common_ratio_l2494_249415

/-- Given a geometric sequence {a_n} with common ratio q, if a₃ = 2S₂ + 1 and a₄ = 2S₃ + 1, then q = 3 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Definition of sum of geometric sequence
  a 3 = 2 * S 2 + 1 →
  a 4 = 2 * S 3 + 1 →
  q = 3 := by
sorry

end geometric_sequence_common_ratio_l2494_249415


namespace sqrt_diff_positive_implies_square_diff_positive_l2494_249462

theorem sqrt_diff_positive_implies_square_diff_positive (a b : ℝ) :
  (∀ (a b : ℝ), Real.sqrt a - Real.sqrt b > 0 → a^2 - b^2 > 0) ∧
  (∃ (a b : ℝ), a^2 - b^2 > 0 ∧ ¬(Real.sqrt a - Real.sqrt b > 0)) :=
by sorry

end sqrt_diff_positive_implies_square_diff_positive_l2494_249462


namespace systematic_sampling_interval_l2494_249489

/-- The sampling interval for systematic sampling -/
def sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The sampling interval for a population of 1200 and sample size of 40 is 30 -/
theorem systematic_sampling_interval :
  sampling_interval 1200 40 = 30 := by
  sorry

end systematic_sampling_interval_l2494_249489


namespace lindas_outfits_l2494_249476

/-- The number of different outfits that can be created from a given number of skirts, blouses, and shoes. -/
def number_of_outfits (skirts blouses shoes : ℕ) : ℕ :=
  skirts * blouses * shoes

/-- Theorem stating that with 5 skirts, 8 blouses, and 2 pairs of shoes, 80 different outfits can be created. -/
theorem lindas_outfits :
  number_of_outfits 5 8 2 = 80 := by
  sorry

end lindas_outfits_l2494_249476


namespace sqrt_equation_solution_l2494_249404

theorem sqrt_equation_solution (z : ℝ) : 
  (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt z) / (Real.sqrt 0.49) = 2.9365079365079367 → 
  z = 1.44 := by
  sorry

end sqrt_equation_solution_l2494_249404


namespace sphere_plane_distance_l2494_249411

/-- The distance between the center of a sphere and a plane intersecting it -/
theorem sphere_plane_distance (r : ℝ) (A : ℝ) (h1 : r = 2) (h2 : A = Real.pi) :
  Real.sqrt (r^2 - (A / Real.pi)) = Real.sqrt 3 := by
  sorry

end sphere_plane_distance_l2494_249411


namespace sin_sum_to_product_l2494_249453

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end sin_sum_to_product_l2494_249453


namespace max_difference_second_largest_smallest_l2494_249406

theorem max_difference_second_largest_smallest (a b c d e f g h : ℕ) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 →
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧ g < h →
  (a + b + c) / 3 = 9 →
  (a + b + c + d + e + f + g + h) / 8 = 19 →
  (f + g + h) / 3 = 29 →
  ∃ (a' b' c' d' e' f' g' h' : ℕ),
    a' ≠ 0 ∧ b' ≠ 0 ∧ c' ≠ 0 ∧ d' ≠ 0 ∧ e' ≠ 0 ∧ f' ≠ 0 ∧ g' ≠ 0 ∧ h' ≠ 0 ∧
    a' < b' ∧ b' < c' ∧ c' < d' ∧ d' < e' ∧ e' < f' ∧ f' < g' ∧ g' < h' ∧
    (a' + b' + c') / 3 = 9 ∧
    (a' + b' + c' + d' + e' + f' + g' + h') / 8 = 19 ∧
    (f' + g' + h') / 3 = 29 ∧
    g' - b' = 26 ∧
    ∀ (a'' b'' c'' d'' e'' f'' g'' h'' : ℕ),
      a'' ≠ 0 ∧ b'' ≠ 0 ∧ c'' ≠ 0 ∧ d'' ≠ 0 ∧ e'' ≠ 0 ∧ f'' ≠ 0 ∧ g'' ≠ 0 ∧ h'' ≠ 0 →
      a'' < b'' ∧ b'' < c'' ∧ c'' < d'' ∧ d'' < e'' ∧ e'' < f'' ∧ f'' < g'' ∧ g'' < h'' →
      (a'' + b'' + c'') / 3 = 9 →
      (a'' + b'' + c'' + d'' + e'' + f'' + g'' + h'') / 8 = 19 →
      (f'' + g'' + h'') / 3 = 29 →
      g'' - b'' ≤ 26 :=
by
  sorry

end max_difference_second_largest_smallest_l2494_249406


namespace root_product_is_root_l2494_249446

/-- Given that a and b are two of the four roots of x^4 + x^3 - 1,
    prove that ab is a root of x^6 + x^4 + x^3 - x^2 - 1 -/
theorem root_product_is_root (a b : ℂ) : 
  (a^4 + a^3 - 1 = 0) → 
  (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
  sorry

end root_product_is_root_l2494_249446


namespace inequality_solution_set_range_of_a_l2494_249456

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2|

-- Theorem for the first part
theorem inequality_solution_set :
  {x : ℝ | 2 * f x < 4 - |x - 1|} = {x : ℝ | -7/3 < x ∧ x < -1} := by sorry

-- Theorem for the second part
theorem range_of_a (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 := by sorry

end inequality_solution_set_range_of_a_l2494_249456


namespace union_A_B_intersection_A_complement_B_l2494_249410

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 2}
def B : Set ℝ := {x | (x - 1) / (x + 5) > 0}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | x < -5 ∨ x > -3} := by sorry

-- Theorem for the intersection of A and complement of B
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x | -3 < x ∧ x ≤ 1} := by sorry

end union_A_B_intersection_A_complement_B_l2494_249410


namespace min_value_of_exponential_expression_l2494_249425

theorem min_value_of_exponential_expression :
  ∀ x : ℝ, 16^x - 4^x + 1 ≥ (3:ℝ)/4 ∧ 
  (16^(-(1:ℝ)/2) - 4^(-(1:ℝ)/2) + 1 = (3:ℝ)/4) := by
  sorry

end min_value_of_exponential_expression_l2494_249425


namespace intersection_point_l2494_249451

-- Define the system of equations
def system_solution (a b : ℝ) : ℝ × ℝ := (-1, 3)

-- Define the condition that the system solution satisfies the equations
def system_satisfies (a b : ℝ) : Prop :=
  let (x, y) := system_solution a b
  2 * x + y = b ∧ x - y = a

-- Define the lines
def line1 (x : ℝ) (b : ℝ) : ℝ := -2 * x + b
def line2 (x : ℝ) (a : ℝ) : ℝ := x - a

-- State the theorem
theorem intersection_point (a b : ℝ) (h : system_satisfies a b) :
  let (x, y) := system_solution a b
  line1 x b = y ∧ line2 x a = y := by sorry

end intersection_point_l2494_249451


namespace water_speed_calculation_l2494_249478

def swim_speed : ℝ := 4
def distance : ℝ := 8
def time : ℝ := 4

theorem water_speed_calculation (v : ℝ) : 
  (swim_speed - v) * time = distance → v = 2 := by
  sorry

end water_speed_calculation_l2494_249478


namespace prob_not_all_same_dice_l2494_249477

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability that not all dice show the same number when rolled -/
def prob_not_all_same : ℚ := 1295 / 1296

/-- Theorem stating that the probability of not all dice showing the same number is 1295/1296 -/
theorem prob_not_all_same_dice (h : sides = 6 ∧ num_dice = 5) : 
  prob_not_all_same = 1295 / 1296 := by sorry

end prob_not_all_same_dice_l2494_249477


namespace smallest_distance_between_complex_points_l2494_249434

theorem smallest_distance_between_complex_points (z w : ℂ) :
  Complex.abs (z - (2 + 4*I)) = 2 →
  Complex.abs (w - (5 + 5*I)) = 4 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 10 + 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 + 4*I)) = 2 →
                   Complex.abs (w' - (5 + 5*I)) = 4 →
                   Complex.abs (z' - w') ≥ min_dist :=
by sorry

end smallest_distance_between_complex_points_l2494_249434


namespace vasyas_numbers_l2494_249445

theorem vasyas_numbers (x y : ℝ) :
  x + y = x * y ∧ x + y = x / y → x = 1/2 ∧ y = -1 := by sorry

end vasyas_numbers_l2494_249445


namespace range_of_a_l2494_249472

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a : 
  ∃ a : ℝ, p a ∧ ¬(q a) ∧ -1 ≤ a ∧ a < 0 ∧
  ∀ b : ℝ, p b ∧ ¬(q b) → -1 ≤ b ∧ b < 0 :=
sorry

end range_of_a_l2494_249472


namespace problem_solution_l2494_249433

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 3)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = -4) :
  a / (a + c) + b / (b + a) + c / (c + b) = -2 := by
sorry

end problem_solution_l2494_249433


namespace solution_set_when_a_is_2_range_of_a_l2494_249460

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part I
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

-- Part II
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 :=
sorry

end solution_set_when_a_is_2_range_of_a_l2494_249460


namespace smallest_angle_trig_equation_l2494_249424

theorem smallest_angle_trig_equation :
  let θ := Real.pi / 14
  (∀ φ > 0, φ < θ → Real.sin (3 * φ) * Real.sin (4 * φ) ≠ Real.cos (3 * φ) * Real.cos (4 * φ)) ∧
  Real.sin (3 * θ) * Real.sin (4 * θ) = Real.cos (3 * θ) * Real.cos (4 * θ) := by
  sorry

end smallest_angle_trig_equation_l2494_249424


namespace circle_tangent_to_line_l2494_249422

/-- A circle with equation x^2 + y^2 = m is tangent to the line x - y = √m if and only if m = 0 -/
theorem circle_tangent_to_line (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x - y = Real.sqrt m) ↔ m = 0 :=
by sorry

end circle_tangent_to_line_l2494_249422


namespace complex_subtraction_simplify_complex_expression_l2494_249447

theorem complex_subtraction (z₁ z₂ : ℂ) : z₁ - z₂ = (z₁.re - z₂.re) + (z₁.im - z₂.im) * I := by sorry

theorem simplify_complex_expression : (3 - 2 * I) - (5 - 2 * I) = -2 := by sorry

end complex_subtraction_simplify_complex_expression_l2494_249447


namespace smallest_integer_with_remainders_l2494_249482

theorem smallest_integer_with_remainders (n : ℕ) : 
  n > 1 ∧ 
  n % 5 = 1 ∧ 
  n % 7 = 1 ∧ 
  n % 8 = 1 ∧ 
  (∀ m : ℕ, m > 1 → m % 5 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) →
  n = 281 ∧ 240 < n ∧ n < 359 := by
sorry

end smallest_integer_with_remainders_l2494_249482


namespace probability_of_non_intersection_l2494_249429

-- Define the circles and their properties
def CircleA : Type := Unit
def CircleB : Type := Unit

-- Define the probability space
def Ω : Type := CircleA × CircleB

-- Define the center distributions
def centerA_distribution : Set ℝ := Set.Icc 0 2
def centerB_distribution : Set ℝ := Set.Icc 0 3

-- Define the radius of each circle
def radiusA : ℝ := 2
def radiusB : ℝ := 1

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the event of non-intersection
def non_intersection : Set Ω := sorry

-- Theorem statement
theorem probability_of_non_intersection :
  P non_intersection = (4 * Real.sqrt 5 - 5) / 3 := by sorry

end probability_of_non_intersection_l2494_249429


namespace stratified_sampling_problem_l2494_249481

/-- Calculates the number of people to be selected from a stratum in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (stratum_size : ℕ) (total_sample_size : ℕ) : ℕ :=
  (total_sample_size * stratum_size) / total_population

/-- The problem statement -/
theorem stratified_sampling_problem (total_population : ℕ) (stratum_size : ℕ) (total_sample_size : ℕ) 
  (h1 : total_population = 360) 
  (h2 : stratum_size = 108) 
  (h3 : total_sample_size = 20) :
  stratified_sample_size total_population stratum_size total_sample_size = 6 := by
  sorry

end stratified_sampling_problem_l2494_249481


namespace binomial_10_3_l2494_249457

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l2494_249457


namespace correct_total_spent_l2494_249428

/-- The total amount Mike spent at the music store after applying the discount -/
def total_spent (trumpet_price songbook_price accessories_price discount_rate : ℝ) : ℝ :=
  let total_before_discount := trumpet_price + songbook_price + accessories_price
  let discount_amount := discount_rate * total_before_discount
  total_before_discount - discount_amount

/-- Theorem stating the correct total amount spent -/
theorem correct_total_spent :
  total_spent 145.16 5.84 18.50 0.12 = 149.16 := by
  sorry

end correct_total_spent_l2494_249428


namespace complex_number_in_first_quadrant_l2494_249416

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 - 4*I) / (1 - 2*I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end complex_number_in_first_quadrant_l2494_249416


namespace binomial_coefficient_ratio_l2494_249409

theorem binomial_coefficient_ratio (n k : ℕ) (h1 : n > 0) (h2 : k > 0) :
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 1 / 2 →
  n + k = 6 :=
sorry

end binomial_coefficient_ratio_l2494_249409


namespace cube_root_function_l2494_249417

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 64, prove that y = 2√3 when x = 8 -/
theorem cube_root_function (k : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = k * x^(1/3)) →
  y 64 = 4 * Real.sqrt 3 →
  y 8 = 2 * Real.sqrt 3 := by
sorry

end cube_root_function_l2494_249417


namespace sector_area_l2494_249448

/-- Given a circular sector with central angle 2 radians and arc length 4, the area of the sector is 4. -/
theorem sector_area (θ : Real) (l : Real) (h1 : θ = 2) (h2 : l = 4) :
  (1/2) * (l/θ)^2 * θ = 4 := by sorry

end sector_area_l2494_249448


namespace sqrt_product_equality_l2494_249402

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l2494_249402


namespace find_first_group_men_l2494_249465

/-- Represents the work rate of one person -/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers -/
structure WorkerGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work done by a group -/
def totalWork (m w : WorkRate) (g : WorkerGroup) : ℝ :=
  (g.men : ℝ) * m.rate + (g.women : ℝ) * w.rate

theorem find_first_group_men (m w : WorkRate) : ∃ x : ℕ, 
  totalWork m w ⟨x, 8⟩ = totalWork m w ⟨6, 2⟩ ∧
  2 * totalWork m w ⟨2, 3⟩ = totalWork m w ⟨x, 8⟩ ∧
  x = 3 := by
  sorry

#check find_first_group_men

end find_first_group_men_l2494_249465


namespace alberto_clara_distance_difference_l2494_249407

/-- The difference in distance traveled between two bikers over a given time -/
def distance_difference (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 * time) - (speed2 * time)

/-- Theorem stating the difference in distance traveled between Alberto and Clara -/
theorem alberto_clara_distance_difference :
  distance_difference 16 12 5 = 20 := by
  sorry

end alberto_clara_distance_difference_l2494_249407


namespace hole_filling_problem_l2494_249494

/-- The amount of additional water needed to fill a hole -/
def additional_water_needed (total_water : ℕ) (initial_water : ℕ) : ℕ :=
  total_water - initial_water

/-- Theorem stating the additional water needed to fill the hole -/
theorem hole_filling_problem (total_water : ℕ) (initial_water : ℕ)
    (h1 : total_water = 823)
    (h2 : initial_water = 676) :
    additional_water_needed total_water initial_water = 147 := by
  sorry

end hole_filling_problem_l2494_249494


namespace cistern_fill_time_l2494_249405

-- Define the time to fill without leak
def T : ℝ := 12

-- Define the time to fill with leak
def time_with_leak : ℝ := T + 2

-- Define the time to empty when full
def time_to_empty : ℝ := 84

-- State the theorem
theorem cistern_fill_time :
  (1 / T - 1 / time_to_empty = 1 / time_with_leak) ∧
  (T > 0) :=
sorry

end cistern_fill_time_l2494_249405


namespace tangent_product_equals_two_l2494_249466

theorem tangent_product_equals_two :
  (1 + Real.tan (20 * π / 180)) * (1 + Real.tan (25 * π / 180)) = 2 := by
  sorry

end tangent_product_equals_two_l2494_249466


namespace difference_from_averages_l2494_249485

theorem difference_from_averages (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) : 
  c - a = 90 := by
sorry

end difference_from_averages_l2494_249485


namespace face_card_then_heart_probability_l2494_249443

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of face cards in a standard deck -/
def FaceCards : ℕ := 12

/-- Number of hearts in a standard deck -/
def Hearts : ℕ := 13

/-- Number of face cards that are hearts -/
def FaceHearts : ℕ := 3

/-- Probability of drawing a face card followed by a heart from a standard deck -/
theorem face_card_then_heart_probability :
  (FaceCards / StandardDeck) * (Hearts / (StandardDeck - 1)) = 19 / 210 :=
sorry

end face_card_then_heart_probability_l2494_249443


namespace x_minus_y_equals_eight_l2494_249464

theorem x_minus_y_equals_eight (x y : ℝ) (h1 : 4 = 0.25 * x) (h2 : 4 = 0.50 * y) : x - y = 8 := by
  sorry

end x_minus_y_equals_eight_l2494_249464


namespace rectangle_area_rectangle_area_is_270_l2494_249471

theorem rectangle_area : ℕ → Prop :=
  fun area =>
    ∃ (square_side : ℕ) (length breadth : ℕ),
      square_side * square_side = 2025 ∧
      length = (2 * square_side) / 5 ∧
      breadth = length / 2 + 5 ∧
      (length + breadth) % 3 = 0 ∧
      length * breadth = area

theorem rectangle_area_is_270 : rectangle_area 270 := by
  sorry

end rectangle_area_rectangle_area_is_270_l2494_249471


namespace units_digit_of_n_l2494_249499

/-- Returns the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Given two natural numbers m and n, returns true if m has a units digit of 4 -/
def hasUnitsDigitFour (m : ℕ) : Prop := unitsDigit m = 4

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : hasUnitsDigitFour m) :
  unitsDigit n = 4 := by sorry

end units_digit_of_n_l2494_249499


namespace similar_triangles_side_length_l2494_249496

/-- Represents a triangle with an area and side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- Given two similar triangles, proves that the corresponding side of the larger triangle is 15 feet -/
theorem similar_triangles_side_length 
  (t1 t2 : Triangle) 
  (h_area_diff : t1.area - t2.area = 50)
  (h_area_ratio : t1.area / t2.area = 9)
  (h_t2_area_int : ∃ n : ℕ, t2.area = n)
  (h_t2_side : t2.side = 5) :
  t1.side = 15 := by
  sorry

end similar_triangles_side_length_l2494_249496


namespace proportion_estimate_correct_l2494_249418

/-- Proportion of households with 3+ housing sets -/
def proportion_with_3plus_housing (total_households : ℕ) 
  (ordinary_households : ℕ) (high_income_households : ℕ)
  (sampled_ordinary : ℕ) (sampled_high_income : ℕ)
  (sampled_ordinary_with_3plus : ℕ) (sampled_high_income_with_3plus : ℕ) : ℚ :=
  let estimated_ordinary_with_3plus := (sampled_ordinary_with_3plus : ℚ) * ordinary_households / sampled_ordinary
  let estimated_high_income_with_3plus := (sampled_high_income_with_3plus : ℚ) * high_income_households / sampled_high_income
  (estimated_ordinary_with_3plus + estimated_high_income_with_3plus) / total_households

theorem proportion_estimate_correct : 
  proportion_with_3plus_housing 100000 99000 1000 990 100 40 80 = 48 / 1000 := by
  sorry

end proportion_estimate_correct_l2494_249418


namespace wall_volume_is_12_8_l2494_249493

/-- Calculates the volume of a wall given its dimensions --/
def wall_volume (breadth : ℝ) : ℝ :=
  let height := 5 * breadth
  let length := 8 * height
  breadth * height * length

/-- Theorem stating that the volume of the wall with given dimensions is 12.8 cubic meters --/
theorem wall_volume_is_12_8 :
  wall_volume (40 / 100) = 12.8 := by sorry

end wall_volume_is_12_8_l2494_249493


namespace sqrt_two_difference_product_l2494_249467

theorem sqrt_two_difference_product : (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 1 := by
  sorry

end sqrt_two_difference_product_l2494_249467


namespace real_part_of_z_l2494_249488

theorem real_part_of_z (i : ℂ) (h : i^2 = -1) : Complex.re ((1 + 2*i)^2) = -3 := by
  sorry

end real_part_of_z_l2494_249488


namespace divisibility_property_l2494_249421

theorem divisibility_property (p m n : ℕ) : 
  Nat.Prime p → 
  p % 2 = 1 →
  m > 1 → 
  n > 0 → 
  Nat.Prime ((m^(p*n) - 1) / (m^n - 1)) → 
  (p * n) ∣ ((p - 1)^n + 1) := by
sorry

end divisibility_property_l2494_249421


namespace marbles_given_correct_l2494_249432

/-- The number of marbles given to the brother -/
def marbles_given : ℕ := 2

/-- The initial number of marbles you have -/
def initial_marbles : ℕ := 16

/-- The total number of marbles among all three people -/
def total_marbles : ℕ := 63

theorem marbles_given_correct :
  -- After giving marbles, you have double your brother's marbles
  2 * ((initial_marbles - marbles_given) / 2) = initial_marbles - marbles_given ∧
  -- Your friend has triple your marbles after giving
  3 * (initial_marbles - marbles_given) = 
    total_marbles - (initial_marbles - marbles_given) - ((initial_marbles - marbles_given) / 2) :=
by sorry

end marbles_given_correct_l2494_249432


namespace ellipse_properties_l2494_249491

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_minor_axis : b = Real.sqrt 3
  h_eccentricity : Real.sqrt (a^2 - b^2) / a = 1/2

/-- The standard form of the ellipse -/
def standard_form (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2/4 + y^2/3 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1

/-- The maximum area of triangle F₁AB -/
def max_triangle_area (e : Ellipse) : Prop :=
  ∃ (max_area : ℝ),
    max_area = 3 ∧
    ∀ (A B : ℝ × ℝ),
      A ≠ B →
      (∃ (m : ℝ), (A.1 = m * A.2 + 1 ∧ A.1^2/e.a^2 + A.2^2/e.b^2 = 1) ∧
                  (B.1 = m * B.2 + 1 ∧ B.1^2/e.a^2 + B.2^2/e.b^2 = 1)) →
      abs (A.2 - B.2) ≤ max_area

theorem ellipse_properties (e : Ellipse) :
  standard_form e ∧ max_triangle_area e := by sorry

end ellipse_properties_l2494_249491


namespace tina_sold_26_more_than_katya_l2494_249403

/-- The number of glasses of lemonade sold by Katya -/
def katya_sales : ℕ := 8

/-- The number of glasses of lemonade sold by Ricky -/
def ricky_sales : ℕ := 9

/-- The number of glasses of lemonade sold by Tina -/
def tina_sales : ℕ := 2 * (katya_sales + ricky_sales)

/-- Theorem: Tina sold 26 more glasses of lemonade than Katya -/
theorem tina_sold_26_more_than_katya : tina_sales - katya_sales = 26 := by
  sorry

end tina_sold_26_more_than_katya_l2494_249403


namespace triangle_segment_equality_l2494_249469

theorem triangle_segment_equality (AB AC : ℝ) (n : ℕ) :
  AB = 33 →
  AC = 21 →
  (∃ (D E : ℝ), 0 ≤ D ∧ D ≤ AB ∧ 0 ≤ E ∧ E ≤ AC ∧ D = n ∧ AB - D = n ∧ E = n ∧ AC - E = n) →
  (∃ (BC : ℕ), BC = 30) :=
by sorry

end triangle_segment_equality_l2494_249469


namespace remaining_doughnuts_theorem_l2494_249450

/-- Represents the types of doughnuts -/
inductive DoughnutType
  | Glazed
  | Chocolate
  | RaspberryFilled

/-- Represents a person who ate doughnuts -/
structure Person where
  glazed : Nat
  chocolate : Nat
  raspberryFilled : Nat

/-- Calculates the remaining doughnuts after consumption -/
def remainingDoughnuts (initial : DoughnutType → Nat) (people : List Person) : DoughnutType → Nat :=
  fun type =>
    initial type - (people.map fun p =>
      match type with
      | DoughnutType.Glazed => p.glazed
      | DoughnutType.Chocolate => p.chocolate
      | DoughnutType.RaspberryFilled => p.raspberryFilled
    ).sum

/-- The main theorem stating the remaining quantities of doughnuts -/
theorem remaining_doughnuts_theorem (initial : DoughnutType → Nat) (people : List Person)
  (h_initial_glazed : initial DoughnutType.Glazed = 10)
  (h_initial_chocolate : initial DoughnutType.Chocolate = 8)
  (h_initial_raspberry : initial DoughnutType.RaspberryFilled = 6)
  (h_people : people = [
    ⟨2, 1, 0⟩, -- Person A
    ⟨1, 0, 0⟩, -- Person B
    ⟨0, 3, 0⟩, -- Person C
    ⟨1, 0, 1⟩, -- Person D
    ⟨0, 0, 1⟩, -- Person E
    ⟨0, 0, 2⟩  -- Person F
  ]) :
  (remainingDoughnuts initial people DoughnutType.Glazed = 6) ∧
  (remainingDoughnuts initial people DoughnutType.Chocolate = 4) ∧
  (remainingDoughnuts initial people DoughnutType.RaspberryFilled = 2) :=
by sorry


end remaining_doughnuts_theorem_l2494_249450


namespace triangle_existence_l2494_249483

/-- Given a square area t, segment length 2s, and angle α, 
    this theorem states the existence condition for a triangle 
    with area t, perimeter 2s, and one angle α. -/
theorem triangle_existence 
  (t s : ℝ) (α : Real) 
  (h_t : t > 0) (h_s : s > 0) (h_α : 0 < α ∧ α < π) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 2 * s ∧
    1/2 * a * b * Real.sin α = t ∧
    ∃ (β γ : Real), 
      β > 0 ∧ γ > 0 ∧
      α + β + γ = π ∧
      a / Real.sin α = b / Real.sin β ∧
      b / Real.sin β = c / Real.sin γ :=
sorry

end triangle_existence_l2494_249483


namespace white_dandelions_on_saturday_l2494_249413

/-- Represents the state of dandelions on a given day -/
structure DandelionState :=
  (yellow : ℕ)
  (white : ℕ)

/-- Represents the lifecycle of a dandelion -/
def dandelionLifecycle : ℕ := 5

/-- The number of days a dandelion is yellow -/
def yellowDays : ℕ := 3

/-- The number of days a dandelion is white -/
def whiteDays : ℕ := 2

/-- Calculates the number of white dandelions on Saturday given the states on Monday and Wednesday -/
def whiteDandelionsOnSaturday (monday : DandelionState) (wednesday : DandelionState) : ℕ :=
  (wednesday.yellow + wednesday.white) - monday.yellow

theorem white_dandelions_on_saturday 
  (monday : DandelionState) 
  (wednesday : DandelionState) 
  (h1 : monday.yellow = 20)
  (h2 : monday.white = 14)
  (h3 : wednesday.yellow = 15)
  (h4 : wednesday.white = 11) :
  whiteDandelionsOnSaturday monday wednesday = 6 := by
  sorry

#check white_dandelions_on_saturday

end white_dandelions_on_saturday_l2494_249413


namespace polynomial_evaluation_l2494_249459

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 10 = 0) : 
  x^3 - 3*x^2 - 10*x + 5 = 5 := by
  sorry

end polynomial_evaluation_l2494_249459


namespace power_function_property_l2494_249452

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 9 / f 3 = 2) : 
  f (1/9) = 1/4 := by
sorry

end power_function_property_l2494_249452


namespace brendans_hourly_wage_l2494_249440

-- Define Brendan's work schedule
def hours_per_week : ℕ := 2 * 8 + 1 * 12

-- Define Brendan's hourly tip rate
def hourly_tips : ℚ := 12

-- Define the fraction of tips reported to IRS
def reported_tips_fraction : ℚ := 1 / 3

-- Define the tax rate
def tax_rate : ℚ := 1 / 5

-- Define the weekly tax amount
def weekly_tax : ℚ := 56

-- Theorem to prove Brendan's hourly wage
theorem brendans_hourly_wage :
  ∃ (hourly_wage : ℚ),
    hourly_wage * hours_per_week +
    reported_tips_fraction * (hourly_tips * hours_per_week) =
    weekly_tax / tax_rate ∧
    hourly_wage = 6 := by
  sorry

end brendans_hourly_wage_l2494_249440


namespace previous_day_visitor_count_l2494_249439

/-- The number of visitors to Buckingham Palace on the current day -/
def current_day_visitors : ℕ := 661

/-- The difference in visitors between the current day and the previous day -/
def visitor_difference : ℕ := 61

/-- The number of visitors on the previous day -/
def previous_day_visitors : ℕ := current_day_visitors - visitor_difference

theorem previous_day_visitor_count : previous_day_visitors = 600 := by
  sorry

end previous_day_visitor_count_l2494_249439


namespace carla_bob_payment_difference_l2494_249458

/-- Represents the pizza and its properties -/
structure Pizza :=
  (total_slices : ℕ)
  (vegetarian_slices : ℕ)
  (plain_cost : ℚ)
  (vegetarian_extra_cost : ℚ)

/-- Calculates the cost per slice of the pizza -/
def cost_per_slice (p : Pizza) : ℚ :=
  (p.plain_cost + p.vegetarian_extra_cost) / p.total_slices

/-- Calculates the cost for a given number of slices -/
def cost_for_slices (p : Pizza) (slices : ℕ) : ℚ :=
  (cost_per_slice p) * slices

/-- The main theorem to prove -/
theorem carla_bob_payment_difference
  (p : Pizza)
  (carla_slices bob_slices : ℕ)
  : p.total_slices = 12 →
    p.vegetarian_slices = 6 →
    p.plain_cost = 10 →
    p.vegetarian_extra_cost = 3 →
    carla_slices = 8 →
    bob_slices = 3 →
    (cost_for_slices p carla_slices) - (cost_for_slices p bob_slices) = 5.42 := by
  sorry

end carla_bob_payment_difference_l2494_249458


namespace smallest_result_l2494_249479

def S : Finset Nat := {2, 3, 4, 6, 8, 9}

def process (a b c : Nat) : Nat :=
  max (max ((a + b) * c) ((a + c) * b)) ((b + c) * a)

theorem smallest_result :
  ∃ (a b c : Nat), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  process a b c = 14 ∧
  ∀ (x y z : Nat), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  process x y z ≥ 14 :=
sorry

end smallest_result_l2494_249479


namespace odd_fraction_in_multiplication_table_l2494_249468

def table_size : Nat := 16

theorem odd_fraction_in_multiplication_table :
  let total_products := table_size * table_size
  let odd_products := (table_size / 2) * (table_size / 2)
  (odd_products : ℚ) / total_products = 1 / 4 := by
  sorry

end odd_fraction_in_multiplication_table_l2494_249468


namespace candy_distribution_theorem_l2494_249441

/-- The number of candy pieces -/
def total_candy : ℕ := 108

/-- Predicate to check if a number divides the total candy evenly -/
def divides_candy (n : ℕ) : Prop := total_candy % n = 0

/-- Predicate to check if a number is a valid student count -/
def valid_student_count (n : ℕ) : Prop :=
  n > 1 ∧ divides_candy n

/-- The set of possible student counts -/
def possible_student_counts : Set ℕ := {12, 36, 54}

/-- Theorem stating that the possible student counts are correct -/
theorem candy_distribution_theorem :
  ∀ n : ℕ, n ∈ possible_student_counts ↔ valid_student_count n :=
by sorry

end candy_distribution_theorem_l2494_249441


namespace sum_of_decimals_l2494_249431

theorem sum_of_decimals :
  5.46 + 2.793 + 3.1 = 11.353 := by
  sorry

end sum_of_decimals_l2494_249431


namespace unique_p_q_l2494_249463

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B (p q : ℝ) : Set ℝ := {x | x^2 + p*x + q ≤ 0}

-- State the theorem
theorem unique_p_q : 
  ∃! (p q : ℝ), 
    (A ∪ B p q = Set.univ) ∧ 
    (A ∩ B p q = Set.Icc (-2) (-1)) ∧
    p = -1 ∧ 
    q = -6 := by sorry

end unique_p_q_l2494_249463


namespace science_club_team_selection_l2494_249442

theorem science_club_team_selection (n : ℕ) (k : ℕ) :
  n = 22 → k = 8 → Nat.choose n k = 319770 := by
  sorry

end science_club_team_selection_l2494_249442
