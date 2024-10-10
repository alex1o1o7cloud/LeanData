import Mathlib

namespace collatz_7_11_collatz_10_probability_l1898_189830

-- Define the Collatz operation
def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

-- Define the Collatz sequence
def collatz_seq (a₀ : ℕ) : ℕ → ℕ
  | 0 => a₀
  | n + 1 => collatz (collatz_seq a₀ n)

-- Statement 1: When a₀ = 7, a₁₁ = 5
theorem collatz_7_11 : collatz_seq 7 11 = 5 := by sorry

-- Helper function to check if a number is odd
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

-- Statement 2: When a₀ = 10, the probability of randomly selecting two numbers
-- from aᵢ (i = 1,2,3,4,5,6), at least one of which is odd, is 3/5
theorem collatz_10_probability :
  let seq := List.range 6 |> List.map (collatz_seq 10)
  let total_pairs := seq.length.choose 2
  let odd_pairs := (seq.filterMap (fun n => if is_odd n then some n else none)).length
  (total_pairs - (seq.length - odd_pairs).choose 2) / total_pairs = 3 / 5 := by sorry

end collatz_7_11_collatz_10_probability_l1898_189830


namespace rectangle_breadth_l1898_189885

/-- 
Given a rectangle where:
1. The area is 24 times its breadth
2. The difference between the length and the breadth is 10 meters
Prove that the breadth is 14 meters
-/
theorem rectangle_breadth (length breadth : ℝ) 
  (h1 : length * breadth = 24 * breadth) 
  (h2 : length - breadth = 10) : 
  breadth = 14 := by
sorry

end rectangle_breadth_l1898_189885


namespace cube_with_holes_surface_area_l1898_189898

/-- Calculates the total surface area of a cube with holes cut through each face --/
def total_surface_area (cube_edge : ℝ) (hole_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_side^2
  let exposed_area := 6 * 4 * hole_side^2
  original_surface_area - hole_area + exposed_area

/-- Theorem stating that the total surface area of the given cube with holes is 222 square meters --/
theorem cube_with_holes_surface_area :
  total_surface_area 5 2 = 222 := by
  sorry

#eval total_surface_area 5 2

end cube_with_holes_surface_area_l1898_189898


namespace necklace_price_l1898_189862

def total_cost : ℕ := 240000
def necklace_count : ℕ := 3

theorem necklace_price (necklace_price : ℕ) 
  (h1 : necklace_count * necklace_price + 3 * necklace_price = total_cost) :
  necklace_price = 40000 := by
  sorry

end necklace_price_l1898_189862


namespace sphere_surface_area_l1898_189881

theorem sphere_surface_area (r : ℝ) (h : r > 0) :
  let plane_distance : ℝ := 3
  let section_area : ℝ := 16 * Real.pi
  let section_radius : ℝ := (section_area / Real.pi).sqrt
  r * r = plane_distance * plane_distance + section_radius * section_radius →
  4 * Real.pi * r * r = 100 * Real.pi := by
  sorry

end sphere_surface_area_l1898_189881


namespace harveys_steak_sales_l1898_189887

/-- Calculates the total number of steaks sold given the initial count, 
    the count after the first sale, and the count of the second sale. -/
def total_steaks_sold (initial : Nat) (after_first_sale : Nat) (second_sale : Nat) : Nat :=
  (initial - after_first_sale) + second_sale

/-- Theorem stating that given Harvey's specific situation, 
    the total number of steaks sold is 17. -/
theorem harveys_steak_sales : 
  total_steaks_sold 25 12 4 = 17 := by
  sorry

end harveys_steak_sales_l1898_189887


namespace cupcake_packages_l1898_189870

/-- Given the initial number of cupcakes, the number of eaten cupcakes, and the number of cupcakes per package,
    calculate the number of complete packages that can be made. -/
def calculate_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package

/-- Theorem: Given 20 initial cupcakes, 11 eaten cupcakes, and 3 cupcakes per package,
    the number of complete packages that can be made is 3. -/
theorem cupcake_packages : calculate_packages 20 11 3 = 3 := by
  sorry

end cupcake_packages_l1898_189870


namespace candy_sampling_theorem_l1898_189897

theorem candy_sampling_theorem (caught_percentage : Real) (total_sampling_percentage : Real)
  (h1 : caught_percentage = 22)
  (h2 : total_sampling_percentage = 24.444444444444443) :
  total_sampling_percentage - caught_percentage = 2.444444444444443 := by
  sorry

end candy_sampling_theorem_l1898_189897


namespace complex_equation_solution_l1898_189863

theorem complex_equation_solution (a : ℝ) (i : ℂ) : 
  i * i = -1 →
  (a^2 - a : ℂ) + (3*a - 1 : ℂ) * i = 2 + 5*i →
  a = 2 := by
sorry

end complex_equation_solution_l1898_189863


namespace systematic_sampling_interval_count_l1898_189807

theorem systematic_sampling_interval_count 
  (total_employees : ℕ) 
  (sample_size : ℕ) 
  (interval_start : ℕ) 
  (interval_end : ℕ) 
  (h1 : total_employees = 840)
  (h2 : sample_size = 42)
  (h3 : interval_start = 481)
  (h4 : interval_end = 720) :
  (interval_end - interval_start + 1) / (total_employees / sample_size) = 12 := by
  sorry

end systematic_sampling_interval_count_l1898_189807


namespace quadratic_j_value_l1898_189837

-- Define the quadratic function
def quadratic (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

-- State the theorem
theorem quadratic_j_value (p q r : ℝ) :
  (∃ m n : ℝ, ∀ x : ℝ, 4 * (quadratic p q r x) = m * (x - 5)^2 + n) →
  (∃ m n : ℝ, ∀ x : ℝ, quadratic p q r x = 3 * (x - 5)^2 + 15) :=
by sorry

end quadratic_j_value_l1898_189837


namespace one_pole_inside_l1898_189884

/-- Represents a non-convex polygon fence -/
structure Fence where
  is_non_convex : Bool

/-- Represents a power line with poles -/
structure PowerLine where
  total_poles : Nat

/-- Represents a spy walking around the fence -/
structure Spy where
  counted_poles : Nat

/-- Theorem stating that given the conditions, there is one pole inside the fence -/
theorem one_pole_inside (fence : Fence) (power_line : PowerLine) (spy : Spy) :
  fence.is_non_convex ∧
  power_line.total_poles = 36 ∧
  spy.counted_poles = 2015 →
  ∃ (poles_inside : Nat), poles_inside = 1 :=
sorry

end one_pole_inside_l1898_189884


namespace barking_ratio_is_one_fourth_l1898_189873

/-- Represents the state of dogs in a park -/
structure DogPark where
  total : ℕ
  running : ℕ
  playing : ℕ
  idle : ℕ

/-- The ratio of barking dogs to total dogs -/
def barkingRatio (park : DogPark) : Rat :=
  let barking := park.total - (park.running + park.playing + park.idle)
  barking / park.total

/-- Theorem stating the barking ratio in the given scenario -/
theorem barking_ratio_is_one_fourth :
  ∃ (park : DogPark),
    park.total = 88 ∧
    park.running = 12 ∧
    park.playing = 44 ∧
    park.idle = 10 ∧
    barkingRatio park = 1 / 4 := by
  sorry


end barking_ratio_is_one_fourth_l1898_189873


namespace common_point_and_tangent_l1898_189834

theorem common_point_and_tangent (t : ℝ) (h : t ≠ 0) :
  let f := fun x : ℝ => x^3 + a*x
  let g := fun x : ℝ => b*x^2 + c
  let f' := fun x : ℝ => 3*x^2 + a
  let g' := fun x : ℝ => 2*b*x
  f t = 0 ∧ g t = 0 ∧ f' t = g' t →
  a = -t^2 ∧ b = t ∧ c = -t^3 :=
by sorry

end common_point_and_tangent_l1898_189834


namespace circular_bead_arrangements_l1898_189804

/-- The number of red beads -/
def num_red : ℕ := 3

/-- The number of blue beads -/
def num_blue : ℕ := 2

/-- The total number of beads -/
def total_beads : ℕ := num_red + num_blue

/-- The symmetry group of the circular arrangement -/
def symmetry_group : ℕ := 2 * total_beads

/-- The number of fixed arrangements under the identity rotation -/
def fixed_identity : ℕ := (total_beads.choose num_red)

/-- The number of fixed arrangements under each reflection -/
def fixed_reflection : ℕ := 2

/-- The number of reflections in the symmetry group -/
def num_reflections : ℕ := total_beads

/-- The total number of fixed arrangements under all symmetries -/
def total_fixed : ℕ := fixed_identity + num_reflections * fixed_reflection

/-- The number of distinct arrangements of beads on the circular ring -/
def distinct_arrangements : ℕ := total_fixed / symmetry_group

theorem circular_bead_arrangements :
  distinct_arrangements = 2 :=
sorry

end circular_bead_arrangements_l1898_189804


namespace crayons_distribution_l1898_189899

/-- Given a total number of crayons and a number of boxes, 
    calculate the number of crayons per box -/
def crayons_per_box (total_crayons : ℕ) (num_boxes : ℕ) : ℕ :=
  total_crayons / num_boxes

/-- Theorem stating that given 80 crayons and 10 boxes, 
    the number of crayons per box is 8 -/
theorem crayons_distribution :
  crayons_per_box 80 10 = 8 := by
  sorry

end crayons_distribution_l1898_189899


namespace division_result_l1898_189861

theorem division_result (h : 144 * 177 = 25488) : 254.88 / 0.177 = 1440 := by
  sorry

end division_result_l1898_189861


namespace symmetric_points_count_l1898_189877

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^2 + 4 * x + 1 else 2 / Real.exp x

theorem symmetric_points_count :
  ∃! (p : ℕ), p = 2 ∧
  ∃ (S : Finset (ℝ × ℝ)),
    S.card = p ∧
    (∀ (x y : ℝ), (x, y) ∈ S → y = f x) ∧
    (∀ (x y : ℝ), (x, y) ∈ S → (-x, -y) ∈ S) ∧
    (∀ (x y : ℝ), (x, y) ∈ S → (x ≠ 0 ∨ y ≠ 0)) :=
by sorry

end symmetric_points_count_l1898_189877


namespace no_existence_of_complex_numbers_l1898_189847

theorem no_existence_of_complex_numbers : ¬∃ (a b c : ℂ) (h : ℕ), 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ 
  (∀ (k l m : ℤ), (abs k + abs l + abs m ≥ 1996) → 
    Complex.abs (1 + k • a + l • b + m • c) > 1 / h) := by
  sorry


end no_existence_of_complex_numbers_l1898_189847


namespace tenth_term_of_specific_sequence_l1898_189842

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1) * seq.diff

/-- The property of the arithmetic sequence we're interested in -/
def hasSpecificTerms (seq : ArithmeticSequence) : Prop :=
  seq.nthTerm 5 = 26 ∧ seq.nthTerm 8 = 50

theorem tenth_term_of_specific_sequence 
  (seq : ArithmeticSequence) 
  (h : hasSpecificTerms seq) : 
  seq.nthTerm 10 = 66 := by
  sorry

end tenth_term_of_specific_sequence_l1898_189842


namespace city_male_population_l1898_189810

theorem city_male_population (total_population : ℕ) (num_parts : ℕ) (male_parts : ℕ) :
  total_population = 1000 →
  num_parts = 5 →
  male_parts = 2 →
  (total_population / num_parts) * male_parts = 400 := by
sorry

end city_male_population_l1898_189810


namespace specific_structure_surface_area_l1898_189886

/-- Represents a complex structure composed of unit cubes -/
structure CubeStructure where
  num_cubes : ℕ
  height : ℕ
  length : ℕ
  width : ℕ

/-- Calculates the surface area of a cube structure -/
def surface_area (s : CubeStructure) : ℕ :=
  2 * (s.length * s.width + s.length * s.height + s.width * s.height)

/-- Theorem stating that a specific cube structure has a surface area of 84 square units -/
theorem specific_structure_surface_area :
  ∃ (s : CubeStructure), s.num_cubes = 15 ∧ s.height = 4 ∧ s.length = 5 ∧ s.width = 3 ∧
  surface_area s = 84 :=
sorry

end specific_structure_surface_area_l1898_189886


namespace rest_stop_distance_l1898_189857

/-- Proves that the distance between rest stops is 10 miles for a man walking 50 miles in 320 minutes with given conditions. -/
theorem rest_stop_distance (walking_speed : ℝ) (rest_duration : ℝ) (total_distance : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 10) -- walking speed in mph
  (h2 : rest_duration = 5 / 60) -- rest duration in hours
  (h3 : total_distance = 50) -- total distance in miles
  (h4 : total_time = 320 / 60) -- total time in hours
  : ∃ (x : ℝ), x = 10 ∧ 
    (total_distance / walking_speed + rest_duration * (total_distance / x - 1) = total_time) := by
  sorry


end rest_stop_distance_l1898_189857


namespace competition_scores_l1898_189833

theorem competition_scores (n d : ℕ) : 
  n > 1 → 
  d > 0 → 
  d * (n * (n + 1)) / 2 = 26 * n → 
  ((n = 3 ∧ d = 13) ∨ (n = 12 ∧ d = 4) ∨ (n = 25 ∧ d = 2)) := by
  sorry

end competition_scores_l1898_189833


namespace line_intersection_y_axis_l1898_189806

-- Define a line by two points
def Line (x₁ y₁ x₂ y₂ : ℝ) := {(x, y) : ℝ × ℝ | ∃ t, x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁)}

-- Define the y-axis
def YAxis := {(x, y) : ℝ × ℝ | x = 0}

-- Theorem statement
theorem line_intersection_y_axis :
  ∃! p : ℝ × ℝ, p ∈ Line 2 9 4 15 ∧ p ∈ YAxis ∧ p = (0, 3) := by
  sorry

end line_intersection_y_axis_l1898_189806


namespace chess_group_players_l1898_189865

theorem chess_group_players (n : ℕ) : n > 0 →
  (n * (n - 1)) / 2 = 21 → n = 7 := by
  sorry

end chess_group_players_l1898_189865


namespace four_distinct_roots_l1898_189840

/-- The equation x^2 - 4|x| + 5 = m has four distinct real roots if and only if 1 < m < 5 -/
theorem four_distinct_roots (m : ℝ) :
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^2 - 4 * |x| + 5 = m ↔ x = a ∨ x = b ∨ x = c ∨ x = d)) ↔
  1 < m ∧ m < 5 := by
  sorry

end four_distinct_roots_l1898_189840


namespace chord_equation_through_midpoint_l1898_189878

/-- The equation of a line containing a chord of an ellipse, where the chord passes through a given point and has that point as its midpoint. -/
theorem chord_equation_through_midpoint (x y : ℝ) :
  (4 * x^2 + 9 * y^2 = 144) →  -- Ellipse equation
  (3 : ℝ)^2 * 4 + 2^2 * 9 < 144 →  -- Point (3, 2) is inside the ellipse
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (4 * x₁^2 + 9 * y₁^2 = 144) ∧  -- Point (x₁, y₁) is on the ellipse
    (4 * x₂^2 + 9 * y₂^2 = 144) ∧  -- Point (x₂, y₂) is on the ellipse
    (x₁ + x₂) / 2 = 3 ∧  -- (3, 2) is the midpoint of (x₁, y₁) and (x₂, y₂)
    (y₁ + y₂) / 2 = 2 ∧
    2 * x + 3 * y - 12 = 0  -- Equation of the line containing the chord
  := by sorry

end chord_equation_through_midpoint_l1898_189878


namespace geometric_sequence_seventh_term_l1898_189815

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_sum1 : a 1 + a 2 = 3)
  (h_sum2 : a 2 + a 3 = 6) :
  a 7 = 64 := by
sorry

end geometric_sequence_seventh_term_l1898_189815


namespace original_ratio_l1898_189889

theorem original_ratio (x y : ℕ) (h1 : y = 48) (h2 : (x + 12) / y = 1/2) : x / y = 1/4 := by
  sorry

end original_ratio_l1898_189889


namespace real_part_reciprocal_l1898_189838

theorem real_part_reciprocal (z : ℂ) (h1 : z ≠ (z.re : ℂ)) (h2 : Complex.abs z = 2) :
  ((2 - z)⁻¹).re = (1 : ℝ) / 2 := by
  sorry

end real_part_reciprocal_l1898_189838


namespace candy_necklace_packs_opened_l1898_189895

/-- Proves the number of candy necklace packs Emily opened for her classmates -/
theorem candy_necklace_packs_opened
  (total_packs : ℕ)
  (necklaces_per_pack : ℕ)
  (necklaces_left : ℕ)
  (h1 : total_packs = 9)
  (h2 : necklaces_per_pack = 8)
  (h3 : necklaces_left = 40) :
  (total_packs * necklaces_per_pack - necklaces_left) / necklaces_per_pack = 4 :=
by sorry

end candy_necklace_packs_opened_l1898_189895


namespace arithmetic_sequence_sum_l1898_189820

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 32 →
  a 4 + a 5 = 55 := by
sorry

end arithmetic_sequence_sum_l1898_189820


namespace turtle_fraction_l1898_189821

theorem turtle_fraction (trey kris kristen : ℕ) : 
  trey = 7 * kris →
  trey = kristen + 9 →
  kristen = 12 →
  kris / kristen = 1 / 4 := by
sorry

end turtle_fraction_l1898_189821


namespace price_A_base_correct_minimum_amount_spent_l1898_189826

-- Define the price of type A seedlings at the base
def price_A_base : ℝ := 20

-- Define the price of type B seedlings at the base
def price_B_base : ℝ := 30

-- Define the total number of bundles to purchase
def total_bundles : ℕ := 100

-- Define the discount rate
def discount_rate : ℝ := 0.9

-- Theorem for part 1
theorem price_A_base_correct :
  price_A_base * (300 / price_A_base) = 
  (5/4 * price_A_base) * (300 / (5/4 * price_A_base) + 3) := by sorry

-- Theorem for part 2
theorem minimum_amount_spent :
  let m := min (total_bundles / 2) total_bundles
  ∃ (n : ℕ), n ≤ total_bundles - n ∧
    discount_rate * (price_A_base * m + price_B_base * (total_bundles - m)) = 2250 := by sorry

end price_A_base_correct_minimum_amount_spent_l1898_189826


namespace special_polynomial_form_l1898_189871

/-- A polynomial of two variables satisfying specific conditions -/
structure SpecialPolynomial where
  P : ℝ → ℝ → ℝ
  n : ℕ+
  homogeneous : ∀ (t x y : ℝ), P (t * x) (t * y) = t ^ n.val * P x y
  sum_condition : ∀ (a b c : ℝ), P (a + b) c + P (b + c) a + P (c + a) b = 0
  normalization : P 1 0 = 1

/-- The theorem stating the form of the special polynomial -/
theorem special_polynomial_form (sp : SpecialPolynomial) :
  ∃ (n : ℕ+), ∀ (x y : ℝ), sp.P x y = (x - 2 * y) * (x + y) ^ (n.val - 1) := by
  sorry

end special_polynomial_form_l1898_189871


namespace no_fermat_in_sequence_l1898_189836

/-- The general term of the second-order arithmetic sequence -/
def a (n k : ℕ) : ℕ := (k - 2) * n * (n - 1) / 2 + n

/-- Fermat number of order m -/
def fermat (m : ℕ) : ℕ := 2^(2^m) + 1

/-- Statement: There are no Fermat numbers in the sequence for k > 2 -/
theorem no_fermat_in_sequence (k : ℕ) (h : k > 2) :
  ∀ (n m : ℕ), a n k ≠ fermat m :=
sorry

end no_fermat_in_sequence_l1898_189836


namespace sum_x_y_l1898_189888

theorem sum_x_y (x y : ℝ) 
  (h1 : |x| + x + y - 2 = 14)
  (h2 : x + |y| - y + 3 = 20) : 
  x + y = 31/5 := by
sorry

end sum_x_y_l1898_189888


namespace trail_length_proof_l1898_189860

/-- The total length of a trail where two friends walk from opposite ends, with one friend 20% faster than the other, and the faster friend walks 12 km when they meet. -/
def trail_length : ℝ := 22

/-- The distance walked by the faster friend when they meet. -/
def faster_friend_distance : ℝ := 12

/-- The ratio of the faster friend's speed to the slower friend's speed. -/
def speed_ratio : ℝ := 1.2

theorem trail_length_proof :
  ∃ (v : ℝ), v > 0 ∧
    trail_length = faster_friend_distance + v * (faster_friend_distance / (speed_ratio * v)) :=
by sorry

end trail_length_proof_l1898_189860


namespace equation_solution_l1898_189852

theorem equation_solution : ∃ x : ℝ, 24 - (4 * 2) = 5 + x ∧ x = 11 := by sorry

end equation_solution_l1898_189852


namespace andy_cake_profit_l1898_189811

/-- Calculates the profit per cake given the total ingredient cost for two cakes,
    the packaging cost per cake, and the selling price per cake. -/
def profit_per_cake (ingredient_cost_for_two : ℚ) (packaging_cost : ℚ) (selling_price : ℚ) : ℚ :=
  selling_price - (ingredient_cost_for_two / 2 + packaging_cost)

/-- Theorem stating that for Andy's cake business, given the specific costs and selling price,
    the profit per cake is $8. -/
theorem andy_cake_profit :
  profit_per_cake 12 1 15 = 8 := by
  sorry

end andy_cake_profit_l1898_189811


namespace complex_square_roots_l1898_189855

theorem complex_square_roots (z : ℂ) : 
  z^2 = -45 - 28*I ↔ z = 2 - 7*I ∨ z = -2 + 7*I :=
sorry

end complex_square_roots_l1898_189855


namespace fraction_simplification_l1898_189846

theorem fraction_simplification :
  (5 : ℝ) / (2 * Real.sqrt 27 + 3 * Real.sqrt 12 + Real.sqrt 108) = (5 * Real.sqrt 3) / 54 := by
  sorry

end fraction_simplification_l1898_189846


namespace perimeter_of_non_shaded_region_l1898_189816

/-- A structure representing the geometrical figure described in the problem -/
structure Figure where
  outer_rectangle_length : ℝ
  outer_rectangle_width : ℝ
  small_rectangle_side : ℝ
  shaded_square_side : ℝ
  shaded_rectangle_length : ℝ
  shaded_rectangle_width : ℝ
  shaded_area : ℝ

/-- The theorem statement based on the problem -/
theorem perimeter_of_non_shaded_region
  (fig : Figure)
  (h1 : fig.outer_rectangle_length = 12)
  (h2 : fig.outer_rectangle_width = 9)
  (h3 : fig.small_rectangle_side = 3)
  (h4 : fig.shaded_square_side = 3)
  (h5 : fig.shaded_rectangle_length = 3)
  (h6 : fig.shaded_rectangle_width = 2)
  (h7 : fig.shaded_area = 65)
  : ∃ (p : ℝ), p = 30 ∧ p = 2 * (12 + 3) :=
sorry

end perimeter_of_non_shaded_region_l1898_189816


namespace classroom_chairs_l1898_189829

theorem classroom_chairs (blue_chairs : ℕ) (green_chairs : ℕ) (white_chairs : ℕ) 
  (h1 : blue_chairs = 10)
  (h2 : green_chairs = 3 * blue_chairs)
  (h3 : white_chairs = blue_chairs + green_chairs - 13) :
  blue_chairs + green_chairs + white_chairs = 67 := by
  sorry

end classroom_chairs_l1898_189829


namespace simplify_and_sum_l1898_189817

theorem simplify_and_sum (d : ℝ) (a b c : ℝ) (h : d ≠ 0) :
  (15 * d + 18 + 12 * d^2) + (5 * d + 2) = a * d + b + c * d^2 →
  a + b + c = 52 := by
  sorry

end simplify_and_sum_l1898_189817


namespace factorization_proof_l1898_189859

theorem factorization_proof (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1)^2 := by
  sorry

end factorization_proof_l1898_189859


namespace equal_x_y_l1898_189841

-- Define the geometric configuration
structure GeometricConfiguration where
  a₁ : ℝ
  a₂ : ℝ
  b₁ : ℝ
  b₂ : ℝ
  x : ℝ
  y : ℝ

-- Define the theorem
theorem equal_x_y (config : GeometricConfiguration) 
  (h1 : config.a₁ = config.a₂) 
  (h2 : config.b₁ = config.b₂) : 
  config.x = config.y := by
  sorry


end equal_x_y_l1898_189841


namespace unique_single_digit_cube_equation_l1898_189893

theorem unique_single_digit_cube_equation :
  ∃! (A : ℕ), A ∈ Finset.range 10 ∧ A ≠ 0 ∧ A^3 = 210 + A :=
by
  -- Proof goes here
  sorry

end unique_single_digit_cube_equation_l1898_189893


namespace polynomial_division_remainder_l1898_189890

theorem polynomial_division_remainder : ∀ (z : ℝ),
  ∃ (r : ℝ),
    3 * z^3 - 4 * z^2 - 14 * z + 3 = (3 * z + 5) * (z^2 - 3 * z + 1/3) + r ∧
    r = 4/3 := by
  sorry

end polynomial_division_remainder_l1898_189890


namespace simplify_fraction_l1898_189801

theorem simplify_fraction : (84 : ℚ) / 1764 * 21 = 1 / 2 := by
  sorry

end simplify_fraction_l1898_189801


namespace circle_condition_l1898_189844

theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x + 2*y - m = 0) ↔ m > -2 := by
  sorry

end circle_condition_l1898_189844


namespace complex_number_value_l1898_189867

theorem complex_number_value (z : ℂ) (h1 : z^2 = 6*z - 27 + 12*I) (h2 : ∃ (n : ℕ), Complex.abs z = n) :
  z = 3 + (Real.sqrt 6 + Real.sqrt 6 * I) ∨ z = 3 - (Real.sqrt 6 + Real.sqrt 6 * I) :=
sorry

end complex_number_value_l1898_189867


namespace add_9999_seconds_to_8am_l1898_189868

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- Converts a natural number to a Time structure -/
def natToTime (n : Nat) : Time :=
  sorry

theorem add_9999_seconds_to_8am (startTime endTime : Time) :
  startTime = { hours := 8, minutes := 0, seconds := 0 } →
  endTime = addSeconds startTime 9999 →
  endTime = { hours := 10, minutes := 46, seconds := 39 } :=
sorry

end add_9999_seconds_to_8am_l1898_189868


namespace consecutive_even_numbers_sum_l1898_189824

theorem consecutive_even_numbers_sum (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4) →  -- a, b, c are consecutive even numbers
  a + b + c = 246 →                                -- their sum is 246
  b = 82                                           -- the second number is 82
:= by sorry

end consecutive_even_numbers_sum_l1898_189824


namespace inequality_proof_l1898_189800

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end inequality_proof_l1898_189800


namespace chip_drawing_probability_l1898_189803

theorem chip_drawing_probability : 
  let total_chips : ℕ := 14
  let tan_chips : ℕ := 5
  let pink_chips : ℕ := 3
  let violet_chips : ℕ := 6
  let favorable_outcomes : ℕ := (Nat.factorial pink_chips) * (Nat.factorial tan_chips) * (Nat.factorial violet_chips)
  let total_outcomes : ℕ := Nat.factorial total_chips
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 168168 := by
sorry

end chip_drawing_probability_l1898_189803


namespace divisibility_problem_l1898_189882

theorem divisibility_problem (n m k : ℕ) (h1 : n = 859722) (h2 : m = 456) (h3 : k = 54) :
  (n + k) % m = 0 :=
sorry

end divisibility_problem_l1898_189882


namespace complement_A_union_B_when_a_3_A_intersect_B_equals_B_iff_l1898_189866

-- Define set A
def A : Set ℝ := {x | x^2 ≤ 3*x + 10}

-- Define set B as a function of a
def B (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Statement 1
theorem complement_A_union_B_when_a_3 :
  (Set.univ \ A) ∪ (B 3) = {x | x ≤ -2 ∨ (4 ≤ x ∧ x ≤ 7)} := by sorry

-- Statement 2
theorem A_intersect_B_equals_B_iff (a : ℝ) :
  A ∩ (B a) = B a ↔ a < 2 := by sorry

end complement_A_union_B_when_a_3_A_intersect_B_equals_B_iff_l1898_189866


namespace library_visitor_average_l1898_189805

/-- Calculates the average number of visitors per day for a month in a library --/
def average_visitors_per_day (sunday_visitors : ℕ) (weekday_visitors : ℕ) 
  (holiday_increase_percent : ℚ) (total_days : ℕ) (sundays : ℕ) (holidays : ℕ) : ℚ :=
  let weekdays := total_days - sundays
  let regular_weekdays := weekdays - holidays
  let holiday_visitors := weekday_visitors * (1 + holiday_increase_percent)
  let total_visitors := sunday_visitors * sundays + 
                        weekday_visitors * regular_weekdays + 
                        holiday_visitors * holidays
  total_visitors / total_days

/-- Theorem stating that the average number of visitors per day is 256 --/
theorem library_visitor_average : 
  average_visitors_per_day 540 240 (1/4) 30 4 4 = 256 := by
  sorry

end library_visitor_average_l1898_189805


namespace statutory_capital_scientific_notation_l1898_189891

/-- The statutory capital of the Asian Infrastructure Investment Bank in U.S. dollars -/
def statutory_capital : ℝ := 100000000000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that the statutory capital in scientific notation is 1 × 10^11 -/
theorem statutory_capital_scientific_notation :
  ∃ (sn : ScientificNotation),
    sn.coefficient = 1 ∧
    sn.exponent = 11 ∧
    statutory_capital = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
by sorry

end statutory_capital_scientific_notation_l1898_189891


namespace door_opening_probability_l1898_189819

/-- Represents the probability of opening a door on the second attempt -/
def probability_second_attempt (total_keys : ℕ) (working_keys : ℕ) (discard : Bool) : ℚ :=
  if discard then
    (working_keys : ℚ) / total_keys * working_keys / (total_keys - 1)
  else
    (working_keys : ℚ) / total_keys * working_keys / total_keys

/-- The main theorem about the probability of opening the door on the second attempt -/
theorem door_opening_probability :
  let total_keys : ℕ := 4
  let working_keys : ℕ := 2
  probability_second_attempt total_keys working_keys true = 1/3 ∧
  probability_second_attempt total_keys working_keys false = 1/4 := by
  sorry

end door_opening_probability_l1898_189819


namespace expression_simplification_l1898_189880

theorem expression_simplification (a : ℝ) (h : a^2 + 3*a - 2 = 0) :
  ((a^2 - 4) / (a^2 - 4*a + 4) - 1 / (2 - a)) / (2 / (a^2 - 2*a)) = 1 :=
by sorry

end expression_simplification_l1898_189880


namespace min_degree_of_specific_polynomial_l1898_189812

/-- A polynomial function from ℝ to ℝ -/
def PolynomialFunction := ℝ → ℝ

/-- The degree of a polynomial function -/
def degree (f : PolynomialFunction) : ℕ := sorry

theorem min_degree_of_specific_polynomial (f : PolynomialFunction)
  (h1 : f (-2) = 3)
  (h2 : f (-1) = -3)
  (h3 : f 1 = -3)
  (h4 : f 2 = 6)
  (h5 : f 3 = 5) :
  degree f = 4 ∧ ∀ g : PolynomialFunction, 
    g (-2) = 3 → g (-1) = -3 → g 1 = -3 → g 2 = 6 → g 3 = 5 → 
    degree g ≥ 4 := by
  sorry

end min_degree_of_specific_polynomial_l1898_189812


namespace total_profit_is_27_l1898_189808

/-- Given the following conditions:
  1. Natasha has 3 times as much money as Carla
  2. Carla has twice as much money as Cosima
  3. Natasha has $60
  4. Sergio has 1.5 times as much money as Cosima
  5. Natasha buys 4 items at $15 each
  6. Carla buys 6 items at $10 each
  7. Cosima buys 5 items at $8 each
  8. Sergio buys 3 items at $12 each
  9. Profit margins: Natasha 10%, Carla 15%, Cosima 12%, Sergio 20%

  Prove that the total profit after selling all goods is $27. -/
theorem total_profit_is_27 (natasha_money carla_money cosima_money sergio_money : ℚ)
  (natasha_items carla_items cosima_items sergio_items : ℕ)
  (natasha_price carla_price cosima_price sergio_price : ℚ)
  (natasha_margin carla_margin cosima_margin sergio_margin : ℚ) :
  natasha_money = 60 ∧
  natasha_money = 3 * carla_money ∧
  carla_money = 2 * cosima_money ∧
  sergio_money = 1.5 * cosima_money ∧
  natasha_items = 4 ∧
  carla_items = 6 ∧
  cosima_items = 5 ∧
  sergio_items = 3 ∧
  natasha_price = 15 ∧
  carla_price = 10 ∧
  cosima_price = 8 ∧
  sergio_price = 12 ∧
  natasha_margin = 0.1 ∧
  carla_margin = 0.15 ∧
  cosima_margin = 0.12 ∧
  sergio_margin = 0.2 →
  natasha_items * natasha_price * natasha_margin +
  carla_items * carla_price * carla_margin +
  cosima_items * cosima_price * cosima_margin +
  sergio_items * sergio_price * sergio_margin = 27 := by
  sorry


end total_profit_is_27_l1898_189808


namespace complex_modulus_problem_l1898_189856

theorem complex_modulus_problem (x y : ℝ) :
  (Complex.I * (x + 2 * Complex.I) = y - Complex.I) →
  Complex.abs (x - y * Complex.I) = Real.sqrt 5 := by
sorry

end complex_modulus_problem_l1898_189856


namespace lower_bound_of_exponential_sum_l1898_189845

theorem lower_bound_of_exponential_sum (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b + c = 1 → 
  ∃ m : ℝ, m = 4 ∧ ∀ x : ℝ, (2^a + 2^b + 2^c < x ↔ m ≤ x) :=
by sorry

end lower_bound_of_exponential_sum_l1898_189845


namespace negative_represents_spending_l1898_189825

/-- Represents a monetary transaction -/
inductive Transaction
| receive (amount : ℤ)
| spend (amount : ℤ)

/-- Converts a transaction to an integer representation -/
def transactionToInt : Transaction → ℤ
| Transaction.receive amount => amount
| Transaction.spend amount => -amount

theorem negative_represents_spending (t : Transaction) : 
  (∃ (a : ℤ), a > 0 ∧ transactionToInt (Transaction.receive a) = a) →
  (∀ (b : ℤ), b > 0 → transactionToInt (Transaction.spend b) = -b) :=
by sorry

end negative_represents_spending_l1898_189825


namespace trajectory_of_moving_circle_center_l1898_189869

-- Define the fixed circle
def fixed_circle (x y : ℝ) : Prop := (x - 5)^2 + (y + 7)^2 = 16

-- Define the moving circle with radius 1
def moving_circle (center_x center_y : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = 1

-- Define the tangency condition
def is_tangent (center_x center_y : ℝ) : Prop :=
  ∃ x y : ℝ, fixed_circle x y ∧ moving_circle center_x center_y x y

-- Theorem statement
theorem trajectory_of_moving_circle_center :
  ∀ center_x center_y : ℝ,
    is_tangent center_x center_y →
    ((center_x - 5)^2 + (center_y + 7)^2 = 25 ∨
     (center_x - 5)^2 + (center_y + 7)^2 = 9) :=
by sorry

end trajectory_of_moving_circle_center_l1898_189869


namespace seven_point_circle_triangles_l1898_189858

/-- The number of triangles formed by intersections of chords in a circle -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose (Nat.choose n 4) 3

/-- Theorem: Given 7 points on a circle with the specified conditions, 
    the number of triangles formed is 6545 -/
theorem seven_point_circle_triangles : num_triangles 7 = 6545 := by
  sorry

end seven_point_circle_triangles_l1898_189858


namespace bug_crawl_distance_l1898_189853

/-- The minimum distance a bug must crawl on the surface of a right circular cone --/
theorem bug_crawl_distance (r h a b θ : ℝ) (hr : r = 500) (hh : h = 300) 
  (ha : a = 100) (hb : b = 400) (hθ : θ = π / 2) : 
  let d := Real.sqrt ((b * Real.cos θ - a)^2 + (b * Real.sin θ)^2)
  d = Real.sqrt 170000 := by
sorry

end bug_crawl_distance_l1898_189853


namespace min_sum_distances_l1898_189876

/-- The minimum sum of distances from a point on the x-axis to two fixed points -/
theorem min_sum_distances (P : ℝ × ℝ) (A B : ℝ × ℝ) : 
  A = (1, 1) → B = (3, 4) → P.2 = 0 → 
  ∀ Q : ℝ × ℝ, Q.2 = 0 → Real.sqrt 29 ≤ dist P A + dist P B :=
sorry

end min_sum_distances_l1898_189876


namespace max_playtime_is_180_minutes_l1898_189874

/-- Represents an arcade bundle with tokens, playtime in hours, and cost --/
structure Bundle where
  tokens : ℕ
  playtime : ℕ
  cost : ℕ

/-- Mike's weekly pay in dollars --/
def weekly_pay : ℕ := 100

/-- Mike's arcade budget in dollars (half of weekly pay) --/
def arcade_budget : ℕ := weekly_pay / 2

/-- Cost of snacks in dollars --/
def snack_cost : ℕ := 5

/-- Available bundles at the arcade --/
def bundles : List Bundle := [
  ⟨50, 1, 25⟩,   -- Bundle A
  ⟨120, 3, 45⟩,  -- Bundle B
  ⟨200, 5, 60⟩   -- Bundle C
]

/-- Remaining budget after buying snacks --/
def remaining_budget : ℕ := arcade_budget - snack_cost

/-- Function to calculate total playtime in minutes for a given bundle and quantity --/
def total_playtime (bundle : Bundle) (quantity : ℕ) : ℕ :=
  bundle.playtime * quantity * 60

/-- Theorem: The maximum playtime Mike can achieve is 180 minutes --/
theorem max_playtime_is_180_minutes :
  ∃ (bundle : Bundle) (quantity : ℕ),
    bundle ∈ bundles ∧
    bundle.cost * quantity ≤ remaining_budget ∧
    total_playtime bundle quantity = 180 ∧
    ∀ (other_bundle : Bundle) (other_quantity : ℕ),
      other_bundle ∈ bundles →
      other_bundle.cost * other_quantity ≤ remaining_budget →
      total_playtime other_bundle other_quantity ≤ 180 :=
sorry

end max_playtime_is_180_minutes_l1898_189874


namespace derivative_implies_antiderivative_l1898_189822

theorem derivative_implies_antiderivative (f : ℝ → ℝ) :
  (∀ x, deriv f x = 6 * x^2 + 5) →
  ∃ c, ∀ x, f x = 2 * x^3 + 5 * x + c :=
sorry

end derivative_implies_antiderivative_l1898_189822


namespace sufficient_not_necessary_l1898_189875

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 6 → a^2 > 36) ∧ (∃ a, a^2 > 36 ∧ a ≤ 6) := by sorry

end sufficient_not_necessary_l1898_189875


namespace intersection_line_slope_l1898_189832

/-- The slope of the line passing through the intersection points of two circles -/
theorem intersection_line_slope (x y : ℝ) : 
  (x^2 + y^2 + 6*x - 8*y - 40 = 0) ∧ 
  (x^2 + y^2 + 22*x - 2*y + 20 = 0) →
  (∃ m : ℝ, m = 8/3 ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      (x₁^2 + y₁^2 + 6*x₁ - 8*y₁ - 40 = 0) ∧ 
      (x₁^2 + y₁^2 + 22*x₁ - 2*y₁ + 20 = 0) ∧
      (x₂^2 + y₂^2 + 6*x₂ - 8*y₂ - 40 = 0) ∧ 
      (x₂^2 + y₂^2 + 22*x₂ - 2*y₂ + 20 = 0) ∧
      (x₁ ≠ x₂) →
      m = (y₂ - y₁) / (x₂ - x₁)) :=
by sorry

end intersection_line_slope_l1898_189832


namespace f_one_geq_25_l1898_189872

/-- A function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The property that f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f m x ≤ f m y

theorem f_one_geq_25 (m : ℝ) (h : is_increasing_on_interval m) : f m 1 ≥ 25 :=
sorry

end f_one_geq_25_l1898_189872


namespace circle_radius_l1898_189879

theorem circle_radius (x y : ℝ) : 
  (x^2 - 10*x + y^2 - 8*y + 29 = 0) → 
  (∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = 2*Real.sqrt 3) :=
by sorry

end circle_radius_l1898_189879


namespace xy_yz_zx_over_x2_y2_z2_l1898_189814

theorem xy_yz_zx_over_x2_y2_z2 (x y z a b c : ℝ) 
  (h_distinct_xyz : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_distinct_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_nonzero_abc : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : a * x + b * y + c * z = 0) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = -1 :=
by sorry

end xy_yz_zx_over_x2_y2_z2_l1898_189814


namespace barbecue_chicken_orders_l1898_189892

/-- Represents the number of pieces of chicken used in different dish types --/
structure ChickenPieces where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- Represents the number of orders for different dish types --/
structure Orders where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- The total number of chicken pieces needed for all orders --/
def totalChickenPieces (cp : ChickenPieces) (o : Orders) : ℕ :=
  cp.pasta * o.pasta + cp.barbecue * o.barbecue + cp.friedDinner * o.friedDinner

/-- The theorem to prove --/
theorem barbecue_chicken_orders
  (cp : ChickenPieces)
  (o : Orders)
  (h1 : cp.pasta = 2)
  (h2 : cp.barbecue = 3)
  (h3 : cp.friedDinner = 8)
  (h4 : o.friedDinner = 2)
  (h5 : o.pasta = 6)
  (h6 : totalChickenPieces cp o = 37) :
  o.barbecue = 3 := by
  sorry

end barbecue_chicken_orders_l1898_189892


namespace min_typical_parallelepipeds_is_four_l1898_189802

/-- A typical parallelepiped has all dimensions different -/
structure TypicalParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  all_different : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- A cube with side length s -/
structure Cube where
  side : ℝ

/-- The minimum number of typical parallelepipeds into which a cube can be cut -/
def min_typical_parallelepipeds_in_cube (c : Cube) : ℕ :=
  4

/-- Theorem stating that the minimum number of typical parallelepipeds 
    into which a cube can be cut is 4 -/
theorem min_typical_parallelepipeds_is_four (c : Cube) :
  min_typical_parallelepipeds_in_cube c = 4 := by
  sorry

end min_typical_parallelepipeds_is_four_l1898_189802


namespace janes_calculation_l1898_189813

theorem janes_calculation (a b c : ℝ) 
  (h1 : a + b + c = 11) 
  (h2 : a + b - c = 19) : 
  a + b = 15 := by
sorry

end janes_calculation_l1898_189813


namespace special_polygon_properties_l1898_189854

/-- A polygon where each interior angle is 4 times the exterior angle at the same vertex -/
structure SpecialPolygon where
  vertices : ℕ
  interior_angle : Fin vertices → ℝ
  exterior_angle : Fin vertices → ℝ
  angle_relation : ∀ i, interior_angle i = 4 * exterior_angle i
  sum_exterior_angles : (Finset.univ.sum exterior_angle) = 360

theorem special_polygon_properties (Q : SpecialPolygon) :
  (Finset.univ.sum Q.interior_angle = 1440) ∧
  (∀ i j, Q.interior_angle i = Q.interior_angle j) := by
  sorry

#check special_polygon_properties

end special_polygon_properties_l1898_189854


namespace weaver_output_increase_l1898_189849

theorem weaver_output_increase (first_day_output : ℝ) (total_days : ℕ) (total_output : ℝ) :
  first_day_output = 5 ∧ total_days = 30 ∧ total_output = 390 →
  ∃ (daily_increase : ℝ),
    daily_increase = 16/29 ∧
    total_output = total_days * first_day_output + (total_days * (total_days - 1) / 2) * daily_increase :=
by sorry

end weaver_output_increase_l1898_189849


namespace polynomial_simplification_l1898_189883

theorem polynomial_simplification (x : ℝ) :
  (15 * x^10 + 10 * x^9 + 5 * x^8) + (3 * x^12 + 2 * x^10 + x^9 + 3 * x^7 + 4 * x^4 + 6 * x^2 + 9) =
  3 * x^12 + 17 * x^10 + 11 * x^9 + 5 * x^8 + 3 * x^7 + 4 * x^4 + 6 * x^2 + 9 :=
by
  sorry

end polynomial_simplification_l1898_189883


namespace multiples_of_seven_between_50_and_150_l1898_189851

theorem multiples_of_seven_between_50_and_150 :
  (Finset.filter (fun n => 50 ≤ 7 * n ∧ 7 * n ≤ 150) (Finset.range (150 / 7 + 1))).card = 14 := by
  sorry

end multiples_of_seven_between_50_and_150_l1898_189851


namespace max_consecutive_integers_sum_45_l1898_189835

/-- Given a natural number n, returns the sum of n consecutive positive integers starting from a -/
def consecutive_sum (n : ℕ) (a : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Predicate that checks if there exists a starting integer a such that n consecutive integers starting from a sum to 45 -/
def exists_consecutive_sum (n : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ consecutive_sum n a = 45

theorem max_consecutive_integers_sum_45 :
  (∀ k : ℕ, k > 9 → ¬ exists_consecutive_sum k) ∧
  exists_consecutive_sum 9 :=
sorry

end max_consecutive_integers_sum_45_l1898_189835


namespace complex_modulus_l1898_189848

theorem complex_modulus (a b : ℝ) (h : (1 + 2*a*Complex.I) * Complex.I = 1 - b*Complex.I) : 
  Complex.abs (a + b*Complex.I) = Real.sqrt 5 / 2 := by
  sorry

end complex_modulus_l1898_189848


namespace green_beads_count_l1898_189809

/-- The number of green beads initially in a container -/
def initial_green_beads (total : ℕ) (brown red taken left : ℕ) : ℕ :=
  total - brown - red

/-- Theorem stating the number of green beads initially in the container -/
theorem green_beads_count (brown red taken left : ℕ) 
  (h1 : brown = 2)
  (h2 : red = 3)
  (h3 : taken = 2)
  (h4 : left = 4) :
  initial_green_beads (taken + left) brown red taken left = 1 := by
  sorry

#check green_beads_count

end green_beads_count_l1898_189809


namespace log_56342_between_consecutive_integers_l1898_189828

theorem log_56342_between_consecutive_integers :
  ∃ (c d : ℕ), c + 1 = d ∧ (c : ℝ) < Real.log 56342 / Real.log 10 ∧ Real.log 56342 / Real.log 10 < d ∧ c + d = 9 :=
by
  -- Assuming 10000 < 56342 < 100000
  have h1 : 10000 < 56342 := by sorry
  have h2 : 56342 < 100000 := by sorry
  sorry

end log_56342_between_consecutive_integers_l1898_189828


namespace find_d_l1898_189850

theorem find_d (A B C D : ℝ) : 
  (A + B + C) / 3 = 130 →
  (A + B + C + D) / 4 = 126 →
  D = 114 := by
sorry

end find_d_l1898_189850


namespace nancy_small_gardens_l1898_189818

/-- Given the total number of seeds, seeds planted in the big garden, and seeds per small garden,
    calculate the number of small gardens Nancy had. -/
def number_of_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Prove that Nancy had 6 small gardens given the conditions. -/
theorem nancy_small_gardens :
  number_of_small_gardens 52 28 4 = 6 := by
  sorry

end nancy_small_gardens_l1898_189818


namespace area_of_LMNOPQ_l1898_189896

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.a * r.b

/-- Represents the polygon LMNOPQ formed by two overlapping rectangles -/
structure PolygonLMNOPQ where
  lmno : Rectangle
  opqr : Rectangle
  lm : ℝ
  mn : ℝ
  no : ℝ
  -- Conditions
  h1 : lmno.a = lm
  h2 : lmno.b = mn
  h3 : opqr.a = mn
  h4 : opqr.b = lm
  h5 : lm = 8
  h6 : mn = 10
  h7 : no = 3

theorem area_of_LMNOPQ (p : PolygonLMNOPQ) : p.lmno.area = 80 := by
  sorry

#check area_of_LMNOPQ

end area_of_LMNOPQ_l1898_189896


namespace intersection_point_l1898_189843

/-- The quadratic function f(x) = x^2 - 5x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 5*x + 1

/-- The y-axis is the set of points with x-coordinate 0 -/
def yAxis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

theorem intersection_point :
  (0, 1) ∈ yAxis ∧ f 0 = 1 := by sorry

end intersection_point_l1898_189843


namespace strawberry_picking_total_weight_l1898_189831

theorem strawberry_picking_total_weight 
  (marco_weight : ℕ) 
  (dad_weight : ℕ) 
  (h1 : marco_weight = 8) 
  (h2 : dad_weight = 32) : 
  marco_weight + dad_weight = 40 := by
sorry

end strawberry_picking_total_weight_l1898_189831


namespace quadratic_no_real_roots_l1898_189823

theorem quadratic_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a ≠ 0) → a < -1 := by sorry

end quadratic_no_real_roots_l1898_189823


namespace room_diagonal_l1898_189894

theorem room_diagonal (l h d : ℝ) (b : ℝ) : 
  l = 12 → h = 9 → d = 17 → d^2 = l^2 + b^2 + h^2 → b = 8 := by sorry

end room_diagonal_l1898_189894


namespace neg_p_sufficient_not_necessary_for_neg_q_l1898_189864

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

-- Define the relationship between ¬p and ¬q
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x : ℝ, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x : ℝ, ¬(q x) → ¬(p x)) := by
  sorry

end neg_p_sufficient_not_necessary_for_neg_q_l1898_189864


namespace coefficient_of_y_l1898_189827

theorem coefficient_of_y (x y a : ℝ) : 
  5 * x + y = 19 →
  x + a * y = 1 →
  3 * x + 2 * y = 10 →
  a = 3 := by
sorry

end coefficient_of_y_l1898_189827


namespace snow_volume_on_blocked_sidewalk_l1898_189839

/-- Calculates the volume of snow to shovel from a partially blocked rectangular sidewalk. -/
theorem snow_volume_on_blocked_sidewalk
  (total_length : ℝ)
  (width : ℝ)
  (blocked_length : ℝ)
  (snow_depth : ℝ)
  (h1 : total_length = 30)
  (h2 : width = 3)
  (h3 : blocked_length = 5)
  (h4 : snow_depth = 2/3)
  : (total_length - blocked_length) * width * snow_depth = 50 := by
  sorry

#check snow_volume_on_blocked_sidewalk

end snow_volume_on_blocked_sidewalk_l1898_189839
