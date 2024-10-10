import Mathlib

namespace section_B_seats_l347_34761

-- Define the number of seats in the different subsections of Section A
def seats_subsection_1 : ℕ := 60
def seats_subsection_2 : ℕ := 80
def num_subsection_2 : ℕ := 3

-- Define the total number of seats in Section A
def total_seats_A : ℕ := seats_subsection_1 + seats_subsection_2 * num_subsection_2

-- Define the number of seats in Section B
def seats_B : ℕ := 3 * total_seats_A + 20

-- Theorem statement
theorem section_B_seats : seats_B = 920 := by sorry

end section_B_seats_l347_34761


namespace least_value_x_minus_y_minus_z_l347_34798

theorem least_value_x_minus_y_minus_z :
  ∀ (x y z : ℕ+), x = 4 → y = 7 → (x : ℤ) - y - z ≥ -4 ∧ ∃ (z : ℕ+), (x : ℤ) - y - z = -4 :=
by sorry

end least_value_x_minus_y_minus_z_l347_34798


namespace least_number_with_remainder_l347_34739

theorem least_number_with_remainder (n : ℕ) : n = 115 ↔ 
  (n > 0 ∧ 
   n % 38 = 1 ∧ 
   n % 3 = 1 ∧ 
   ∀ m : ℕ, m > 0 → m % 38 = 1 → m % 3 = 1 → n ≤ m) :=
by sorry

end least_number_with_remainder_l347_34739


namespace cos_150_degrees_l347_34744

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l347_34744


namespace bridge_length_calculation_l347_34755

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  train_length = 320 →
  train_speed_kmh = 45 →
  time_to_pass = 36.8 →
  ∃ (bridge_length : ℝ), bridge_length = 140 ∧
    bridge_length = (train_speed_kmh * 1000 / 3600 * time_to_pass) - train_length :=
by sorry

end bridge_length_calculation_l347_34755


namespace sum_of_roots_is_twelve_l347_34799

/-- A function g: ℝ → ℝ that satisfies g(3+x) = g(3-x) for all real x -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The proposition that g has exactly four distinct real roots -/
def HasFourDistinctRoots (g : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (∀ x : ℝ, g x = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d))

/-- The theorem stating that the sum of roots is 12 -/
theorem sum_of_roots_is_twelve (g : ℝ → ℝ) 
    (h1 : SymmetricAboutThree g) (h2 : HasFourDistinctRoots g) : 
    ∃ (a b c d : ℝ), (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧ (a + b + c + d = 12) :=
  sorry

end sum_of_roots_is_twelve_l347_34799


namespace john_shorter_than_rebeca_l347_34710

def height_difference (john_height lena_height rebeca_height : ℕ) : Prop :=
  (john_height = lena_height + 15) ∧
  (john_height < rebeca_height) ∧
  (john_height = 152) ∧
  (lena_height + rebeca_height = 295)

theorem john_shorter_than_rebeca (john_height lena_height rebeca_height : ℕ) :
  height_difference john_height lena_height rebeca_height →
  rebeca_height - john_height = 6 :=
by
  sorry

end john_shorter_than_rebeca_l347_34710


namespace root_triple_relation_l347_34735

theorem root_triple_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c := by
  sorry

end root_triple_relation_l347_34735


namespace eleven_in_base_two_l347_34774

theorem eleven_in_base_two : 11 = 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 := by
  sorry

#eval toString (1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0)

end eleven_in_base_two_l347_34774


namespace cube_sum_inequality_l347_34718

theorem cube_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + b = c + d) (square_sum_gt : a^2 + b^2 > c^2 + d^2) :
  a^3 + b^3 > c^3 + d^3 := by
  sorry

end cube_sum_inequality_l347_34718


namespace triangle_properties_l347_34730

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.c * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.C ∧
  t.a^2 - t.c^2 = 2 * t.b^2 ∧
  (1/2) * t.a * t.b * Real.sin t.C = 21 * Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.C = π/3 ∧ t.b = 2 * Real.sqrt 7 := by
  sorry

end triangle_properties_l347_34730


namespace instantaneous_velocity_at_4_seconds_l347_34737

/-- The displacement function of an object with respect to time -/
def displacement (t : ℝ) : ℝ := 4 - 2*t + t^2

/-- The velocity function of an object with respect to time -/
def velocity (t : ℝ) : ℝ := 2*t - 2

theorem instantaneous_velocity_at_4_seconds :
  velocity 4 = 6 := by sorry

end instantaneous_velocity_at_4_seconds_l347_34737


namespace max_weighings_for_15_coins_l347_34751

/-- Represents a coin which can be either genuine or counterfeit -/
inductive Coin
| genuine : Coin
| counterfeit : Coin

/-- Represents the result of a weighing -/
inductive WeighingResult
| left_heavier : WeighingResult
| right_heavier : WeighingResult
| equal : WeighingResult

/-- A function that simulates weighing two groups of coins -/
def weigh (left : List Coin) (right : List Coin) : WeighingResult := sorry

/-- A function that finds the counterfeit coin -/
def find_counterfeit (coins : List Coin) : Nat → Option Coin := sorry

theorem max_weighings_for_15_coins :
  ∀ (coins : List Coin),
    coins.length = 15 →
    (∃! c, c ∈ coins ∧ c = Coin.counterfeit) →
    ∃ n, n ≤ 3 ∧ (find_counterfeit coins n).isSome ∧
        ∀ m, m < n → (find_counterfeit coins m).isNone := by sorry

#check max_weighings_for_15_coins

end max_weighings_for_15_coins_l347_34751


namespace invitation_ways_l347_34757

-- Define the total number of classmates
def total_classmates : ℕ := 10

-- Define the number of classmates to invite
def invited_classmates : ℕ := 6

-- Define the number of classmates excluding A and B
def remaining_classmates : ℕ := total_classmates - 2

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem invitation_ways : 
  combination remaining_classmates (invited_classmates - 2) + 
  combination remaining_classmates invited_classmates = 98 := by
sorry

end invitation_ways_l347_34757


namespace ambiguous_date_characterization_max_consecutive_ambiguous_proof_l347_34720

/-- Represents a date with day and month -/
structure Date where
  day : Nat
  month : Nat
  h1 : day ≥ 1 ∧ day ≤ 31
  h2 : month ≥ 1 ∧ month ≤ 12

/-- Defines when a date is ambiguous -/
def is_ambiguous (d : Date) : Prop :=
  d.day ≥ 1 ∧ d.day ≤ 12 ∧ d.day ≠ d.month

/-- The maximum number of consecutive ambiguous dates in any month -/
def max_consecutive_ambiguous : Nat := 11

theorem ambiguous_date_characterization (d : Date) :
  is_ambiguous d ↔ d.day ≥ 1 ∧ d.day ≤ 12 ∧ d.day ≠ d.month :=
sorry

theorem max_consecutive_ambiguous_proof :
  ∀ m : Nat, m ≥ 1 → m ≤ 12 →
    (∃ consecutive : List Date,
      consecutive.length = max_consecutive_ambiguous ∧
      (∀ d ∈ consecutive, d.month = m ∧ is_ambiguous d) ∧
      (∀ d : Date, d.month = m → is_ambiguous d → d ∈ consecutive)) :=
sorry

end ambiguous_date_characterization_max_consecutive_ambiguous_proof_l347_34720


namespace product_of_x_values_l347_34787

theorem product_of_x_values (x₁ x₂ : ℝ) : 
  (|20 / x₁ + 4| = 3 ∧ |20 / x₂ + 4| = 3 ∧ x₁ ≠ x₂) → x₁ * x₂ = 400 / 7 :=
by sorry

end product_of_x_values_l347_34787


namespace positive_real_array_inequalities_l347_34748

theorem positive_real_array_inequalities
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h1 : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h2 : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h3 : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h4 : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h5 : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ := by
sorry

end positive_real_array_inequalities_l347_34748


namespace line_intersects_circle_twice_tangent_line_m_value_l347_34758

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 25

-- Define the line L
def line_L (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Define the circle D
def circle_D (R x y : ℝ) : Prop := (x + 1)^2 + (y - 5)^2 = R^2

theorem line_intersects_circle_twice (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ :=
sorry

theorem tangent_line_m_value :
  ∃ (R : ℝ), R > 0 ∧
    (∀ (R' : ℝ), R' > 0 →
      (∃ (x y : ℝ), circle_D R' x y ∧ line_L (-2/3) x y) →
      R' ≤ R) ∧
    (∃ (x y : ℝ), circle_D R x y ∧ line_L (-2/3) x y) :=
sorry

end line_intersects_circle_twice_tangent_line_m_value_l347_34758


namespace average_of_multiples_of_seven_l347_34721

theorem average_of_multiples_of_seven (n : ℕ) : 
  (n / 2 : ℚ) * (7 + 7 * n) / n = 77 → n = 21 := by
  sorry

end average_of_multiples_of_seven_l347_34721


namespace birch_count_l347_34776

/-- Represents the number of trees of each species in the forest -/
structure ForestComposition where
  oak : ℕ
  pine : ℕ
  spruce : ℕ
  birch : ℕ

/-- The total number of trees in the forest -/
def total_trees : ℕ := 4000

/-- The forest composition satisfies the given conditions -/
def is_valid_composition (fc : ForestComposition) : Prop :=
  fc.oak + fc.pine + fc.spruce + fc.birch = total_trees ∧
  fc.spruce = total_trees / 10 ∧
  fc.pine = total_trees * 13 / 100 ∧
  fc.oak = fc.spruce + fc.pine

theorem birch_count (fc : ForestComposition) (h : is_valid_composition fc) : fc.birch = 2160 := by
  sorry

#check birch_count

end birch_count_l347_34776


namespace polygon_diagonals_l347_34749

theorem polygon_diagonals (n : ℕ) (d : ℕ) : n = 17 ∧ d = 104 →
  (n - 1) * (n - 4) / 2 = d := by
  sorry

end polygon_diagonals_l347_34749


namespace vertices_count_l347_34765

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for convex polyhedra -/
axiom eulers_formula (p : ConvexPolyhedron) : p.vertices - p.edges + p.faces = 2

/-- A face of a polyhedron -/
inductive Face
| Triangle : Face

/-- Our specific polyhedron -/
def our_polyhedron : ConvexPolyhedron where
  vertices := 12  -- This is what we want to prove
  edges := 30
  faces := 20

/-- All faces of our polyhedron are triangles -/
axiom all_faces_triangular : ∀ f : Face, f = Face.Triangle

/-- The number of vertices in our polyhedron is correct -/
theorem vertices_count : our_polyhedron.vertices = 12 := by sorry

end vertices_count_l347_34765


namespace probability_even_sum_l347_34777

def set_A : Finset ℕ := {3, 4, 5, 8}
def set_B : Finset ℕ := {6, 7, 9}

def is_sum_even (a b : ℕ) : Bool :=
  (a + b) % 2 = 0

def count_even_sums : ℕ :=
  (set_A.card * set_B.card).div 2

theorem probability_even_sum :
  (count_even_sums : ℚ) / (set_A.card * set_B.card) = 1 / 2 := by
  sorry

end probability_even_sum_l347_34777


namespace unique_solution_mn_l347_34768

theorem unique_solution_mn : 
  ∃! (m n : ℕ+), 18 * (m : ℝ) * (n : ℝ) = 73 - 9 * (m : ℝ) - 3 * (n : ℝ) ∧ m = 4 ∧ n = 18 := by
  sorry

end unique_solution_mn_l347_34768


namespace missing_number_proof_l347_34713

/-- Given a list of 10 numbers with an average of 750, where 9 of the numbers are known,
    prove that the remaining number is 1747. -/
theorem missing_number_proof (numbers : List ℕ) (h1 : numbers.length = 10)
  (h2 : numbers.sum / numbers.length = 750)
  (h3 : numbers.count 744 = 1)
  (h4 : numbers.count 745 = 1)
  (h5 : numbers.count 748 = 1)
  (h6 : numbers.count 749 = 1)
  (h7 : numbers.count 752 = 2)
  (h8 : numbers.count 753 = 1)
  (h9 : numbers.count 755 = 2)
  : numbers.any (· = 1747) := by
  sorry

end missing_number_proof_l347_34713


namespace ant_path_count_l347_34762

/-- The number of paths from A to B -/
def paths_AB : ℕ := 3

/-- The number of paths from B to C -/
def paths_BC : ℕ := 3

/-- The total number of paths from A to C through B -/
def total_paths : ℕ := paths_AB * paths_BC

/-- Theorem stating that the total number of paths from A to C through B is 9 -/
theorem ant_path_count : total_paths = 9 := by
  sorry

end ant_path_count_l347_34762


namespace fraction_less_than_two_l347_34783

theorem fraction_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end fraction_less_than_two_l347_34783


namespace john_tax_increase_l347_34759

/-- Calculates the increase in taxes paid given old and new tax rates and incomes -/
def tax_increase (old_rate new_rate : ℚ) (old_income new_income : ℕ) : ℚ :=
  new_rate * new_income - old_rate * old_income

/-- Proves that John's tax increase is $250,000 -/
theorem john_tax_increase :
  let old_rate : ℚ := 1/5
  let new_rate : ℚ := 3/10
  let old_income : ℕ := 1000000
  let new_income : ℕ := 1500000
  tax_increase old_rate new_rate old_income new_income = 250000 := by
  sorry

#eval tax_increase (1/5) (3/10) 1000000 1500000

end john_tax_increase_l347_34759


namespace third_side_length_l347_34790

/-- A right-angled isosceles triangle with specific dimensions -/
structure RightIsoscelesTriangle where
  /-- The length of the equal sides -/
  a : ℝ
  /-- The length of the hypotenuse -/
  c : ℝ
  /-- The triangle is right-angled -/
  right_angled : a^2 + a^2 = c^2
  /-- The triangle is isosceles -/
  isosceles : a = 50
  /-- The perimeter of the triangle -/
  perimeter : a + a + c = 160

/-- The theorem stating the length of the third side -/
theorem third_side_length (t : RightIsoscelesTriangle) : t.c = 60 := by
  sorry

end third_side_length_l347_34790


namespace subset_implies_a_equals_one_l347_34773

theorem subset_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {0, -a}
  let B : Set ℝ := {1, a-2, 2*a-2}
  A ⊆ B → a = 1 := by
sorry

end subset_implies_a_equals_one_l347_34773


namespace jimmy_stair_climb_time_l347_34795

def stairClimbTime (n : ℕ) : ℕ :=
  let baseTime := 25
  let increment := 7
  let flightTimes := List.range n |>.map (λ i => baseTime + i * increment)
  let totalFlightTime := flightTimes.sum
  let stopTime := (n - 1) / 2 * 10
  totalFlightTime + stopTime

theorem jimmy_stair_climb_time :
  stairClimbTime 7 = 342 := by
  sorry

end jimmy_stair_climb_time_l347_34795


namespace pirate_treasure_sum_l347_34791

def base7_to_base10 (n : ℕ) : ℕ := sorry

def diamonds : ℕ := 6352
def ancient_coins : ℕ := 3206
def silver : ℕ := 156

theorem pirate_treasure_sum :
  base7_to_base10 diamonds + base7_to_base10 ancient_coins + base7_to_base10 silver = 3465 := by
  sorry

end pirate_treasure_sum_l347_34791


namespace function_form_exists_l347_34700

noncomputable def f (a b c x : ℝ) : ℝ := a * b^x + c

theorem function_form_exists :
  ∃ (a b c : ℝ),
    (∀ x : ℝ, x ≥ 0 → -2 ≤ f a b c x ∧ f a b c x < 3) ∧
    (0 < b ∧ b < 1) ∧
    (∀ x : ℝ, x ≥ 0 → f a b c x = -5 * b^x + 3) :=
by sorry

end function_form_exists_l347_34700


namespace base7_to_base10_conversion_l347_34771

-- Define the base-7 number as a list of digits
def base7Number : List Nat := [4, 5, 3, 6]

-- Define the function to convert from base 7 to base 10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

-- Theorem statement
theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 1644 := by
  sorry

end base7_to_base10_conversion_l347_34771


namespace three_digit_sum_relation_l347_34731

/-- Given a three-digit number with tens digit zero, prove the relationship between m and n -/
theorem three_digit_sum_relation (x y m n : ℕ) : 
  (100 * y + x = m * (x + y)) →   -- Original number is m times sum of digits
  (100 * x + y = n * (x + y)) →   -- Swapped number is n times sum of digits
  n = 101 - m := by
  sorry

end three_digit_sum_relation_l347_34731


namespace binomial_coefficient_20_11_l347_34709

theorem binomial_coefficient_20_11 :
  (Nat.choose 18 9 = 48620) →
  (Nat.choose 18 8 = 43758) →
  (Nat.choose 20 11 = 168168) := by
  sorry

end binomial_coefficient_20_11_l347_34709


namespace students_liking_both_desserts_l347_34753

/-- Given a class of students, calculate the number who like both apple pie and chocolate cake. -/
theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (like_apple_pie : ℕ) 
  (like_chocolate_cake : ℕ) 
  (like_neither : ℕ) 
  (h1 : total_students = 50)
  (h2 : like_apple_pie = 22)
  (h3 : like_chocolate_cake = 20)
  (h4 : like_neither = 15) :
  like_apple_pie + like_chocolate_cake - (total_students - like_neither) = 7 := by
  sorry

#check students_liking_both_desserts

end students_liking_both_desserts_l347_34753


namespace last_colored_cell_position_l347_34754

/-- Represents a cell position in the grid -/
structure CellPosition where
  row : Nat
  col : Nat

/-- Represents the dimensions of the rectangle -/
structure RectangleDimensions where
  width : Nat
  height : Nat

/-- Represents the coloring process in a spiral pattern -/
def spiralColor (dim : RectangleDimensions) : CellPosition :=
  sorry

/-- Theorem: The last cell colored in a 200x100 rectangle with spiral coloring is at (51, 50) -/
theorem last_colored_cell_position :
  let dim : RectangleDimensions := ⟨200, 100⟩
  spiralColor dim = ⟨51, 50⟩ := by
  sorry

end last_colored_cell_position_l347_34754


namespace cubic_function_and_tangent_lines_l347_34782

/-- Given a cubic function f(x) = ax³ + b with a tangent line y = 3x - 1 at x = 1,
    prove that f(x) = x³ + 1 and find the equations of tangent lines passing through (-1, 0) --/
theorem cubic_function_and_tangent_lines 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b)
  (h2 : ∃ c d, ∀ x, (3 : ℝ) * x - 1 = c * (x - 1) + d ∧ f 1 = d ∧ (deriv f) 1 = c) :
  (∀ x, f x = x^3 + 1) ∧ 
  (∃ m₁ m₂ : ℝ, 
    (m₁ = 3 ∧ f (-1) = 0 ∧ (deriv f) (-1) = m₁) ∨ 
    (m₂ = 3/4 ∧ f (-1) = 0 ∧ (deriv f) (-1) = m₂)) :=
by sorry

end cubic_function_and_tangent_lines_l347_34782


namespace cistern_wet_surface_area_l347_34793

/-- Calculates the total wet surface area of a rectangular cistern -/
def totalWetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem: The total wet surface area of a cistern with given dimensions -/
theorem cistern_wet_surface_area :
  let length : ℝ := 9
  let width : ℝ := 4
  let depth : ℝ := 1.25
  totalWetSurfaceArea length width depth = 68.5 := by
  sorry

end cistern_wet_surface_area_l347_34793


namespace total_donation_is_1570_l347_34711

/-- Represents the donation amounts to different parks -/
structure Donations where
  treetown_and_forest : ℝ
  forest_reserve : ℝ
  animal_preservation : ℝ

/-- Calculates the total donation to all three parks -/
def total_donation (d : Donations) : ℝ :=
  d.treetown_and_forest + d.animal_preservation

/-- Theorem stating the total donation to all three parks -/
theorem total_donation_is_1570 (d : Donations) 
  (h1 : d.treetown_and_forest = 570)
  (h2 : d.forest_reserve = d.animal_preservation + 140)
  (h3 : d.treetown_and_forest = d.forest_reserve + d.animal_preservation) : 
  total_donation d = 1570 := by
  sorry

#check total_donation_is_1570

end total_donation_is_1570_l347_34711


namespace train_journey_solution_l347_34775

/-- Represents the train journey problem -/
structure TrainJourney where
  distance : ℝ  -- Distance between stations in km
  speed : ℝ     -- Initial speed of the train in km/h

/-- Conditions of the train journey -/
def journey_conditions (j : TrainJourney) : Prop :=
  let reduced_speed := j.speed / 3
  let first_day_time := 2 + 0.5 + (j.distance - 2 * j.speed) / reduced_speed
  let second_day_time := (2 * j.speed + 14) / j.speed + 0.5 + (j.distance - (2 * j.speed + 14)) / reduced_speed
  first_day_time = j.distance / j.speed + 7/6 ∧
  second_day_time = j.distance / j.speed + 5/6

/-- The theorem to prove -/
theorem train_journey_solution :
  ∃ j : TrainJourney, journey_conditions j ∧ j.distance = 196 ∧ j.speed = 84 :=
sorry

end train_journey_solution_l347_34775


namespace cross_in_square_l347_34716

/-- Given a square with side length S containing a cross made of two large squares
    (each with side length S/2) and two small squares (each with side length S/4),
    if the total area of the cross is 810 cm², then S = 36 cm. -/
theorem cross_in_square (S : ℝ) : 
  (2 * (S/2)^2 + 2 * (S/4)^2 = 810) → S = 36 := by
  sorry

end cross_in_square_l347_34716


namespace no_hamiltonian_cycle_in_circ_2016_2_3_l347_34742

/-- A circulant digraph with n vertices and jump sizes a and b -/
structure CirculantDigraph (n : ℕ) (a b : ℕ) where
  vertices : Fin n

/-- Condition for the existence of a Hamiltonian cycle in a circulant digraph -/
def has_hamiltonian_cycle (G : CirculantDigraph n a b) : Prop :=
  ∃ (s t : ℕ), s + t = Nat.gcd n (a - b) ∧ Nat.gcd n (s * a + t * b) = 1

/-- The main theorem about the non-existence of a Hamiltonian cycle in Circ(2016; 2, 3) -/
theorem no_hamiltonian_cycle_in_circ_2016_2_3 :
  ¬ ∃ (G : CirculantDigraph 2016 2 3), has_hamiltonian_cycle G :=
by sorry

end no_hamiltonian_cycle_in_circ_2016_2_3_l347_34742


namespace sinusoidal_function_properties_l347_34780

theorem sinusoidal_function_properties (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let f := fun x => a * Real.sin (b * x + c)
  (∀ x, f x ≤ 3) ∧ (f (π / 3) = 3) → a = 3 ∧ c = π / 6 := by
  sorry

end sinusoidal_function_properties_l347_34780


namespace min_distinct_values_l347_34732

/-- A list of positive integers -/
def IntegerList := List ℕ+

/-- The number of occurrences of the most frequent element in a list -/
def modeCount (l : IntegerList) : ℕ := sorry

/-- The number of distinct elements in a list -/
def distinctCount (l : IntegerList) : ℕ := sorry

/-- Theorem: Minimum number of distinct values in a list of 4000 positive integers
    with a unique mode occurring exactly 20 times is 211 -/
theorem min_distinct_values (l : IntegerList) 
  (h1 : l.length = 4000)
  (h2 : ∃! x, modeCount l = x)
  (h3 : modeCount l = 20) :
  distinctCount l ≥ 211 := by sorry

end min_distinct_values_l347_34732


namespace no_real_solutions_l347_34785

theorem no_real_solutions (x : ℝ) :
  x ≠ -1 → (x^2 + x + 1) / (x + 1) ≠ x^2 + 5*x + 6 :=
by sorry

end no_real_solutions_l347_34785


namespace cone_height_relationship_l347_34704

/-- Represents the properties of a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Given two cones with equal volume and the second cone's radius 10% larger than the first,
    prove that the height of the first cone is 21% larger than the second -/
theorem cone_height_relationship (cone1 cone2 : Cone) 
  (h_volume : (1/3) * π * cone1.radius^2 * cone1.height = (1/3) * π * cone2.radius^2 * cone2.height)
  (h_radius : cone2.radius = 1.1 * cone1.radius) : 
  cone1.height = 1.21 * cone2.height := by
  sorry

end cone_height_relationship_l347_34704


namespace total_snakes_count_l347_34784

/-- Represents the total population in the neighborhood -/
def total_population : ℕ := 200

/-- Represents the percentage of people who own only snakes -/
def only_snakes_percent : ℚ := 5 / 100

/-- Represents the percentage of people who own both cats and snakes, but no other pets -/
def cats_and_snakes_percent : ℚ := 4 / 100

/-- Represents the percentage of people who own both snakes and rabbits, but no other pets -/
def snakes_and_rabbits_percent : ℚ := 5 / 100

/-- Represents the percentage of people who own both snakes and birds, but no other pets -/
def snakes_and_birds_percent : ℚ := 3 / 100

/-- Represents the percentage of exotic pet owners who also own snakes -/
def exotic_and_snakes_percent : ℚ := 25 / 100

/-- Represents the total percentage of exotic pet owners -/
def total_exotic_percent : ℚ := 34 / 100

/-- Calculates the total percentage of snake owners in the neighborhood -/
def total_snake_owners_percent : ℚ :=
  only_snakes_percent + cats_and_snakes_percent + snakes_and_rabbits_percent + 
  snakes_and_birds_percent + (exotic_and_snakes_percent * total_exotic_percent)

/-- Theorem stating that the total number of snakes in the neighborhood is 51 -/
theorem total_snakes_count : ⌊(total_snake_owners_percent * total_population : ℚ)⌋ = 51 := by
  sorry

end total_snakes_count_l347_34784


namespace wrench_force_calculation_l347_34789

/-- The force required to loosen a nut with a wrench -/
def force_to_loosen (handle_length : ℝ) (force : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * handle_length = k

theorem wrench_force_calculation 
  (h₁ : force_to_loosen 12 480) 
  (h₂ : force_to_loosen 18 f) : 
  f = 320 := by
  sorry

end wrench_force_calculation_l347_34789


namespace gas_cost_problem_l347_34778

theorem gas_cost_problem (x : ℝ) : 
  (x / 4 - x / 7 = 15) → x = 140 := by
  sorry

end gas_cost_problem_l347_34778


namespace arithmetic_calculation_l347_34786

theorem arithmetic_calculation : 4 * 10 + 5 * 11 + 12 * 4 + 4 * 9 = 179 := by
  sorry

end arithmetic_calculation_l347_34786


namespace half_of_eighteen_is_nine_l347_34756

theorem half_of_eighteen_is_nine : (18 : ℝ) / 2 = 9 := by sorry

end half_of_eighteen_is_nine_l347_34756


namespace remaining_average_l347_34746

theorem remaining_average (n : ℕ) (total_avg : ℚ) (partial_avg : ℚ) :
  n = 10 →
  total_avg = 80 →
  partial_avg = 58 →
  ∃ (m : ℕ), m = 6 ∧
    (n * total_avg - m * partial_avg) / (n - m) = 113 :=
by sorry

end remaining_average_l347_34746


namespace constant_value_l347_34722

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- Define the equation
def equation (c : ℝ) (x : ℝ) : Prop :=
  (3 * f (x - 2)) / f 0 + 4 = f (c * x + 1)

-- Theorem statement
theorem constant_value :
  ∀ c : ℝ, equation c 0.4 → c = 2 := by
  sorry

end constant_value_l347_34722


namespace ratio_problem_l347_34703

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.1875) :
  e / f = 0.125 := by
sorry

end ratio_problem_l347_34703


namespace horner_method_evaluation_l347_34725

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

theorem horner_method_evaluation : f 5 = 4881 := by
  sorry

end horner_method_evaluation_l347_34725


namespace finite_triples_sum_reciprocals_l347_34796

theorem finite_triples_sum_reciprocals :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 →
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = (1 : ℚ) / 1000 →
    (a, b, c) ∈ S :=
by sorry

end finite_triples_sum_reciprocals_l347_34796


namespace second_row_sum_is_528_l347_34763

/-- Represents a square grid -/
structure Grid (n : ℕ) :=
  (elements : Matrix (Fin n) (Fin n) ℕ)

/-- Fills the grid with numbers from 1 to n^2 in a clockwise spiral starting from the center -/
def fillGrid (n : ℕ) : Grid n :=
  sorry

/-- Returns the second row from the top of the grid -/
def secondRow (g : Grid 17) : Fin 17 → ℕ :=
  sorry

/-- The greatest number in the second row -/
def maxSecondRow (g : Grid 17) : ℕ :=
  sorry

/-- The least number in the second row -/
def minSecondRow (g : Grid 17) : ℕ :=
  sorry

/-- Theorem stating that the sum of the greatest and least numbers in the second row is 528 -/
theorem second_row_sum_is_528 :
  let g := fillGrid 17
  maxSecondRow g + minSecondRow g = 528 :=
sorry

end second_row_sum_is_528_l347_34763


namespace crayons_per_pack_l347_34770

/-- Given that Nancy bought a total of 615 crayons in 41 packs,
    prove that there were 15 crayons in each pack. -/
theorem crayons_per_pack :
  ∀ (total_crayons : ℕ) (num_packs : ℕ),
    total_crayons = 615 →
    num_packs = 41 →
    total_crayons / num_packs = 15 :=
by sorry

end crayons_per_pack_l347_34770


namespace average_problem_l347_34747

theorem average_problem (x : ℝ) : (0.4 + x) / 2 = 0.2025 → x = 0.005 := by
  sorry

end average_problem_l347_34747


namespace average_age_of_boys_l347_34792

def boys_ages (x : ℝ) : Fin 3 → ℝ
| 0 => 3 * x
| 1 => 5 * x
| 2 => 7 * x

theorem average_age_of_boys (x : ℝ) (h1 : boys_ages x 2 = 21) :
  (boys_ages x 0 + boys_ages x 1 + boys_ages x 2) / 3 = 15 := by
  sorry

#check average_age_of_boys

end average_age_of_boys_l347_34792


namespace mean_calculation_l347_34738

theorem mean_calculation (x : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 62 → 
  (48 + 62 + 98 + 124 + x) / 5 = 78 := by
sorry

end mean_calculation_l347_34738


namespace equation_solution_l347_34708

theorem equation_solution (x : ℝ) : 
  x ≠ 3 ∧ x ≠ -3 → (4 / (x^2 - 9) - x / (3 - x) = 1 ↔ x = -13/3) :=
by sorry

end equation_solution_l347_34708


namespace f_properties_l347_34701

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem f_properties (a b : ℝ) (h : a * b ≠ 0) :
  (∀ x, f 1 (-Real.sqrt 3) x = 2 * Real.sin (2 * (x - Real.pi / 6))) ∧
  (a = b → ∀ x, f a b (x + Real.pi / 4) = f a b (Real.pi / 4 - x)) :=
sorry

end f_properties_l347_34701


namespace equation_solution_l347_34726

theorem equation_solution : ∃ x : ℝ, x ≠ 2 ∧ (4*x^2 + 3*x + 2) / (x - 2) = 4*x + 5 → x = -2 := by
  sorry

end equation_solution_l347_34726


namespace expression_divisibility_l347_34715

theorem expression_divisibility (n : ℕ) (x : ℝ) (hx : x ≠ 1) :
  ∃ g : ℝ → ℝ, n * x^(n+1) * (1 - 1/x) - x^n * (1 - 1/x^n) = (x - 1)^2 * g x :=
by sorry

end expression_divisibility_l347_34715


namespace jake_kendra_weight_ratio_l347_34788

/-- The problem of Jake and Kendra's weight ratio -/
theorem jake_kendra_weight_ratio :
  ∀ (j k : ℝ),
  j + k = 293 →
  j - 8 = 2 * k →
  (j - 8) / k = 2 :=
by sorry

end jake_kendra_weight_ratio_l347_34788


namespace necessary_but_not_sufficient_l347_34705

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ M ∩ P → (x ∈ M ∨ x ∈ P)) ∧
  (∃ x, (x ∈ M ∨ x ∈ P) ∧ x ∉ M ∩ P) :=
by sorry

end necessary_but_not_sufficient_l347_34705


namespace modulus_of_12_plus_5i_l347_34727

theorem modulus_of_12_plus_5i : Complex.abs (12 + 5 * Complex.I) = 13 := by
  sorry

end modulus_of_12_plus_5i_l347_34727


namespace tv_sale_value_change_l347_34766

theorem tv_sale_value_change 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_reduction_percent : ℝ) 
  (sales_increase_percent : ℝ) 
  (h1 : price_reduction_percent = 10) 
  (h2 : sales_increase_percent = 85) : 
  let new_price := original_price * (1 - price_reduction_percent / 100)
  let new_quantity := original_quantity * (1 + sales_increase_percent / 100)
  let original_value := original_price * original_quantity
  let new_value := new_price * new_quantity
  (new_value - original_value) / original_value * 100 = 66.5 := by
sorry

end tv_sale_value_change_l347_34766


namespace number_puzzle_l347_34797

theorem number_puzzle : ∃ x : ℝ, x + (1/5) * x + 1 = 10 ∧ x = 7.5 := by sorry

end number_puzzle_l347_34797


namespace function_bound_l347_34772

def ContinuousFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f x - f y| < |x - y|

theorem function_bound (f : ℝ → ℝ) (h1 : ContinuousFunction f) (h2 : f 0 = f 1) :
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f x - f y| < (1 : ℝ) / 2 := by
  sorry

end function_bound_l347_34772


namespace distance_minus_two_to_three_l347_34781

-- Define the distance function between two points on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem distance_minus_two_to_three : distance (-2) 3 = 5 := by
  sorry

end distance_minus_two_to_three_l347_34781


namespace arithmetic_sequence_sum_l347_34750

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: In an arithmetic sequence, if a_1 + a_2 = 5 and a_3 + a_4 = 7, then a_5 + a_6 = 9 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_sum_12 : a 1 + a 2 = 5)
    (h_sum_34 : a 3 + a 4 = 7) :
  a 5 + a 6 = 9 := by
  sorry

end arithmetic_sequence_sum_l347_34750


namespace shipping_cost_per_unit_l347_34706

/-- A computer manufacturer produces electronic components with the following parameters:
  * Production cost per component: $80
  * Fixed monthly costs: $16,200
  * Monthly production and sales: 150 components
  * Lowest break-even selling price: $190 per component
  This theorem proves that the shipping cost per unit is $2. -/
theorem shipping_cost_per_unit (production_cost : ℝ) (fixed_costs : ℝ) (units : ℝ) (selling_price : ℝ)
  (h1 : production_cost = 80)
  (h2 : fixed_costs = 16200)
  (h3 : units = 150)
  (h4 : selling_price = 190) :
  ∃ (shipping_cost : ℝ), 
    units * (production_cost + shipping_cost) + fixed_costs = units * selling_price ∧ 
    shipping_cost = 2 := by
  sorry

end shipping_cost_per_unit_l347_34706


namespace number_problem_l347_34728

theorem number_problem (x : ℝ) : 0.95 * x - 12 = 178 ↔ x = 200 := by
  sorry

end number_problem_l347_34728


namespace geometric_sequence_fourth_term_l347_34719

/-- Given a geometric sequence {aₙ} with a₁ + a₂ = -1 and a₁ - a₃ = -3, prove that a₄ = -8 -/
theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∃ (q : ℝ), ∀ n, a (n + 1) = a n * q)  -- Geometric sequence condition
  (h_sum : a 1 + a 2 = -1)  -- First condition
  (h_diff : a 1 - a 3 = -3)  -- Second condition
  : a 4 = -8 := by
  sorry

end geometric_sequence_fourth_term_l347_34719


namespace cos_570_deg_l347_34741

theorem cos_570_deg : Real.cos (570 * π / 180) = - Real.sqrt 3 / 2 := by
  sorry

end cos_570_deg_l347_34741


namespace find_b_value_l347_34769

theorem find_b_value (x b : ℝ) (h1 : 5 * x + 3 = b * x - 22) (h2 : x = 5) : b = 10 := by
  sorry

end find_b_value_l347_34769


namespace sum_of_three_numbers_l347_34745

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 149)
  (sum_of_products : a*b + b*c + c*a = 70) :
  a + b + c = 17 := by
  sorry

end sum_of_three_numbers_l347_34745


namespace circle_center_coordinate_sum_l347_34779

theorem circle_center_coordinate_sum (x y : ℝ) : 
  (x^2 + y^2 = 8*x - 6*y - 20) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 20) ∧ h + k = 1) :=
by sorry

end circle_center_coordinate_sum_l347_34779


namespace prize_selection_theorem_l347_34743

/-- Represents the systematic sampling of prizes -/
def systematicSampling (totalPrizes : ℕ) (sampleSize : ℕ) (firstPrize : ℕ) : List ℕ :=
  let interval := totalPrizes / sampleSize
  List.range sampleSize |>.map (fun i => firstPrize + i * interval)

/-- Theorem: Given the conditions of the prize selection, the other four prizes are 46, 86, 126, and 166 -/
theorem prize_selection_theorem (totalPrizes : ℕ) (sampleSize : ℕ) (firstPrize : ℕ) 
    (h1 : totalPrizes = 200)
    (h2 : sampleSize = 5)
    (h3 : firstPrize = 6) :
  systematicSampling totalPrizes sampleSize firstPrize = [6, 46, 86, 126, 166] := by
  sorry

#eval systematicSampling 200 5 6

end prize_selection_theorem_l347_34743


namespace permutations_of_six_books_l347_34729

theorem permutations_of_six_books : Nat.factorial 6 = 720 := by
  sorry

end permutations_of_six_books_l347_34729


namespace quadratic_inequality_solution_set_l347_34724

theorem quadratic_inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) → -2 < m ∧ m < 2 := by
  sorry

end quadratic_inequality_solution_set_l347_34724


namespace percentage_boys_playing_soccer_is_86_percent_l347_34733

/-- Calculates the percentage of boys among students playing soccer -/
def percentage_boys_playing_soccer (total_students : ℕ) (num_boys : ℕ) (students_playing_soccer : ℕ) (girls_not_playing_soccer : ℕ) : ℚ :=
  let total_girls : ℕ := total_students - num_boys
  let girls_playing_soccer : ℕ := total_girls - girls_not_playing_soccer
  let boys_playing_soccer : ℕ := students_playing_soccer - girls_playing_soccer
  (boys_playing_soccer : ℚ) / (students_playing_soccer : ℚ) * 100

/-- Theorem stating that the percentage of boys playing soccer is 86% -/
theorem percentage_boys_playing_soccer_is_86_percent :
  percentage_boys_playing_soccer 420 312 250 73 = 86 := by
  sorry

end percentage_boys_playing_soccer_is_86_percent_l347_34733


namespace inequality_solution_set_inequality_with_conditions_l347_34752

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x + f (x + 4) ≥ 8} = {x : ℝ | x ≤ -5 ∨ x ≥ 3} := by sorry

-- Theorem for the inequality with conditions
theorem inequality_with_conditions (a b : ℝ) 
  (ha : |a| < 1) (hb : |b| < 1) (ha_neq_zero : a ≠ 0) :
  f (a * b) > |a| * f (b / a) := by sorry

end inequality_solution_set_inequality_with_conditions_l347_34752


namespace some_mythical_beings_are_mystical_spirits_l347_34794

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (Dragon : U → Prop)
variable (MythicalBeing : U → Prop)
variable (MysticalSpirit : U → Prop)

-- State the theorem
theorem some_mythical_beings_are_mystical_spirits
  (h1 : ∀ x, Dragon x → MythicalBeing x)
  (h2 : ∃ x, MysticalSpirit x ∧ Dragon x) :
  ∃ x, MythicalBeing x ∧ MysticalSpirit x :=
by sorry

end some_mythical_beings_are_mystical_spirits_l347_34794


namespace sum_odd_numbers_100_to_200_l347_34723

def sum_odd_numbers_between (a b : ℕ) : ℕ :=
  let first_odd := if a % 2 = 0 then a + 1 else a
  let last_odd := if b % 2 = 0 then b - 1 else b
  let n := (last_odd - first_odd) / 2 + 1
  n * (first_odd + last_odd) / 2

theorem sum_odd_numbers_100_to_200 :
  sum_odd_numbers_between 100 200 = 7500 := by
  sorry

end sum_odd_numbers_100_to_200_l347_34723


namespace circle_radius_from_chords_l347_34712

/-- Given a circle with two chords of lengths 20 cm and 26 cm starting from the same point
    and forming an angle of 36° 38', the radius of the circle is approximately 24.84 cm. -/
theorem circle_radius_from_chords (chord1 chord2 angle : ℝ) (h1 : chord1 = 20)
    (h2 : chord2 = 26) (h3 : angle = 36 + 38 / 60) : ∃ r : ℝ, 
    abs (r - 24.84) < 0.01 ∧ 
    chord1^2 + chord2^2 - 2 * chord1 * chord2 * Real.cos (angle * Real.pi / 180) = 
    4 * r^2 * Real.sin ((angle * Real.pi / 180) / 2)^2 := by
  sorry

end circle_radius_from_chords_l347_34712


namespace cost_reduction_per_meter_l347_34736

/-- Proves that the reduction in cost per meter is 1 Rs -/
theorem cost_reduction_per_meter
  (original_cost : ℝ)
  (original_length : ℝ)
  (new_length : ℝ)
  (h_original_cost : original_cost = 35)
  (h_original_length : original_length = 10)
  (h_new_length : new_length = 14)
  (h_total_cost_unchanged : original_cost = new_length * (original_cost / original_length - x))
  : x = 1 :=
by
  sorry

end cost_reduction_per_meter_l347_34736


namespace playground_girls_l347_34734

theorem playground_girls (total_children : ℕ) (boys : ℕ) (girls : ℕ) :
  total_children = 97 → boys = 44 → girls = total_children - boys → girls = 53 := by
  sorry

end playground_girls_l347_34734


namespace tank_plastering_cost_per_sqm_l347_34702

/-- Given a tank with specified dimensions and total plastering cost, 
    calculate the cost per square meter for plastering. -/
theorem tank_plastering_cost_per_sqm 
  (length width depth : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 25) 
  (h2 : width = 12) 
  (h3 : depth = 6) 
  (h4 : total_cost = 186) : 
  total_cost / (length * width + 2 * length * depth + 2 * width * depth) = 0.25 := by
  sorry

#check tank_plastering_cost_per_sqm

end tank_plastering_cost_per_sqm_l347_34702


namespace a_minus_b_value_l347_34714

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 2) (h2 : b^2 = 9) (h3 : a < b) :
  a - b = -1 ∨ a - b = -5 := by
sorry

end a_minus_b_value_l347_34714


namespace lcm_gcd_product_36_60_l347_34767

theorem lcm_gcd_product_36_60 : Nat.lcm 36 60 * Nat.gcd 36 60 = 2160 := by
  sorry

end lcm_gcd_product_36_60_l347_34767


namespace singleton_quadratic_set_l347_34760

theorem singleton_quadratic_set (m : ℝ) : 
  (∃! x : ℝ, x^2 - 4*x + m = 0) → m = 4 := by
sorry

end singleton_quadratic_set_l347_34760


namespace rectangular_plot_breadth_l347_34717

theorem rectangular_plot_breadth (length breadth area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 972 →
  breadth = 18 := by
sorry

end rectangular_plot_breadth_l347_34717


namespace phi_bounded_by_one_l347_34707

/-- The functional equation satisfied by f and φ -/
def FunctionalEquation (f φ : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) + f (x - y) = 2 * φ y * f x

/-- f is not identically zero -/
def NotIdenticallyZero (f : ℝ → ℝ) : Prop :=
  ∃ x, f x ≠ 0

/-- The absolute value of f is bounded by 1 -/
def BoundedByOne (f : ℝ → ℝ) : Prop :=
  ∀ x, |f x| ≤ 1

/-- The main theorem -/
theorem phi_bounded_by_one
    (f φ : ℝ → ℝ)
    (h_eq : FunctionalEquation f φ)
    (h_nz : NotIdenticallyZero f)
    (h_bound : BoundedByOne f) :
    BoundedByOne φ := by
  sorry

end phi_bounded_by_one_l347_34707


namespace domain_of_composite_function_l347_34764

theorem domain_of_composite_function 
  (f : ℝ → ℝ) 
  (h : ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (2 * k * Real.pi - Real.pi / 6) (2 * k * Real.pi + 2 * Real.pi / 3) → f (Real.cos x) ∈ Set.range f) :
  Set.range f = Set.Icc (-1/2) 1 := by
sorry

end domain_of_composite_function_l347_34764


namespace consecutive_product_sum_l347_34740

theorem consecutive_product_sum : ∃ (a b c d e : ℤ),
  (b = a + 1) ∧
  (d = c + 1) ∧
  (e = d + 1) ∧
  (a * b = 990) ∧
  (c * d * e = 990) ∧
  (a + b + c + d + e = 90) :=
by sorry

end consecutive_product_sum_l347_34740
