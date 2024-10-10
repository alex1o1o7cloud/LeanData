import Mathlib

namespace smallest_square_multiplier_ten_l2878_287885

def is_smallest_square_multiplier (y : ℕ) (n : ℕ) : Prop :=
  y > 0 ∧ ∃ (m : ℕ), y * n = m^2 ∧
  ∀ (k : ℕ), k > 0 → k < y → ¬∃ (m : ℕ), k * n = m^2

theorem smallest_square_multiplier_ten (n : ℕ) :
  is_smallest_square_multiplier 10 n → n = 10 :=
by sorry

end smallest_square_multiplier_ten_l2878_287885


namespace first_term_greater_than_2017_l2878_287833

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem first_term_greater_than_2017 :
  ∃ n : ℕ, 
    arithmetic_sequence 5 7 n > 2017 ∧
    arithmetic_sequence 5 7 n = 2021 ∧
    ∀ m : ℕ, m < n → arithmetic_sequence 5 7 m ≤ 2017 :=
by sorry

end first_term_greater_than_2017_l2878_287833


namespace unique_even_solution_l2878_287896

def f (n : ℤ) : ℤ :=
  if n < 0 then n^2 + 4*n + 4 else 3*n - 15

theorem unique_even_solution :
  ∃! a : ℤ, Even a ∧ f (-3) + f 3 + f a = 0 :=
sorry

end unique_even_solution_l2878_287896


namespace age_ratio_problem_l2878_287816

/-- Mike's current age -/
def m : ℕ := sorry

/-- Ana's current age -/
def a : ℕ := sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem age_ratio_problem :
  (m - 3 = 4 * (a - 3)) ∧ 
  (m - 7 = 5 * (a - 7)) →
  x = 77 ∧ 
  (m + x) * 2 = (a + x) * 3 := by
  sorry

end age_ratio_problem_l2878_287816


namespace bus_rental_equation_l2878_287869

theorem bus_rental_equation (x : ℝ) (h : x > 2) :
  180 / x - 180 / (x + 2) = 3 :=
by sorry


end bus_rental_equation_l2878_287869


namespace fraction_equality_l2878_287828

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 4 / 3) 
  (h2 : r / t = 9 / 14) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end fraction_equality_l2878_287828


namespace bruce_mangoes_purchase_l2878_287829

theorem bruce_mangoes_purchase :
  let grapes_kg : ℕ := 8
  let grapes_price : ℕ := 70
  let mango_price : ℕ := 55
  let total_paid : ℕ := 1165
  let mango_kg : ℕ := (total_paid - grapes_kg * grapes_price) / mango_price
  mango_kg = 11 := by sorry

end bruce_mangoes_purchase_l2878_287829


namespace smallest_right_triangle_area_l2878_287857

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 7) (hb : b = 10) :
  let c := Real.sqrt (a^2 + b^2)
  let area := (1/2) * a * b
  area = 35 := by sorry

end smallest_right_triangle_area_l2878_287857


namespace correct_share_distribution_l2878_287882

def total_amount : ℕ := 12000
def ratio : List ℕ := [2, 4, 6, 3, 5]

def share_amount (total : ℕ) (ratios : List ℕ) : List ℕ :=
  let total_parts := ratios.sum
  let part_value := total / total_parts
  ratios.map (· * part_value)

theorem correct_share_distribution :
  share_amount total_amount ratio = [1200, 2400, 3600, 1800, 3000] := by
  sorry

end correct_share_distribution_l2878_287882


namespace equal_roots_real_roots_l2878_287873

/-- The quadratic equation given in the problem -/
def quadratic_equation (m x : ℝ) : Prop :=
  2 * (m + 1) * x^2 + 4 * m * x + 3 * m = 2

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  -8 * m^2 - 8 * m + 16

theorem equal_roots (m : ℝ) : 
  (∃ x : ℝ, quadratic_equation m x ∧ 
    ∀ y : ℝ, quadratic_equation m y → y = x) ↔ 
  (m = -2 ∨ m = 1) :=
sorry

theorem real_roots (m : ℝ) :
  (m = -1 → ∃ x : ℝ, quadratic_equation m x ∧ x = -5/4) ∧
  (m ≠ -1 → ∃ x : ℝ, quadratic_equation m x ∧ 
    ∃ s : ℝ, s^2 = -2*m^2 - 2*m + 4 ∧ 
      (x = (-2*m + s) / (2*(m+1)) ∨ x = (-2*m - s) / (2*(m+1)))) :=
sorry

end equal_roots_real_roots_l2878_287873


namespace complex_modulus_and_argument_l2878_287868

open Complex

theorem complex_modulus_and_argument : 
  let z : ℂ := -Complex.sin (π/8) - Complex.I * Complex.cos (π/8)
  (abs z = 1) ∧ (arg z = -5*π/8) := by sorry

end complex_modulus_and_argument_l2878_287868


namespace theta_value_l2878_287831

theorem theta_value : ∃! θ : ℕ, θ ∈ Finset.range 10 ∧ θ ≠ 0 ∧ 294 / θ = 30 + 4 * θ := by
  sorry

end theta_value_l2878_287831


namespace sum_cis_angle_sequence_l2878_287864

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the arithmetic sequence of angles
def angleSequence : List ℝ := List.range 12 |>.map (λ n => 70 + 8 * n)

-- State the theorem
theorem sum_cis_angle_sequence (r : ℝ) (θ : ℝ) 
  (h_r : r > 0) (h_θ : 0 ≤ θ ∧ θ < 360) :
  (angleSequence.map (λ α => cis α)).sum = r * cis θ → θ = 114 := by
  sorry

end sum_cis_angle_sequence_l2878_287864


namespace smallest_prime_divisor_of_sum_l2878_287830

theorem smallest_prime_divisor_of_sum (p : Nat) : 
  Prime p → p ∣ (2^14 + 7^9) → p > 7 := by
  sorry

end smallest_prime_divisor_of_sum_l2878_287830


namespace detergent_per_pound_l2878_287819

/-- Given that 18 ounces of detergent are used for 9 pounds of clothes,
    prove that 2 ounces of detergent are used per pound of clothes. -/
theorem detergent_per_pound (total_detergent : ℝ) (total_clothes : ℝ) 
  (h1 : total_detergent = 18) (h2 : total_clothes = 9) :
  total_detergent / total_clothes = 2 := by
  sorry

end detergent_per_pound_l2878_287819


namespace negation_of_forall_geq_zero_l2878_287854

theorem negation_of_forall_geq_zero (R : Type*) [OrderedRing R] :
  (¬ (∀ x : R, x^2 - 3 ≥ 0)) ↔ (∃ x₀ : R, x₀^2 - 3 < 0) :=
by sorry

end negation_of_forall_geq_zero_l2878_287854


namespace number_975_in_column_B_l2878_287858

/-- Represents the columns in the arrangement --/
inductive Column
| A | B | C | D | E | F

/-- Determines if a given row number is odd --/
def isOddRow (n : ℕ) : Bool :=
  n % 2 = 1

/-- Calculates the column for a given number in the arrangement --/
def columnForNumber (n : ℕ) : Column :=
  let adjustedN := n - 1
  let rowNumber := (adjustedN / 6) + 1
  let positionInRow := adjustedN % 6
  if isOddRow rowNumber then
    match positionInRow with
    | 0 => Column.A
    | 1 => Column.B
    | 2 => Column.C
    | 3 => Column.D
    | 4 => Column.E
    | _ => Column.F
  else
    match positionInRow with
    | 0 => Column.F
    | 1 => Column.E
    | 2 => Column.D
    | 3 => Column.C
    | 4 => Column.B
    | _ => Column.A

/-- Theorem: The integer 975 is in column B in the given arrangement --/
theorem number_975_in_column_B : columnForNumber 975 = Column.B := by
  sorry

end number_975_in_column_B_l2878_287858


namespace additive_increasing_non_neg_implies_odd_and_increasing_l2878_287888

/-- A function satisfying f(x₁ + x₂) = f(x₁) + f(x₂) for all x₁, x₂ ∈ ℝ -/
def IsAdditive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂

/-- A function that is increasing on non-negative reals -/
def IsIncreasingNonNeg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ x₂ → x₂ ≥ 0 → f x₁ ≥ f x₂

/-- Main theorem: If f is additive and increasing on non-negative reals,
    then it is odd and increasing on all reals -/
theorem additive_increasing_non_neg_implies_odd_and_increasing
    (f : ℝ → ℝ) (h1 : IsAdditive f) (h2 : IsIncreasingNonNeg f) :
    (∀ x, f (-x) = -f x) ∧ (∀ x₁ x₂, x₁ ≥ x₂ → f x₁ ≥ f x₂) := by
  sorry

end additive_increasing_non_neg_implies_odd_and_increasing_l2878_287888


namespace lunch_breakfast_difference_l2878_287809

def muffin_cost : ℚ := 2
def coffee_cost : ℚ := 4
def soup_cost : ℚ := 3
def salad_cost : ℚ := 5.25
def lemonade_cost : ℚ := 0.75

def breakfast_cost : ℚ := muffin_cost + coffee_cost
def lunch_cost : ℚ := soup_cost + salad_cost + lemonade_cost

theorem lunch_breakfast_difference :
  lunch_cost - breakfast_cost = 3 := by sorry

end lunch_breakfast_difference_l2878_287809


namespace blue_regular_polygon_l2878_287844

/-- A circle with some red points and the rest blue -/
structure ColoredCircle where
  redPoints : Finset ℝ
  (red_count : redPoints.card = 2016)

/-- A regular n-gon inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Finset ℝ
  (vertex_count : vertices.card = n)

/-- The theorem statement -/
theorem blue_regular_polygon
  (circle : ColoredCircle)
  (n : ℕ)
  (h : n ≥ 3) :
  ∃ (poly : RegularPolygon n), poly.vertices ∩ circle.redPoints = ∅ :=
sorry

end blue_regular_polygon_l2878_287844


namespace house_legs_l2878_287859

/-- The number of legs in a house with humans and various pets -/
def total_legs (humans dogs cats parrots goldfish : ℕ) : ℕ :=
  humans * 2 + dogs * 4 + cats * 4 + parrots * 2 + goldfish * 0

/-- Theorem: The total number of legs in the house is 38 -/
theorem house_legs : total_legs 5 2 3 4 5 = 38 := by
  sorry

end house_legs_l2878_287859


namespace harkamal_payment_l2878_287826

/-- The total amount Harkamal paid for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 965 for his purchase -/
theorem harkamal_payment : total_amount 8 70 9 45 = 965 := by
  sorry

end harkamal_payment_l2878_287826


namespace a_2010_at_1_l2878_287870

def a : ℕ → (ℝ → ℝ)
  | 0 => λ x => 1
  | 1 => λ x => x^2 + x + 1
  | (n+2) => λ x => (x^(n+2) + 1) * a (n+1) x - a n x

theorem a_2010_at_1 : a 2010 1 = 4021 := by
  sorry

end a_2010_at_1_l2878_287870


namespace quadratic_function_properties_l2878_287837

theorem quadratic_function_properties (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ioo 1 3 ↔ a * x^2 + b * x + c > -2 * x) →
  (a < 0 ∧
   b = -4 * a - 2 ∧
   (∀ x : ℝ, (a * x^2 + b * x + c + 6 * a = 0 → 
    ∃! r : ℝ, a * r^2 + b * r + c + 6 * a = 0) → a = -1/5)) :=
by sorry

end quadratic_function_properties_l2878_287837


namespace grid_bottom_right_value_l2878_287845

/-- Represents a 3x3 grid with some known values -/
structure Grid :=
  (a b c d e f g h i : ℕ)
  (b_eq_6 : b = 6)
  (c_eq_3 : c = 3)
  (h_eq_2 : h = 2)
  (all_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0)

/-- The product of each row, column, and diagonal is the same -/
def grid_property (grid : Grid) : Prop :=
  let p := grid.a * grid.b * grid.c
  p = grid.d * grid.e * grid.f ∧
  p = grid.g * grid.h * grid.i ∧
  p = grid.a * grid.d * grid.g ∧
  p = grid.b * grid.e * grid.h ∧
  p = grid.c * grid.f * grid.i ∧
  p = grid.a * grid.e * grid.i ∧
  p = grid.c * grid.e * grid.g

theorem grid_bottom_right_value (grid : Grid) (h : grid_property grid) : grid.i = 36 := by
  sorry

end grid_bottom_right_value_l2878_287845


namespace circular_table_seating_l2878_287889

theorem circular_table_seating (n : ℕ) (a : Fin (2*n) → Fin (2*n)) 
  (h_perm : Function.Bijective a) :
  ∃ i j : Fin (2*n), i ≠ j ∧ 
    (a i - a j : ℤ) % (2*n) = (i - j : ℤ) % (2*n) ∨
    (a i - a j : ℤ) % (2*n) = (i - j - 2*n : ℤ) % (2*n) :=
by sorry

end circular_table_seating_l2878_287889


namespace cross_section_area_l2878_287891

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_side : ℝ
  height : ℝ

/-- Plane that intersects the pyramid -/
structure IntersectingPlane where
  perpendicular_to_base : Prop
  bisects_two_sides : Prop

/-- The cross-section created by the intersecting plane -/
def cross_section (p : RegularTriangularPyramid) (plane : IntersectingPlane) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The area of a given set in 3D space -/
def area (s : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area 
  (p : RegularTriangularPyramid) 
  (plane : IntersectingPlane) 
  (h1 : p.base_side = 2) 
  (h2 : p.height = 4) 
  (h3 : plane.perpendicular_to_base) 
  (h4 : plane.bisects_two_sides) : 
  area (cross_section p plane) = 1.5 :=
by sorry

end cross_section_area_l2878_287891


namespace fourth_root_logarithm_equality_l2878_287866

theorem fourth_root_logarithm_equality : 
  (16 ^ 3) ^ (1/4) - (25/4) ^ (1/2) + (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) = 15/2 := by
  sorry

end fourth_root_logarithm_equality_l2878_287866


namespace hcf_of_48_and_64_l2878_287895

theorem hcf_of_48_and_64 : 
  let a := 48
  let b := 64
  let lcm := 192
  Nat.lcm a b = lcm → Nat.gcd a b = 16 := by
sorry

end hcf_of_48_and_64_l2878_287895


namespace mapping_not_necessarily_injective_l2878_287817

variable {A B : Type}
variable (f : A → B)

theorem mapping_not_necessarily_injective : 
  ¬(∀ (x y : A), f x = f y → x = y) :=
sorry

end mapping_not_necessarily_injective_l2878_287817


namespace cubic_polynomial_satisfies_conditions_l2878_287898

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => -2/3 * x^3 + 2 * x^2 - 8/3 * x - 16/3
  (q 1 = -6) ∧ (q 2 = -8) ∧ (q 3 = -14) ∧ (q 4 = -28) := by
  sorry

end cubic_polynomial_satisfies_conditions_l2878_287898


namespace mean_score_calculation_l2878_287810

theorem mean_score_calculation (f s : ℕ) (F S : ℝ) : 
  F = 92 →
  S = 78 →
  f = 2 * s / 3 →
  (F * f + S * s) / (f + s) = 83.6 :=
by
  sorry

end mean_score_calculation_l2878_287810


namespace elvis_songwriting_time_l2878_287824

/-- Given Elvis's album recording scenario, prove that the time spent writing each song is 15 minutes. -/
theorem elvis_songwriting_time (total_songs : ℕ) (studio_time : ℕ) (recording_time_per_song : ℕ) (total_editing_time : ℕ) :
  total_songs = 10 →
  studio_time = 5 * 60 →
  recording_time_per_song = 12 →
  total_editing_time = 30 →
  (studio_time - (total_songs * recording_time_per_song + total_editing_time)) / total_songs = 15 :=
by sorry

end elvis_songwriting_time_l2878_287824


namespace original_movie_length_l2878_287801

/-- The original length of a movie, given the length of a cut scene and the final length -/
theorem original_movie_length (cut_scene_length final_length : ℕ) :
  cut_scene_length = 8 ∧ final_length = 52 →
  cut_scene_length + final_length = 60 := by
  sorry

#check original_movie_length

end original_movie_length_l2878_287801


namespace inscribed_circle_square_side_length_l2878_287892

theorem inscribed_circle_square_side_length 
  (circle_area : ℝ) 
  (h_area : circle_area = 36 * Real.pi) : 
  ∃ (square_side : ℝ), 
    square_side = 12 ∧ 
    circle_area = Real.pi * (square_side / 2) ^ 2 :=
by sorry

end inscribed_circle_square_side_length_l2878_287892


namespace ball_box_arrangements_count_l2878_287863

/-- The number of ways to put 4 distinguishable balls into 4 distinguishable boxes,
    where one particular ball cannot be placed in one specific box. -/
def ball_box_arrangements : ℕ :=
  let num_balls : ℕ := 4
  let num_boxes : ℕ := 4
  let restricted_ball_choices : ℕ := num_boxes - 1
  let unrestricted_ball_choices : ℕ := num_boxes
  restricted_ball_choices * (unrestricted_ball_choices ^ (num_balls - 1))

/-- Theorem stating that the number of arrangements is 192. -/
theorem ball_box_arrangements_count : ball_box_arrangements = 192 := by
  sorry

end ball_box_arrangements_count_l2878_287863


namespace binomial_coefficient_sum_l2878_287840

theorem binomial_coefficient_sum (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 3^7 - 1 := by
sorry

end binomial_coefficient_sum_l2878_287840


namespace johns_purchase_cost_l2878_287875

/-- The total cost of John's purchase of gum and candy bars -/
def total_cost (gum_packs : ℕ) (candy_bars : ℕ) (candy_bar_price : ℚ) : ℚ :=
  let gum_pack_price := candy_bar_price / 2
  gum_packs * gum_pack_price + candy_bars * candy_bar_price

/-- Theorem stating that the total cost of John's purchase is $6 -/
theorem johns_purchase_cost :
  total_cost 2 3 (3/2) = 6 := by
  sorry

end johns_purchase_cost_l2878_287875


namespace root_product_sum_l2878_287802

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (Real.sqrt 2015) * x₁^3 - 4030 * x₁^2 + 2 = 0 ∧
  (Real.sqrt 2015) * x₂^3 - 4030 * x₂^2 + 2 = 0 ∧
  (Real.sqrt 2015) * x₃^3 - 4030 * x₃^2 + 2 = 0 →
  x₂ * (x₁ + x₃) = 2 := by
sorry

end root_product_sum_l2878_287802


namespace initial_music_files_count_l2878_287852

/-- The number of music files Vanessa initially had -/
def initial_music_files : ℕ := sorry

/-- The number of video files Vanessa initially had -/
def initial_video_files : ℕ := 48

/-- The number of files Vanessa deleted -/
def deleted_files : ℕ := 30

/-- The number of files remaining after deletion -/
def remaining_files : ℕ := 34

/-- Theorem stating that the initial number of music files is 16 -/
theorem initial_music_files_count : initial_music_files = 16 := by
  sorry

end initial_music_files_count_l2878_287852


namespace max_fraction_value_l2878_287856

theorem max_fraction_value (A B : ℝ) (h1 : A + B = 2020) (h2 : A / B < 1 / 4) :
  A / B ≤ 403 / 1617 :=
sorry

end max_fraction_value_l2878_287856


namespace balls_per_bag_l2878_287818

theorem balls_per_bag (total_balls : ℕ) (num_bags : ℕ) (balls_per_bag : ℕ) 
  (h1 : total_balls = 36)
  (h2 : num_bags = 9)
  (h3 : total_balls = num_bags * balls_per_bag) :
  balls_per_bag = 4 := by
sorry

end balls_per_bag_l2878_287818


namespace imaginary_part_of_product_l2878_287897

/-- The imaginary part of the product of two complex numbers -/
theorem imaginary_part_of_product (ω₁ ω₂ : ℂ) : 
  let z := ω₁ * ω₂
  ω₁ = -1/2 + (Real.sqrt 3/2) * I →
  ω₂ = Complex.exp (I * (π/12)) →
  z.im = Real.sqrt 2/2 := by
  sorry

end imaginary_part_of_product_l2878_287897


namespace white_l_shapes_imply_all_white_2x2_l2878_287876

/-- Represents a grid cell that can be either black or white -/
inductive Color
| Black
| White

/-- Represents an m × n grid -/
def Grid (m n : ℕ) := Fin m → Fin n → Color

/-- Counts the number of L-shapes with exactly three white squares in a grid -/
def countWhiteLShapes (g : Grid m n) : ℕ := sorry

/-- Checks if there exists a 2 × 2 grid with all white squares -/
def existsAllWhite2x2 (g : Grid m n) : Prop := sorry

/-- Main theorem: If the number of L-shapes with three white squares is at least mn/3,
    then there exists a 2 × 2 grid with all white squares -/
theorem white_l_shapes_imply_all_white_2x2 
  (m n : ℕ) (hm : m > 0) (hn : n > 0) (g : Grid m n) :
  countWhiteLShapes g ≥ m * n / 3 → existsAllWhite2x2 g :=
sorry

end white_l_shapes_imply_all_white_2x2_l2878_287876


namespace sqrt_less_than_linear_l2878_287805

theorem sqrt_less_than_linear (x : ℝ) (hx : x > 0) : 
  Real.sqrt (1 + x) < 1 + x / 2 := by
  sorry

end sqrt_less_than_linear_l2878_287805


namespace circle_center_and_radius_l2878_287804

/-- Given a circle with equation (x+1)^2 + (y-1)^2 = 4, prove that its center is (-1,1) and its radius is 2 -/
theorem circle_center_and_radius :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    (∀ (x y : ℝ), (x + 1)^2 + (y - 1)^2 = 4 ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) ∧
    C = (-1, 1) ∧
    r = 2 := by
  sorry

end circle_center_and_radius_l2878_287804


namespace expression_range_l2878_287890

/-- The quadratic equation in terms of x with parameter m -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*(m-2)*x + m^2 + 4

/-- Predicate to check if the quadratic equation has two real roots -/
def has_two_real_roots (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0

/-- The expression we want to find the range of -/
def expression (x₁ x₂ : ℝ) : ℝ := x₁^2 + x₂^2 - x₁*x₂

theorem expression_range :
  ∀ m : ℝ, has_two_real_roots m →
    (∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ 
      expression x₁ x₂ ≥ 4 ∧ 
      ∀ ε > 0, ∃ m' : ℝ, has_two_real_roots m' ∧ 
        ∃ y₁ y₂ : ℝ, quadratic m' y₁ = 0 ∧ quadratic m' y₂ = 0 ∧ 
          expression y₁ y₂ < 4 + ε) :=
sorry

end expression_range_l2878_287890


namespace function_properties_l2878_287867

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - 2 * Real.sin (ω * x / 2) ^ 2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem function_properties
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (h_period : is_periodic (f ω) (3 * Real.pi))
  (h_min_period : ∀ T, 0 < T ∧ T < 3 * Real.pi → ¬ is_periodic (f ω) T)
  (A B C : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (h_f_C : f ω C = 1)
  (h_trig_eq : 2 * Real.sin (2 * B) = Real.cos B + Real.cos (A - C)) :
  (∃ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), ∀ y ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f ω x ≤ f ω y) ∧
  f ω (Real.pi / 2) = Real.sqrt 3 - 1 ∧
  Real.sin A = (Real.sqrt 5 - 1) / 2 := by
sorry

end function_properties_l2878_287867


namespace susan_reading_time_l2878_287811

/-- Given Susan's free time activities ratio and time spent with friends, calculate reading time -/
theorem susan_reading_time (swimming reading friends : ℕ) 
  (ratio : swimming + reading + friends = 15) 
  (swim_ratio : swimming = 1)
  (read_ratio : reading = 4)
  (friend_ratio : friends = 10)
  (friend_time : ℕ) 
  (friend_hours : friend_time = 20) : 
  (friend_time * reading) / friends = 8 := by
  sorry

end susan_reading_time_l2878_287811


namespace chord_length_problem_l2878_287835

/-- The length of the chord formed by the intersection of a line and a circle -/
def chord_length (line_point : ℝ × ℝ) (parallel_line : ℝ → ℝ → ℝ → Prop) 
  (circle_center : ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  sorry

/-- The problem statement -/
theorem chord_length_problem :
  let line_point := (1, 0)
  let parallel_line := λ x y c => x - Real.sqrt 2 * y + c = 0
  let circle_center := (6, Real.sqrt 2)
  let circle_radius := Real.sqrt 12
  chord_length line_point parallel_line circle_center circle_radius = 6 := by
  sorry

end chord_length_problem_l2878_287835


namespace angle_with_same_terminal_side_l2878_287832

theorem angle_with_same_terminal_side : ∃ k : ℤ, 2019 + k * 360 = -141 := by
  sorry

end angle_with_same_terminal_side_l2878_287832


namespace min_sum_squares_l2878_287865

/-- Parabola defined by y² = 4x -/
def Parabola (x y : ℝ) : Prop := y^2 = 4 * x

/-- Line passing through (4, 0) -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

/-- Intersection points of the line and parabola -/
def Intersection (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  Parabola x₁ y₁ ∧ Parabola x₂ y₂ ∧ Line k x₁ y₁ ∧ Line k x₂ y₂ ∧ x₁ ≠ x₂

theorem min_sum_squares :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  Intersection k x₁ y₁ x₂ y₂ →
  y₁^2 + y₂^2 ≥ 32 :=
sorry

end min_sum_squares_l2878_287865


namespace partial_fraction_sum_l2878_287872

open Real

theorem partial_fraction_sum : ∃ (A B C D E : ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) ∧
  A + B + C + D + E = 1 / 6 := by
sorry

end partial_fraction_sum_l2878_287872


namespace ladder_length_l2878_287879

theorem ladder_length (angle : Real) (adjacent : Real) (hypotenuse : Real) :
  angle = Real.pi / 3 →  -- 60 degrees in radians
  adjacent = 9.493063650744542 →
  hypotenuse = adjacent / Real.cos angle →
  hypotenuse = 18.986127301489084 := by
  sorry

end ladder_length_l2878_287879


namespace no_prime_roots_for_quadratic_l2878_287855

-- Define a quadratic equation
def quadratic_equation (k : ℤ) (x : ℤ) : Prop := x^2 - 65*x + k = 0

-- Define primality
def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem no_prime_roots_for_quadratic :
  ¬∃ k : ℤ, ∃ x y : ℤ, 
    x ≠ y ∧ 
    quadratic_equation k x ∧ 
    quadratic_equation k y ∧
    is_prime x ∧ 
    is_prime y :=
sorry

end no_prime_roots_for_quadratic_l2878_287855


namespace triangle_with_equal_angles_isosceles_l2878_287838

/-- A triangle is isosceles if it has at least two equal angles. -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

/-- Given a triangle ABC where ∠A = ∠B = 2∠C, prove that the triangle is isosceles. -/
theorem triangle_with_equal_angles_isosceles (a b c : ℝ) 
  (h1 : a + b + c = 180) -- Sum of angles in a triangle is 180°
  (h2 : a = b) -- ∠A = ∠B
  (h3 : a = 2 * c) -- ∠A = 2∠C
  : IsIsosceles a b c :=
sorry

end triangle_with_equal_angles_isosceles_l2878_287838


namespace area_of_quadrilateral_ABD_l2878_287853

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a plane -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Represents a quadrilateral -/
structure Quadrilateral where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D

/-- Function to calculate the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Main theorem statement -/
theorem area_of_quadrilateral_ABD (cube : Cube) (plane : Plane) (quadABD : Quadrilateral) :
  cube.sideLength = 2 →
  -- A is a vertex of the cube
  -- B and D are midpoints of edges adjacent to A
  -- C' is the midpoint of a face diagonal not including A
  -- Plane passes through A, B, D, and C'
  -- quadABD lies in the plane
  areaQuadrilateral quadABD = 2 := by sorry

end area_of_quadrilateral_ABD_l2878_287853


namespace sum_equals_product_l2878_287871

theorem sum_equals_product (x : ℝ) (h : x ≠ 1) :
  ∃! y : ℝ, x + y = x * y ∧ y = x / (x - 1) := by sorry

end sum_equals_product_l2878_287871


namespace fraction_multiplication_l2878_287886

theorem fraction_multiplication : (2 : ℚ) / 5 * (7 : ℚ) / 10 = (7 : ℚ) / 25 := by
  sorry

end fraction_multiplication_l2878_287886


namespace candle_arrangement_l2878_287899

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of candles that satisfies the given conditions -/
def num_candles : ℕ := 4

theorem candle_arrangement :
  (∀ c : ℕ, (choose_2 c * 9 = 54) → c = num_candles) :=
by sorry

end candle_arrangement_l2878_287899


namespace expression_multiple_of_six_l2878_287877

theorem expression_multiple_of_six (n : ℕ) (h : n ≥ 12) :
  ∃ k : ℤ, ((n + 3).factorial - 2 * (n + 2).factorial) / (n + 1).factorial = 6 * k := by
  sorry

end expression_multiple_of_six_l2878_287877


namespace arithmetic_sequence_fifth_term_l2878_287851

/-- Given an arithmetic sequence where:
  a₁ = 3 (first term)
  a₂ = 7 (second term)
  a₃ = 11 (third term)
  Prove that a₅ = 19 (fifth term)
-/
theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℤ), 
    (a 1 = 3) →  -- First term
    (a 2 = 7) →  -- Second term
    (a 3 = 11) → -- Third term
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) → -- Arithmetic sequence property
    a 5 = 19 := by
  sorry

end arithmetic_sequence_fifth_term_l2878_287851


namespace rectangular_plot_length_difference_l2878_287850

theorem rectangular_plot_length_difference (length breadth : ℝ) : 
  length = 70 ∧ 
  length > breadth ∧ 
  26.50 * (2 * length + 2 * breadth) = 5300 → 
  length - breadth = 40 := by
sorry

end rectangular_plot_length_difference_l2878_287850


namespace cone_surface_area_l2878_287807

/-- The surface area of a cone with lateral surface as a sector of a circle 
    with radius 2 and central angle π/2 is 5π/4 -/
theorem cone_surface_area : 
  ∀ (cone : Real → Real → Real),
  (∀ r θ, cone r θ = 2 * π * r^2 * (θ / (2 * π)) + π * r^2) →
  cone 2 (π / 2) = 5 * π / 4 :=
by sorry

end cone_surface_area_l2878_287807


namespace orange_count_l2878_287803

theorem orange_count (initial_apples : ℕ) (initial_oranges : ℕ) : 
  initial_apples = 14 →
  (initial_apples : ℚ) / (initial_apples + initial_oranges - 14 : ℚ) = 70 / 100 →
  initial_oranges = 20 :=
by sorry

end orange_count_l2878_287803


namespace chord_length_is_four_l2878_287878

/-- A circle with center at (0, 1) and radius 2, tangent to the line y = -1 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  center_eq : center = (0, 1)
  radius_eq : radius = 2
  tangent_to_line : ∀ (x y : ℝ), y = -1 → (x - center.1)^2 + (y - center.2)^2 ≥ radius^2

/-- The length of the chord intercepted by the circle on the y-axis -/
def chord_length (c : Circle) : ℝ :=
  let y₁ := c.center.2 + c.radius
  let y₂ := c.center.2 - c.radius
  y₁ - y₂

theorem chord_length_is_four (c : Circle) : chord_length c = 4 := by
  sorry

end chord_length_is_four_l2878_287878


namespace soda_filling_time_difference_l2878_287836

/-- Proves that the additional time needed to fill 12 barrels with a leak is 24 minutes -/
theorem soda_filling_time_difference 
  (normal_time : ℕ) 
  (leak_time : ℕ) 
  (barrel_count : ℕ) 
  (h1 : normal_time = 3)
  (h2 : leak_time = 5)
  (h3 : barrel_count = 12) :
  leak_time * barrel_count - normal_time * barrel_count = 24 := by
  sorry

end soda_filling_time_difference_l2878_287836


namespace line_passes_through_third_quadrant_l2878_287808

theorem line_passes_through_third_quadrant 
  (A B C : ℝ) (h1 : A * B < 0) (h2 : B * C < 0) :
  ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ A * x + B * y + C = 0 :=
sorry

end line_passes_through_third_quadrant_l2878_287808


namespace complex_equation_solution_l2878_287887

theorem complex_equation_solution (x y : ℝ) :
  Complex.I * (x + y) = x - 1 → x = 1 ∧ y = -1 := by
  sorry

end complex_equation_solution_l2878_287887


namespace unwashed_shirts_l2878_287834

theorem unwashed_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 40 → long_sleeve = 23 → washed = 29 → 
  short_sleeve + long_sleeve - washed = 34 := by
  sorry

end unwashed_shirts_l2878_287834


namespace distance_after_rest_l2878_287820

/-- The length of a football field in meters -/
def football_field_length : ℝ := 168

/-- The distance Nate ran before resting, in meters -/
def distance_before_rest : ℝ := 4 * football_field_length

/-- The total distance Nate ran, in meters -/
def total_distance : ℝ := 1172

/-- Theorem: The distance Nate ran after resting is 500 meters -/
theorem distance_after_rest :
  total_distance - distance_before_rest = 500 := by sorry

end distance_after_rest_l2878_287820


namespace comparison_arithmetic_geometric_mean_l2878_287839

theorem comparison_arithmetic_geometric_mean (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ¬(∀ a b c, (a + b + c) / 3 ≥ (a^2 * b * b * c * c * a)^(1/3)) ∧ 
  ¬(∀ a b c, (a + b + c) / 3 ≤ (a^2 * b * b * c * c * a)^(1/3)) ∧ 
  ¬(∀ a b c, (a + b + c) / 3 = (a^2 * b * b * c * c * a)^(1/3)) :=
by sorry

end comparison_arithmetic_geometric_mean_l2878_287839


namespace order_of_f_values_l2878_287849

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

theorem order_of_f_values :
  let a := f (Real.sqrt 2 / 2)
  let b := f (Real.sqrt 3 / 2)
  let c := f (Real.sqrt 6 / 2)
  b > c ∧ c > a := by sorry

end order_of_f_values_l2878_287849


namespace parabola_properties_l2878_287842

/-- A parabola with equation y = ax^2 + (2m-6)x + 1 passing through (1, 2m-4) -/
def Parabola (a m : ℝ) : ℝ → ℝ := λ x => a * x^2 + (2*m - 6) * x + 1

/-- Points on the parabola -/
def PointsOnParabola (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-m, Parabola 3 m (-m)), (m, Parabola 3 m m), (m+2, Parabola 3 m (m+2)))

theorem parabola_properties (m : ℝ) :
  let (y1, y2, y3) := (Parabola 3 m (-m), Parabola 3 m m, Parabola 3 m (m+2))
  Parabola 3 m 1 = 2*m - 4 ∧ 
  y2 < y3 ∧ y3 ≤ y1 →
  (3 : ℝ) = 3 ∧
  (3 - m : ℝ) = -((2*m - 6) / (2*3)) ∧
  1 < m ∧ m ≤ 2 := by sorry

#check parabola_properties

end parabola_properties_l2878_287842


namespace sum_mod_seven_l2878_287812

theorem sum_mod_seven : (5000 + 5001 + 5002 + 5003 + 5004) % 7 = 0 := by
  sorry

end sum_mod_seven_l2878_287812


namespace sins_match_prayers_l2878_287894

structure Sin :=
  (teDeum : ℕ)
  (paterNoster : ℕ)
  (credo : ℕ)

def pride : Sin := ⟨1, 2, 0⟩
def slander : Sin := ⟨0, 2, 7⟩
def sloth : Sin := ⟨2, 0, 0⟩
def adultery : Sin := ⟨10, 10, 10⟩
def gluttony : Sin := ⟨1, 0, 0⟩
def selfishness : Sin := ⟨0, 3, 1⟩
def jealousy : Sin := ⟨0, 3, 0⟩
def evilSpeaking : Sin := ⟨0, 7, 2⟩

def totalPrayers (sins : List Sin) : Sin :=
  sins.foldl (λ acc sin => ⟨acc.teDeum + sin.teDeum, acc.paterNoster + sin.paterNoster, acc.credo + sin.credo⟩) ⟨0, 0, 0⟩

theorem sins_match_prayers :
  let sins := [slander] ++ List.replicate 2 evilSpeaking ++ [selfishness] ++ List.replicate 9 gluttony
  totalPrayers sins = ⟨9, 12, 10⟩ := by sorry

end sins_match_prayers_l2878_287894


namespace min_hypotenuse_right_triangle_l2878_287825

theorem min_hypotenuse_right_triangle (k : ℝ) (h : k > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = k ∧
  a^2 + b^2 = c^2 ∧
  c = (Real.sqrt 2 - 1) * k ∧
  ∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 0 →
    a' + b' + c' = k → a'^2 + b'^2 = c'^2 → c' ≥ (Real.sqrt 2 - 1) * k := by
  sorry

end min_hypotenuse_right_triangle_l2878_287825


namespace m_range_theorem_l2878_287806

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x y : ℝ, y = x + m - 2 → ¬(x < 0 ∧ y > 0)

def q (m : ℝ) : Prop := 0 < 1 - m ∧ m < 1

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2)

-- State the theorem
theorem m_range_theorem :
  (∀ m : ℝ, ¬(p m ∧ q m)) →
  (∀ m : ℝ, p m ∨ q m) →
  ∀ m : ℝ, m_range m ↔ (p m ∨ q m) :=
sorry

end m_range_theorem_l2878_287806


namespace prob_at_least_two_long_specific_l2878_287893

/-- Represents the probability of a road being at least 5 miles long -/
structure RoadProbability where
  ab : ℚ  -- Probability for road A to B
  bc : ℚ  -- Probability for road B to C
  cd : ℚ  -- Probability for road C to D

/-- Calculates the probability of selecting at least two roads that are at least 5 miles long -/
def prob_at_least_two_long (p : RoadProbability) : ℚ :=
  p.ab * p.bc * (1 - p.cd) +  -- A to B and B to C are long, C to D is not
  p.ab * (1 - p.bc) * p.cd +  -- A to B and C to D are long, B to C is not
  (1 - p.ab) * p.bc * p.cd +  -- B to C and C to D are long, A to B is not
  p.ab * p.bc * p.cd          -- All three roads are long

theorem prob_at_least_two_long_specific : 
  let p : RoadProbability := { ab := 3/4, bc := 2/3, cd := 1/2 }
  prob_at_least_two_long p = 11/24 := by
  sorry

end prob_at_least_two_long_specific_l2878_287893


namespace bee_speed_solution_l2878_287880

/-- The speed of a honey bee's flight between flowers -/
def bee_speed_problem (time_daisy_rose time_rose_poppy : ℝ) 
  (distance_difference speed_difference : ℝ) : Prop :=
  let speed_daisy_rose : ℝ := 6.5
  let speed_rose_poppy : ℝ := speed_daisy_rose + speed_difference
  let distance_daisy_rose : ℝ := speed_daisy_rose * time_daisy_rose
  let distance_rose_poppy : ℝ := speed_rose_poppy * time_rose_poppy
  distance_daisy_rose = distance_rose_poppy + distance_difference ∧
  speed_daisy_rose = 6.5

theorem bee_speed_solution :
  bee_speed_problem 10 6 8 3 := by
  sorry

#check bee_speed_solution

end bee_speed_solution_l2878_287880


namespace min_distance_between_curves_l2878_287884

open Real

/-- The minimum distance between two points on different curves -/
theorem min_distance_between_curves (a : ℝ) : 
  ∃ (x₁ x₂ : ℝ), 
    a = 3 * x₁ + 3 ∧ 
    a = 2 * x₂ + log x₂ ∧
    ∀ (y₁ y₂ : ℝ), 
      (a = 3 * y₁ + 3 ∧ a = 2 * y₂ + log y₂) → 
      |x₂ - x₁| ≤ |y₂ - y₁| ∧
      |x₂ - x₁| = 4/3 := by
  sorry

end min_distance_between_curves_l2878_287884


namespace union_of_A_and_B_l2878_287814

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

-- Define the union of A and B
def AUnionB : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = AUnionB := by sorry

end union_of_A_and_B_l2878_287814


namespace equation_solution_l2878_287813

theorem equation_solution : ∃ y : ℝ, y = (18 : ℝ) / 4 ∧ (8 * y^2 + 50 * y + 3) / (4 * y + 21) = 2 * y + 1 := by
  sorry

end equation_solution_l2878_287813


namespace max_fourth_term_arithmetic_seq_l2878_287862

/-- Given a sequence of six positive integers in arithmetic progression with a sum of 90,
    the maximum possible value of the fourth term is 17. -/
theorem max_fourth_term_arithmetic_seq : ∀ (a d : ℕ),
  a > 0 → d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 90 →
  a + 3*d ≤ 17 :=
by sorry

end max_fourth_term_arithmetic_seq_l2878_287862


namespace verbal_to_inequality_l2878_287881

/-- The inequality that represents "twice x plus 8 is less than five times x" -/
def twice_x_plus_8_less_than_5x (x : ℝ) : Prop :=
  2 * x + 8 < 5 * x

theorem verbal_to_inequality :
  ∀ x : ℝ, twice_x_plus_8_less_than_5x x ↔ (2 * x + 8 < 5 * x) :=
by
  sorry

#check verbal_to_inequality

end verbal_to_inequality_l2878_287881


namespace gcd_5280_12155_l2878_287843

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end gcd_5280_12155_l2878_287843


namespace arithmetic_sequence_property_l2878_287846

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {a_n} where a_3 + a_11 = 40, 
    the value of a_6 - a_7 + a_8 is equal to 20 -/
theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 11 = 40) : 
  a 6 - a 7 + a 8 = 20 := by
sorry

end arithmetic_sequence_property_l2878_287846


namespace area_lower_bound_l2878_287821

/-- A plane convex polygon with given projections -/
structure ConvexPolygon where
  /-- Projection onto OX axis -/
  proj_ox : ℝ
  /-- Projection onto bisector of 1st and 3rd coordinate angles -/
  proj_bisector13 : ℝ
  /-- Projection onto OY axis -/
  proj_oy : ℝ
  /-- Projection onto bisector of 2nd and 4th coordinate angles -/
  proj_bisector24 : ℝ
  /-- Area of the polygon -/
  area : ℝ
  /-- Convexity property (simplified) -/
  convex : True

/-- Theorem: The area of a convex polygon with given projections is at least 10 -/
theorem area_lower_bound (p : ConvexPolygon)
  (h1 : p.proj_ox = 4)
  (h2 : p.proj_bisector13 = 3 * Real.sqrt 2)
  (h3 : p.proj_oy = 5)
  (h4 : p.proj_bisector24 = 4 * Real.sqrt 2) :
  p.area ≥ 10 := by
  sorry


end area_lower_bound_l2878_287821


namespace apples_per_basket_l2878_287841

theorem apples_per_basket (total_baskets : ℕ) (total_apples : ℕ) (h1 : total_baskets = 37) (h2 : total_apples = 629) :
  total_apples / total_baskets = 17 := by
  sorry

end apples_per_basket_l2878_287841


namespace product_of_three_consecutive_integers_divisible_by_six_l2878_287860

theorem product_of_three_consecutive_integers_divisible_by_six (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) = 6 * k := by
sorry

end product_of_three_consecutive_integers_divisible_by_six_l2878_287860


namespace cubic_kilometer_to_cubic_meters_l2878_287800

/-- Given that one kilometer equals 1000 meters, prove that one cubic kilometer equals 1,000,000,000 cubic meters. -/
theorem cubic_kilometer_to_cubic_meters :
  (1 : ℝ) * (kilometer ^ 3) = 1000000000 * (meter ^ 3) :=
by
  sorry

end cubic_kilometer_to_cubic_meters_l2878_287800


namespace complex_power_36_l2878_287848

theorem complex_power_36 :
  (Complex.exp (160 * π / 180 * Complex.I))^36 = 1 :=
by sorry

end complex_power_36_l2878_287848


namespace inverse_proportion_comparison_l2878_287827

/-- Given two points A(-2, y₁) and B(-1, y₂) on the inverse proportion function y = 2/x,
    prove that y₁ > y₂ -/
theorem inverse_proportion_comparison :
  ∀ y₁ y₂ : ℝ,
  y₁ = 2 / (-2) →
  y₂ = 2 / (-1) →
  y₁ > y₂ := by
  sorry

end inverse_proportion_comparison_l2878_287827


namespace odell_kershaw_passing_l2878_287822

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- track radius in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (duration : ℝ) : ℕ :=
  sorry

theorem odell_kershaw_passing :
  let odell : Runner := { speed := 260, radius := 55, direction := 1 }
  let kershaw : Runner := { speed := 310, radius := 65, direction := -1 }
  passingCount odell kershaw 35 = 52 :=
sorry

end odell_kershaw_passing_l2878_287822


namespace valentines_given_to_children_l2878_287823

/-- The number of Valentines Mrs. Wong had initially -/
def initial_valentines : ℕ := 30

/-- The number of Valentines Mrs. Wong was left with -/
def remaining_valentines : ℕ := 22

/-- The number of Valentines Mrs. Wong gave to her children -/
def given_valentines : ℕ := initial_valentines - remaining_valentines

theorem valentines_given_to_children :
  given_valentines = 8 :=
by sorry

end valentines_given_to_children_l2878_287823


namespace cubic_function_property_l2878_287815

/-- Given a cubic function f(x) = ax³ + bx + 8 where f(-2) = 10, prove that f(2) = 6 -/
theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 8
  f (-2) = 10 → f 2 = 6 := by
sorry

end cubic_function_property_l2878_287815


namespace ratio_a_to_b_l2878_287861

-- Define an arithmetic sequence
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

-- Define our specific sequence
def our_sequence (s : ℕ → ℝ) (y : ℝ) : Prop :=
  s 0 = s 1 - y ∧ s 1 = y ∧ s 2 = s 1 + y ∧ s 3 = 3 * y

theorem ratio_a_to_b (s : ℕ → ℝ) (y : ℝ) :
  is_arithmetic_sequence s → our_sequence s y → s 0 / s 2 = 0 := by
  sorry


end ratio_a_to_b_l2878_287861


namespace quadratic_equality_l2878_287847

theorem quadratic_equality (a b c x : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∃ p q r : Fin 6, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (let f : Fin 6 → ℝ := λ i =>
      match i with
      | 0 => a*x^2 + b*x + c
      | 1 => a*x^2 + c*x + b
      | 2 => b*x^2 + c*x + a
      | 3 => b*x^2 + a*x + c
      | 4 => c*x^2 + a*x + b
      | 5 => c*x^2 + b*x + a
    f p = f q ∧ f q = f r)) →
  x = 1 := by
sorry

end quadratic_equality_l2878_287847


namespace no_super_sextalternado_smallest_sextalternado_l2878_287874

/-- Checks if a number has 8 digits --/
def has_eight_digits (n : ℕ) : Prop := 10000000 ≤ n ∧ n < 100000000

/-- Checks if consecutive digits have different parity --/
def has_alternating_parity (n : ℕ) : Prop :=
  ∀ i : ℕ, i < 7 → (n / 10^i % 2) ≠ (n / 10^(i+1) % 2)

/-- Defines a sextalternado number --/
def is_sextalternado (n : ℕ) : Prop :=
  has_eight_digits n ∧ n % 30 = 0 ∧ has_alternating_parity n

/-- Defines a super sextalternado number --/
def is_super_sextalternado (n : ℕ) : Prop :=
  is_sextalternado n ∧ n % 12 = 0

theorem no_super_sextalternado :
  ¬ ∃ n : ℕ, is_super_sextalternado n :=
sorry

theorem smallest_sextalternado :
  ∃ n : ℕ, is_sextalternado n ∧ ∀ m : ℕ, is_sextalternado m → n ≤ m ∧ n = 10101030 :=
sorry

end no_super_sextalternado_smallest_sextalternado_l2878_287874


namespace arithmetic_mean_of_fractions_l2878_287883

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((x^2 + a^2) / x^2 + (x^2 - a^2) / x^2) = 1 := by
  sorry

end arithmetic_mean_of_fractions_l2878_287883
