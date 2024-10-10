import Mathlib

namespace congruent_triangles_equal_perimeter_l509_50989

/-- Represents a triangle -/
structure Triangle where
  perimeter : ℝ

/-- Two triangles are congruent -/
def Congruent (t1 t2 : Triangle) : Prop := sorry

theorem congruent_triangles_equal_perimeter (t1 t2 : Triangle) 
  (h1 : Congruent t1 t2) (h2 : t1.perimeter = 5) : t2.perimeter = 5 := by
  sorry

end congruent_triangles_equal_perimeter_l509_50989


namespace katies_miles_l509_50965

/-- Proves that Katie's miles run is 10, given Adam's miles and the difference between their runs -/
theorem katies_miles (adam_miles : ℕ) (difference : ℕ) (h1 : adam_miles = 35) (h2 : difference = 25) :
  adam_miles - difference = 10 := by
  sorry

end katies_miles_l509_50965


namespace certain_number_threshold_l509_50907

theorem certain_number_threshold (k : ℤ) : 0.0010101 * (10 : ℝ)^(k : ℝ) > 10.101 → k ≥ 5 := by
  sorry

end certain_number_threshold_l509_50907


namespace circular_paper_pieces_for_square_border_l509_50981

theorem circular_paper_pieces_for_square_border (side_length : ℝ) (pieces_per_circle : ℕ) : 
  side_length = 10 → pieces_per_circle = 20 → (4 * side_length) / (2 * π) * pieces_per_circle = 40 := by
  sorry

#check circular_paper_pieces_for_square_border

end circular_paper_pieces_for_square_border_l509_50981


namespace fathers_age_l509_50948

theorem fathers_age (M F : ℕ) : 
  M = (2 : ℕ) * F / (5 : ℕ) →
  M + 6 = (F + 6) / (2 : ℕ) →
  F = 30 := by
sorry

end fathers_age_l509_50948


namespace fraction_addition_l509_50963

theorem fraction_addition : (1 : ℚ) / 420 + 19 / 35 = 229 / 420 := by
  sorry

end fraction_addition_l509_50963


namespace farm_chickens_l509_50922

/-- Represents the number of chickens on a farm -/
def num_chickens (total_legs total_animals : ℕ) : ℕ :=
  total_animals - (total_legs - 2 * total_animals) / 2

/-- Theorem stating that given the conditions of the farm, there are 5 chickens -/
theorem farm_chickens : num_chickens 38 12 = 5 := by
  sorry

end farm_chickens_l509_50922


namespace arithmetic_progression_theorem_l509_50912

/-- An arithmetic progression with a non-zero common difference -/
def ArithmeticProgression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- The condition that b_n is also an arithmetic progression -/
def BnIsArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d' : ℝ, d' ≠ 0 ∧ ∀ n, a (n + 1) * Real.cos (a (n + 1)) = a n * Real.cos (a n) + d'

/-- The given equation holds for all n -/
def EquationHolds (a : ℕ → ℝ) : Prop :=
  ∀ n, Real.sin (2 * a n) + Real.cos (a (n + 1)) = 0

theorem arithmetic_progression_theorem (a : ℕ → ℝ) (d : ℝ) :
  ArithmeticProgression a d →
  BnIsArithmeticProgression a →
  EquationHolds a →
  (∃ m k : ℤ, k ≠ 0 ∧ 
    ((a 1 = -π / 6 + 2 * π * ↑m ∧ d = 2 * π * ↑k) ∨
     (a 1 = -5 * π / 6 + 2 * π * ↑m ∧ d = 2 * π * ↑k))) :=
by sorry

end arithmetic_progression_theorem_l509_50912


namespace factorization_equality_l509_50955

theorem factorization_equality (x y : ℝ) :
  3 * x^2 - x * y - y^2 = ((Real.sqrt 13 + 1) / 2 * x + y) * ((Real.sqrt 13 - 1) / 2 * x - y) := by
  sorry

end factorization_equality_l509_50955


namespace balls_sold_count_l509_50960

def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 72
def loss : ℕ := 5 * cost_price_per_ball

theorem balls_sold_count :
  ∃ n : ℕ, n * cost_price_per_ball - selling_price = loss ∧ n = 15 :=
by sorry

end balls_sold_count_l509_50960


namespace foldable_rectangle_short_side_l509_50970

/-- A rectangle with the property that when folded along its diagonal,
    it forms a trapezoid with three equal sides. -/
structure FoldableRectangle where
  long_side : ℝ
  short_side : ℝ
  long_side_positive : 0 < long_side
  short_side_positive : 0 < short_side
  long_side_longer : short_side ≤ long_side
  forms_equal_sided_trapezoid : True  -- This is a placeholder for the folding property

/-- The theorem stating that a rectangle with longer side 12 cm, when folded to form
    a trapezoid with three equal sides, has a shorter side of 4√3 cm. -/
theorem foldable_rectangle_short_side
  (rect : FoldableRectangle)
  (h_long : rect.long_side = 12) :
  rect.short_side = 4 * Real.sqrt 3 := by
  sorry

#check foldable_rectangle_short_side

end foldable_rectangle_short_side_l509_50970


namespace rational_cosine_values_l509_50916

theorem rational_cosine_values : 
  {k : ℚ | 0 ≤ k ∧ k ≤ 1/2 ∧ ∃ (q : ℚ), Real.cos (k * Real.pi) = q} = {0, 1/2, 1/3} := by sorry

end rational_cosine_values_l509_50916


namespace line_x_intercept_l509_50973

/-- Given a line passing through points (2, -2) and (6, 6), its x-intercept is 3 -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ), 
  (f 2 = -2) → 
  (f 6 = 6) → 
  (∀ x y : ℝ, f y - f x = (y - x) * ((6 - (-2)) / (6 - 2))) →
  (∃ x : ℝ, f x = 0 ∧ x = 3) := by
sorry

end line_x_intercept_l509_50973


namespace volume_maximized_at_one_l509_50993

/-- The volume function of the lidless square box -/
def V (x : ℝ) : ℝ := x * (6 - 2*x)^2

/-- The derivative of the volume function -/
def V' (x : ℝ) : ℝ := 12*x^2 - 48*x + 36

theorem volume_maximized_at_one :
  ∀ x ∈ Set.Ioo 0 3, V x ≤ V 1 :=
sorry

end volume_maximized_at_one_l509_50993


namespace find_a_l509_50911

theorem find_a : ∃ a : ℝ, 
  (∀ x : ℝ, (x^2 - 4*x + a) + |x - 3| ≤ 5) ∧
  (∀ x : ℝ, x > 3 → (x^2 - 4*x + a) + |x - 3| > 5) →
  a = 2 := by
sorry

end find_a_l509_50911


namespace even_function_theorem_l509_50935

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of a function -/
def Domain (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x, x ∈ s ↔ ∃ y, f x = y

theorem even_function_theorem (a b : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + 3 * a + b
  IsEven f ∧ Domain f (Set.Icc (a - 1) (2 * a)) →
  a + b = 1/3 := by
  sorry

end even_function_theorem_l509_50935


namespace shaded_area_calculation_l509_50929

/-- The area of the region outside a circle of radius 2 and inside two circles of radius 4 
    that are internally tangent to the smaller circle at opposite ends of its diameter -/
theorem shaded_area_calculation : 
  ∃ (r₁ r₂ : ℝ) (A B : ℝ × ℝ),
    r₁ = 2 ∧ 
    r₂ = 4 ∧
    A.1^2 + A.2^2 = r₁^2 ∧
    B.1^2 + B.2^2 = r₁^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * r₁)^2 →
    let shaded_area := 
      2 * (π * r₂^2 / 6 - r₁ * (r₂^2 - r₁^2).sqrt / 2 - π * r₁^2 / 4)
    shaded_area = (20 / 3) * π - 8 * Real.sqrt 3 := by
  sorry

end shaded_area_calculation_l509_50929


namespace complex_fraction_pure_imaginary_l509_50954

/-- Given that a is a real number and i is the imaginary unit, 
    if (a+3i)/(1-2i) is a pure imaginary number, then a = 6 -/
theorem complex_fraction_pure_imaginary (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (∃ b : ℝ, (a + 3 * Complex.I) / (1 - 2 * Complex.I) = b * Complex.I) →
  a = 6 :=
by sorry

end complex_fraction_pure_imaginary_l509_50954


namespace greatest_odd_factors_below_100_l509_50942

/-- A number has an odd number of positive factors if and only if it is a perfect square. -/
def has_odd_factors (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The greatest whole number less than 100 that has an odd number of positive factors is 81. -/
theorem greatest_odd_factors_below_100 : 
  (∀ m : ℕ, m < 100 → has_odd_factors m → m ≤ 81) ∧ has_odd_factors 81 ∧ 81 < 100 := by
  sorry

end greatest_odd_factors_below_100_l509_50942


namespace regular_triangle_rotation_l509_50940

/-- The minimum angle of rotation (in degrees) for a regular triangle to coincide with itself. -/
def min_rotation_angle_regular_triangle : ℝ := 120

/-- Theorem stating that the minimum angle of rotation for a regular triangle to coincide with itself is 120 degrees. -/
theorem regular_triangle_rotation :
  min_rotation_angle_regular_triangle = 120 := by sorry

end regular_triangle_rotation_l509_50940


namespace simplify_fraction_l509_50956

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by sorry

end simplify_fraction_l509_50956


namespace log_sum_equal_one_power_mult_equal_l509_50925

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem for the first expression
theorem log_sum_equal_one : log10 2 + log10 5 = 1 := by sorry

-- Theorem for the second expression
theorem power_mult_equal : 4 * (-100)^4 = 400000000 := by sorry

end log_sum_equal_one_power_mult_equal_l509_50925


namespace intersection_M_N_l509_50949

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 + x ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end intersection_M_N_l509_50949


namespace sequence_ratio_l509_50988

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that (a₂ - a₁) / b₂ = 1/2 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-1 : ℝ) - a₁ = a₁ - a₂) →  -- arithmetic sequence condition
  (a₂ - (-4 : ℝ) = a₁ - a₂) →  -- arithmetic sequence condition
  (b₁ / (-1 : ℝ) = b₂ / b₁) →  -- geometric sequence condition
  (b₂ / b₁ = b₃ / b₂) →        -- geometric sequence condition
  (b₃ / b₂ = (-4 : ℝ) / b₃) →  -- geometric sequence condition
  (a₂ - a₁) / b₂ = (1 : ℝ) / 2 := by
sorry

end sequence_ratio_l509_50988


namespace fourth_side_length_l509_50914

/-- A quadrilateral inscribed in a circle with radius 150√2, where three sides have lengths 150, 150, and 150√3 -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the first side -/
  side1 : ℝ
  /-- The length of the second side -/
  side2 : ℝ
  /-- The length of the third side -/
  side3 : ℝ
  /-- The length of the fourth side -/
  side4 : ℝ
  /-- The radius is 150√2 -/
  radius_eq : radius = 150 * Real.sqrt 2
  /-- The first side has length 150 -/
  side1_eq : side1 = 150
  /-- The second side has length 150 -/
  side2_eq : side2 = 150
  /-- The third side has length 150√3 -/
  side3_eq : side3 = 150 * Real.sqrt 3

/-- The theorem stating that the fourth side has length 150√7 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.side4 = 150 * Real.sqrt 7 := by
  sorry

end fourth_side_length_l509_50914


namespace cube_root_two_solves_equation_l509_50943

theorem cube_root_two_solves_equation :
  let x : ℝ := Real.rpow 2 (1/3)
  (x + 1)^3 = 1 / (x - 1) ∧ x ≠ 1 :=
by sorry

end cube_root_two_solves_equation_l509_50943


namespace cube_greater_iff_l509_50998

theorem cube_greater_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end cube_greater_iff_l509_50998


namespace xy_value_l509_50991

theorem xy_value (x y : ℝ) (h : x / 2 + 2 * y - 2 = Real.log x + Real.log y) : 
  x ^ y = Real.sqrt 2 := by
  sorry

end xy_value_l509_50991


namespace zhang_qiujian_suanjing_problem_l509_50903

theorem zhang_qiujian_suanjing_problem (a : ℕ → ℚ) :
  (∀ i j, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
  a 1 + a 2 + a 3 = 4 →                         -- sum of first 3 terms
  a 8 + a 9 + a 10 = 3 →                        -- sum of last 3 terms
  a 5 + a 6 = 7/3 :=                            -- sum of 5th and 6th terms
by sorry

end zhang_qiujian_suanjing_problem_l509_50903


namespace no_nonzero_rational_solution_l509_50913

theorem no_nonzero_rational_solution :
  ∀ (x y z : ℚ), x^3 + 3*y^3 + 9*z^3 = 9*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end no_nonzero_rational_solution_l509_50913


namespace games_missed_l509_50918

theorem games_missed (planned_this_month planned_last_month attended : ℕ) 
  (h1 : planned_this_month = 11)
  (h2 : planned_last_month = 17)
  (h3 : attended = 12) :
  planned_this_month + planned_last_month - attended = 16 := by
  sorry

end games_missed_l509_50918


namespace linear_function_property_l509_50947

/-- A linear function is a function of the form f(x) = mx + b for some constants m and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) (h_linear : LinearFunction g) 
    (h_diff : g 4 - g 1 = 9) : g 10 - g 1 = 27 := by
  sorry

end linear_function_property_l509_50947


namespace divisible_by_eleven_l509_50967

theorem divisible_by_eleven (n : ℤ) : (18888 - n) % 11 = 0 → n = 7 := by
  sorry

end divisible_by_eleven_l509_50967


namespace correct_pizza_dough_amounts_l509_50992

/-- Calculates the required amounts of milk and water for a given amount of flour in Luca's pizza dough recipe. -/
def pizzaDoughCalculation (flourAmount : ℚ) : ℚ × ℚ :=
  let milkToFlourRatio : ℚ := 80 / 400
  let waterToMilkRatio : ℚ := 1 / 2
  let milkAmount : ℚ := flourAmount * milkToFlourRatio
  let waterAmount : ℚ := milkAmount * waterToMilkRatio
  (milkAmount, waterAmount)

/-- Theorem stating the correct amounts of milk and water for 1200 mL of flour. -/
theorem correct_pizza_dough_amounts :
  pizzaDoughCalculation 1200 = (240, 120) := by
  sorry

#eval pizzaDoughCalculation 1200

end correct_pizza_dough_amounts_l509_50992


namespace value_multiplied_with_b_l509_50917

theorem value_multiplied_with_b (a b x : ℚ) : 
  a / b = 6 / 5 → 
  (5 * a + x * b) / (5 * a - x * b) = 5 →
  x = 4 := by
sorry

end value_multiplied_with_b_l509_50917


namespace inequality_solution_set_l509_50985

theorem inequality_solution_set :
  let S := {x : ℝ | (x + 5) * (3 - 2*x) ≤ 6}
  S = {x : ℝ | -9 ≤ x ∧ x ≤ 1/2} := by
sorry

end inequality_solution_set_l509_50985


namespace arithmetic_sequence_and_general_formula_l509_50945

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => 2 * sequence_a n / (sequence_a n + 2)

def sequence_b (n : ℕ) : ℚ := 1 / sequence_a n

theorem arithmetic_sequence_and_general_formula :
  (∀ n : ℕ, ∃ d : ℚ, sequence_b (n + 1) - sequence_b n = d) ∧
  (∀ n : ℕ, sequence_a n = 2 / (n + 1)) :=
sorry

end arithmetic_sequence_and_general_formula_l509_50945


namespace quadratic_rewrite_l509_50959

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x : ℝ, 4 * x^2 + 20 * x - 24 = (d * x + e)^2 + f) → d * e = 10 := by
  sorry

end quadratic_rewrite_l509_50959


namespace prob_three_even_in_five_rolls_l509_50994

/-- A fair 10-sided die -/
def TenSidedDie : Type := Fin 10

/-- The probability of rolling an even number on a 10-sided die -/
def probEven : ℚ := 1 / 2

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of dice we want to show even numbers -/
def numEven : ℕ := 3

/-- The probability of rolling exactly three even numbers when five fair 10-sided dice are rolled -/
theorem prob_three_even_in_five_rolls : 
  (numDice.choose numEven : ℚ) * probEven ^ numEven * (1 - probEven) ^ (numDice - numEven) = 5 / 16 := by
  sorry

end prob_three_even_in_five_rolls_l509_50994


namespace mark_soup_donation_l509_50972

theorem mark_soup_donation (shelters : ℕ) (people_per_shelter : ℕ) (cans_per_person : ℕ)
  (h1 : shelters = 6)
  (h2 : people_per_shelter = 30)
  (h3 : cans_per_person = 10) :
  shelters * people_per_shelter * cans_per_person = 1800 :=
by sorry

end mark_soup_donation_l509_50972


namespace sqrt_tan_domain_l509_50962

theorem sqrt_tan_domain (x : ℝ) :
  ∃ (y : ℝ), y = Real.sqrt (Real.tan x) ↔ ∃ (k : ℤ), k * Real.pi ≤ x ∧ x < k * Real.pi + Real.pi / 2 :=
by sorry

end sqrt_tan_domain_l509_50962


namespace fourth_student_guess_is_525_l509_50996

/-- Represents the number of jellybeans guessed by each student -/
def jellybean_guess : Fin 4 → ℕ
  | 0 => 100  -- First student's guess
  | 1 => 8 * jellybean_guess 0  -- Second student's guess
  | 2 => jellybean_guess 1 - 200  -- Third student's guess
  | 3 => (jellybean_guess 0 + jellybean_guess 1 + jellybean_guess 2) / 3 + 25  -- Fourth student's guess

/-- Theorem stating that the fourth student's guess is 525 -/
theorem fourth_student_guess_is_525 : jellybean_guess 3 = 525 := by
  sorry

end fourth_student_guess_is_525_l509_50996


namespace sandy_parentheses_problem_l509_50982

theorem sandy_parentheses_problem (p q r s : ℤ) (h1 : p = 2) (h2 : q = 4) (h3 : r = 6) (h4 : s = 8) :
  ∃ t : ℤ, p + (q - (r + (s - t))) = p + q - r + s - 10 ∧ t = 8 := by
sorry

end sandy_parentheses_problem_l509_50982


namespace composition_equation_solution_l509_50941

/-- Given two functions f and g, and a condition on their composition, prove the value of b. -/
theorem composition_equation_solution (f g : ℝ → ℝ) (b : ℝ) 
  (hf : ∀ x, f x = 3 * x - 2)
  (hg : ∀ x, g x = 7 - 2 * x)
  (h_comp : g (f b) = 1) : 
  b = 5 / 3 := by
sorry

end composition_equation_solution_l509_50941


namespace tiles_difference_8th_7th_l509_50958

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := n^2

/-- The theorem stating the difference in tiles between the 8th and 7th squares -/
theorem tiles_difference_8th_7th : 
  tiles_in_square 8 - tiles_in_square 7 = 15 := by
  sorry

end tiles_difference_8th_7th_l509_50958


namespace basket_weight_l509_50995

/-- Given a basket of persimmons, prove the weight of the empty basket. -/
theorem basket_weight (total_weight half_weight : ℝ) 
  (h1 : total_weight = 62)
  (h2 : half_weight = 34) : 
  ∃ (basket_weight persimmons_weight : ℝ),
    basket_weight + persimmons_weight = total_weight ∧ 
    basket_weight + persimmons_weight / 2 = half_weight ∧
    basket_weight = 6 := by
  sorry

end basket_weight_l509_50995


namespace cassies_nail_cutting_l509_50946

/-- The number of nails Cassie needs to cut -/
def total_nails (num_dogs : ℕ) (num_parrots : ℕ) (dog_feet : ℕ) (dog_nails_per_foot : ℕ) 
                (parrot_legs : ℕ) (parrot_claws_per_leg : ℕ) (extra_claw : ℕ) : ℕ :=
  num_dogs * dog_feet * dog_nails_per_foot + 
  (num_parrots - 1) * parrot_legs * parrot_claws_per_leg + 
  (parrot_legs * parrot_claws_per_leg + extra_claw)

/-- Theorem stating the total number of nails Cassie needs to cut -/
theorem cassies_nail_cutting : 
  total_nails 4 8 4 4 2 3 1 = 113 := by
  sorry

end cassies_nail_cutting_l509_50946


namespace range_of_a_l509_50930

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^3 + 6 * a * x^2 - 1

def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x - 1

theorem range_of_a (a : ℝ) (h1 : a > 0) :
  (∃ x₁ > 0, ∃ x₂, f a x₁ ≥ g a x₂) → a ≥ 1/2 := by
  sorry

end range_of_a_l509_50930


namespace geometric_sequence_sum_l509_50928

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sequence_sum_l509_50928


namespace y_derivative_l509_50931

noncomputable def y (x : ℝ) : ℝ := -2 * Real.exp x * Real.sin x

theorem y_derivative (x : ℝ) : 
  deriv y x = -2 * Real.exp x * (Real.sin x + Real.cos x) := by sorry

end y_derivative_l509_50931


namespace M_equals_N_l509_50910

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | 1/x < 1}

-- Theorem statement
theorem M_equals_N : M = N := by
  sorry

end M_equals_N_l509_50910


namespace ellipse_intersection_dot_product_range_l509_50923

def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem ellipse_intersection_dot_product_range :
  ∀ k : ℝ, k ≠ 0 →
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    -4 ≤ dot_product x₁ y₁ x₂ y₂ ∧ dot_product x₁ y₁ x₂ y₂ < 13/4 :=
sorry

end ellipse_intersection_dot_product_range_l509_50923


namespace polygon_sides_count_l509_50979

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (2 * 360 : ℝ) = (n - 2 : ℝ) * 180 → n = 6 := by
  sorry

end polygon_sides_count_l509_50979


namespace complex_modulus_problem_l509_50902

theorem complex_modulus_problem (z : ℂ) (h : (z - 2*Complex.I)*(1 - Complex.I) = -2) :
  Complex.abs z = Real.sqrt 2 := by sorry

end complex_modulus_problem_l509_50902


namespace inequality_proof_l509_50957

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), x = 2*k ∧ y = k ∧ z = k) :=
by sorry

end inequality_proof_l509_50957


namespace min_tan_product_l509_50936

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and satisfying bsinC + csinB = 4asinBsinC, the minimum value of tanAtanBtanC is (12 + 7√3) / 3 -/
theorem min_tan_product (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C →
  (∀ A' B' C' : ℝ,
    0 < A' ∧ A' < π/2 →
    0 < B' ∧ B' < π/2 →
    0 < C' ∧ C' < π/2 →
    A' + B' + C' = π →
    Real.tan A' * Real.tan B' * Real.tan C' ≥ (12 + 7 * Real.sqrt 3) / 3) ∧
  (∃ A' B' C' : ℝ,
    0 < A' ∧ A' < π/2 ∧
    0 < B' ∧ B' < π/2 ∧
    0 < C' ∧ C' < π/2 ∧
    A' + B' + C' = π ∧
    Real.tan A' * Real.tan B' * Real.tan C' = (12 + 7 * Real.sqrt 3) / 3) :=
by sorry

end min_tan_product_l509_50936


namespace quadratic_inequality_l509_50990

theorem quadratic_inequality (x : ℝ) : 
  10 * x^2 - 2 * x - 3 < 0 ↔ (1 - Real.sqrt 31) / 10 < x ∧ x < (1 + Real.sqrt 31) / 10 := by
  sorry

end quadratic_inequality_l509_50990


namespace exactly_one_real_solution_l509_50971

theorem exactly_one_real_solution :
  ∃! x : ℝ, ((-4 * (x - 3)^2 : ℝ) ≥ 0) := by sorry

end exactly_one_real_solution_l509_50971


namespace cost_price_calculation_l509_50969

/-- Proves that the cost price of an article is 480, given the selling price and profit percentage -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 595.2 → 
  profit_percentage = 24 → 
  ∃ (cost_price : ℝ), 
    cost_price = 480 ∧ 
    selling_price = cost_price * (1 + profit_percentage / 100) := by
  sorry

end cost_price_calculation_l509_50969


namespace cube_of_4_minus_3i_l509_50974

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem cube_of_4_minus_3i :
  (4 - 3 * i) ^ 3 = -44 - 117 * i :=
by sorry

end cube_of_4_minus_3i_l509_50974


namespace tan_problem_l509_50921

noncomputable def α : ℝ := Real.arctan 3

theorem tan_problem (h : Real.tan (π - α) = -3) :
  (Real.tan α = 3) ∧
  ((Real.sin (π - α) - Real.cos (π + α) - Real.sin (2*π - α) + Real.cos (-α)) /
   (Real.sin (π/2 - α) + Real.cos (3*π/2 - α)) = -4) :=
by sorry

end tan_problem_l509_50921


namespace rhombus_area_calculation_l509_50901

/-- Represents a rhombus -/
structure Rhombus where
  side_length : ℝ
  area : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  ABCD : Rhombus
  BAFC : Rhombus
  AF_parallel_BD : Prop

/-- Main theorem -/
theorem rhombus_area_calculation (setup : ProblemSetup) 
  (h1 : setup.ABCD.side_length = 13)
  (h2 : setup.BAFC.area = 65)
  : setup.ABCD.area = 120 := by
  sorry

end rhombus_area_calculation_l509_50901


namespace expand_product_l509_50900

theorem expand_product (x : ℝ) : (x + 3) * (x - 4) * (x + 1) = x^3 - 13*x - 12 := by
  sorry

end expand_product_l509_50900


namespace line_slope_l509_50905

/-- The slope of the line given by the equation x/4 + y/3 = 2 is -3/4 -/
theorem line_slope (x y : ℝ) : (x / 4 + y / 3 = 2) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) := by
  sorry

end line_slope_l509_50905


namespace intersection_of_A_and_B_l509_50909

def A : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}
def B : Set (ℝ × ℝ) := {p | 2 ≤ p.1 ∧ p.1 ≤ 3 ∧ 1 ≤ p.2 ∧ p.2 ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {(2, 1)} := by
  sorry

end intersection_of_A_and_B_l509_50909


namespace hyperbola_real_axis_length_l509_50938

theorem hyperbola_real_axis_length
  (p : ℝ)
  (a b : ℝ)
  (h_p_pos : p > 0)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_directrix_tangent : 3 + p / 2 = 15)
  (h_asymptote : b / a = Real.sqrt 3)
  (h_focus : a^2 + b^2 = 144) :
  2 * a = 12 := by
  sorry

end hyperbola_real_axis_length_l509_50938


namespace regular_17gon_symmetry_sum_l509_50933

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  n_pos : 0 < n

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle (in degrees) for rotational symmetry -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_17gon_symmetry_sum :
  ∀ (p : RegularPolygon 17),
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by
  sorry

end regular_17gon_symmetry_sum_l509_50933


namespace least_integer_with_12_factors_l509_50961

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Theorem: 72 is the least positive integer with exactly 12 positive factors -/
theorem least_integer_with_12_factors :
  (∀ m : ℕ+, m < 72 → num_factors m ≠ 12) ∧ num_factors 72 = 12 := by
  sorry

end least_integer_with_12_factors_l509_50961


namespace ball_color_probability_l509_50934

def num_balls : ℕ := 8
def num_colors : ℕ := 2

theorem ball_color_probability :
  let p : ℚ := 1 / 2  -- probability of each color
  let n : ℕ := num_balls
  let k : ℕ := n / 2  -- number of balls of each color
  (n.choose k) * p^n = 35 / 128 := by
  sorry

end ball_color_probability_l509_50934


namespace vector_sum_length_l509_50939

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_length (a b : ℝ × ℝ) : 
  angle_between a b = π / 3 →
  a = (3, -4) →
  Real.sqrt ((a.1)^2 + (a.2)^2) = 2 →
  Real.sqrt (((a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2)) = Real.sqrt 61 := by
  sorry

end vector_sum_length_l509_50939


namespace margo_walk_distance_l509_50978

/-- Calculates the total distance of a round trip given the times for each leg and the average speed -/
def round_trip_distance (outbound_time inbound_time : ℚ) (average_speed : ℚ) : ℚ :=
  let total_time := outbound_time + inbound_time
  average_speed * (total_time / 60)

/-- Proves that given the specific conditions of Margo's walk, the total distance is 2 miles -/
theorem margo_walk_distance :
  round_trip_distance (15 : ℚ) (25 : ℚ) (3 : ℚ) = 2 := by
  sorry

end margo_walk_distance_l509_50978


namespace evaluate_expression_l509_50906

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 1) :
  z * (z - 4 * x) = -15 := by
  sorry

end evaluate_expression_l509_50906


namespace systematic_sampling_result_l509_50980

def systematic_sampling (total : ℕ) (sample_size : ℕ) (interval_start : ℕ) (interval_end : ℕ) : ℕ :=
  let sampling_interval := total / sample_size
  let interval_size := interval_end - interval_start + 1
  interval_size / sampling_interval

theorem systematic_sampling_result :
  systematic_sampling 420 21 281 420 = 7 := by
  sorry

end systematic_sampling_result_l509_50980


namespace smallest_n_cube_and_square_l509_50944

theorem smallest_n_cube_and_square : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (a : ℕ), 4 * n = a^3) ∧ 
  (∃ (b : ℕ), 5 * n = b^2) ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (c : ℕ), 4 * m = c^3) → 
    (∃ (d : ℕ), 5 * m = d^2) → 
    m ≥ n) ∧
  n = 400 :=
sorry

end smallest_n_cube_and_square_l509_50944


namespace epidemic_test_analysis_l509_50908

/-- Represents a class of students with their test scores and statistics -/
structure ClassData where
  scores : List Nat
  frequency_table : List (Nat × Nat)
  mean : Nat
  mode : Nat
  median : Nat
  variance : Float

/-- The data for the entire school -/
structure SchoolData where
  total_students : Nat
  class_a : ClassData
  class_b : ClassData

/-- Definition of excellent performance -/
def excellent_score : Nat := 90

/-- The given school data -/
def school_data : SchoolData := {
  total_students := 600,
  class_a := {
    scores := [78, 83, 89, 97, 98, 85, 100, 94, 87, 90, 93, 92, 99, 95, 100],
    frequency_table := [(1, 75), (1, 80), (3, 85), (4, 90), (6, 95)],
    mean := 92,
    mode := 100,
    median := 93,
    variance := 41.07
  },
  class_b := {
    scores := [91, 92, 94, 90, 93],
    frequency_table := [(1, 75), (2, 80), (3, 85), (5, 90), (4, 95)],
    mean := 90,
    mode := 87,
    median := 91,
    variance := 50.2
  }
}

theorem epidemic_test_analysis (data : SchoolData := school_data) :
  (data.class_a.mode = 100) ∧
  (data.class_b.median = 91) ∧
  (((data.class_a.frequency_table.filter (λ x => x.2 ≥ 90)).map (λ x => x.1)).sum +
   ((data.class_b.frequency_table.filter (λ x => x.2 ≥ 90)).map (λ x => x.1)).sum) * 20 = 380 ∧
  (data.class_a.mean > data.class_b.mean ∧ data.class_a.variance < data.class_b.variance) := by
  sorry

end epidemic_test_analysis_l509_50908


namespace calvins_roaches_l509_50966

theorem calvins_roaches (total insects : ℕ) (scorpions : ℕ) (roaches crickets caterpillars : ℕ) : 
  insects = 27 →
  scorpions = 3 →
  crickets = roaches / 2 →
  caterpillars = 2 * scorpions →
  insects = roaches + scorpions + crickets + caterpillars →
  roaches = 12 := by
sorry

end calvins_roaches_l509_50966


namespace greatest_integer_with_gcd_six_gcd_138_18_is_6_exists_no_greater_main_result_l509_50983

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
by sorry

theorem gcd_138_18_is_6 : Nat.gcd 138 18 = 6 :=
by sorry

theorem exists_no_greater : ¬∃ m : ℕ, 138 < m ∧ m < 150 ∧ Nat.gcd m 18 = 6 :=
by sorry

theorem main_result : ∃! n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ n :=
by sorry

end greatest_integer_with_gcd_six_gcd_138_18_is_6_exists_no_greater_main_result_l509_50983


namespace A_union_B_eq_real_l509_50968

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem A_union_B_eq_real : A ∪ B = Set.univ := by sorry

end A_union_B_eq_real_l509_50968


namespace li_fang_outfits_l509_50987

/-- The number of unique outfit combinations given a set of shirts, skirts, and dresses -/
def outfit_combinations (num_shirts num_skirts num_dresses : ℕ) : ℕ :=
  num_shirts * num_skirts + num_dresses

/-- Theorem: Given 4 shirts, 3 skirts, and 2 dresses, the total number of unique outfit combinations is 14 -/
theorem li_fang_outfits : outfit_combinations 4 3 2 = 14 := by
  sorry

end li_fang_outfits_l509_50987


namespace quadratic_roots_sum_l509_50986

theorem quadratic_roots_sum (a b : ℝ) : 
  (a^2 + a - 2024 = 0) → 
  (b^2 + b - 2024 = 0) → 
  (a^2 + 2*a + b = 2023) := by
  sorry

end quadratic_roots_sum_l509_50986


namespace largest_n_satisfying_inequality_l509_50977

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), (n ≤ 6 ∧ ((1 : ℚ) / 5 + (n : ℚ) / 8 + 1 < 2)) ∧
  ∀ (m : ℕ), m > 6 → ((1 : ℚ) / 5 + (m : ℚ) / 8 + 1 ≥ 2) :=
by sorry

end largest_n_satisfying_inequality_l509_50977


namespace sprint_distance_l509_50920

def sprint_problem (speed : ℝ) (time : ℝ) : Prop :=
  speed = 6 ∧ time = 4 → speed * time = 24

theorem sprint_distance : sprint_problem 6 4 := by
  sorry

end sprint_distance_l509_50920


namespace number_of_nickels_l509_50976

def pennies : ℕ := 123
def dimes : ℕ := 35
def quarters : ℕ := 26
def family_members : ℕ := 5
def ice_cream_cost_per_member : ℚ := 3
def leftover_cents : ℕ := 48

def total_ice_cream_cost : ℚ := family_members * ice_cream_cost_per_member

def total_without_nickels : ℚ := 
  (pennies : ℚ) / 100 + (dimes : ℚ) / 10 + (quarters : ℚ) / 4

theorem number_of_nickels : 
  ∃ (n : ℕ), total_without_nickels + (n : ℚ) / 20 = total_ice_cream_cost + (leftover_cents : ℚ) / 100 ∧ n = 85 := by
  sorry

end number_of_nickels_l509_50976


namespace mathematician_paths_l509_50919

/-- Represents the number of rows in the diagram --/
def num_rows : ℕ := 13

/-- Represents whether the diagram is symmetric --/
def is_symmetric : Prop := true

/-- Represents that each move can be either down-left or down-right --/
def two_move_options : Prop := true

/-- The number of paths spelling "MATHEMATICIAN" in the diagram --/
def num_paths : ℕ := 2^num_rows - 1

theorem mathematician_paths :
  is_symmetric ∧ two_move_options → num_paths = 2^num_rows - 1 := by
  sorry

end mathematician_paths_l509_50919


namespace total_energy_consumption_l509_50926

/-- Calculate total electric energy consumption for given appliances over 30 days -/
theorem total_energy_consumption
  (fan_power : Real) (fan_hours : Real)
  (computer_power : Real) (computer_hours : Real)
  (ac_power : Real) (ac_hours : Real)
  (h : fan_power = 75 ∧ fan_hours = 8 ∧
       computer_power = 100 ∧ computer_hours = 5 ∧
       ac_power = 1500 ∧ ac_hours = 3) :
  (fan_power / 1000 * fan_hours +
   computer_power / 1000 * computer_hours +
   ac_power / 1000 * ac_hours) * 30 = 168 := by
sorry

end total_energy_consumption_l509_50926


namespace specific_building_occupancy_l509_50984

/-- Represents the building structure and occupancy --/
structure Building where
  floors : Nat
  first_floor_apartments : Nat
  common_difference : Nat
  one_bedroom_occupancy : Nat
  two_bedroom_occupancy : Nat
  three_bedroom_occupancy : Nat

/-- Calculates the total number of people in the building --/
def total_occupancy (b : Building) : Nat :=
  let last_floor_apartments := b.first_floor_apartments + (b.floors - 1) * b.common_difference
  let total_apartments := (b.floors * (b.first_floor_apartments + last_floor_apartments)) / 2
  let apartments_per_type := total_apartments / 3
  apartments_per_type * (b.one_bedroom_occupancy + b.two_bedroom_occupancy + b.three_bedroom_occupancy)

/-- Theorem stating the total occupancy of the specific building --/
theorem specific_building_occupancy :
  let b : Building := {
    floors := 25,
    first_floor_apartments := 3,
    common_difference := 2,
    one_bedroom_occupancy := 2,
    two_bedroom_occupancy := 4,
    three_bedroom_occupancy := 5
  }
  total_occupancy b = 2475 := by
  sorry

end specific_building_occupancy_l509_50984


namespace triangle_longest_side_l509_50927

/-- Given a triangle with side lengths 10, y+5, and 3y-2, and a perimeter of 50,
    prove that the longest side length is 25.75. -/
theorem triangle_longest_side (y : ℝ) : 
  10 + (y + 5) + (3 * y - 2) = 50 →
  max 10 (max (y + 5) (3 * y - 2)) = 25.75 := by
sorry

end triangle_longest_side_l509_50927


namespace multiply_121_54_l509_50951

theorem multiply_121_54 : 121 * 54 = 6534 := by
  sorry

end multiply_121_54_l509_50951


namespace dennis_teaching_years_l509_50937

theorem dennis_teaching_years 
  (total_years : ℕ) 
  (virginia_adrienne_diff : ℕ) 
  (dennis_virginia_diff : ℕ) 
  (h1 : total_years = 75)
  (h2 : virginia_adrienne_diff = 9)
  (h3 : dennis_virginia_diff = 9) :
  ∃ (adrienne virginia dennis : ℕ),
    adrienne + virginia + dennis = total_years ∧
    virginia = adrienne + virginia_adrienne_diff ∧
    dennis = virginia + dennis_virginia_diff ∧
    dennis = 34 := by
  sorry

end dennis_teaching_years_l509_50937


namespace multiply_three_point_five_by_zero_point_twenty_five_l509_50952

theorem multiply_three_point_five_by_zero_point_twenty_five : 3.5 * 0.25 = 0.875 := by
  sorry

end multiply_three_point_five_by_zero_point_twenty_five_l509_50952


namespace calvins_bug_collection_l509_50975

theorem calvins_bug_collection (roaches scorpions caterpillars crickets : ℕ) : 
  roaches = 12 →
  scorpions = 3 →
  caterpillars = 2 * scorpions →
  roaches + scorpions + caterpillars + crickets = 27 →
  crickets * 2 = roaches :=
by sorry

end calvins_bug_collection_l509_50975


namespace raghu_investment_l509_50904

theorem raghu_investment
  (vishal_investment : ℝ)
  (trishul_investment : ℝ)
  (raghu_investment : ℝ)
  (vishal_more_than_trishul : vishal_investment = 1.1 * trishul_investment)
  (trishul_less_than_raghu : trishul_investment = 0.9 * raghu_investment)
  (total_investment : vishal_investment + trishul_investment + raghu_investment = 6936) :
  raghu_investment = 2400 := by
sorry

end raghu_investment_l509_50904


namespace adjacent_complementary_angles_are_complementary_l509_50964

/-- Two angles are complementary if their sum is 90 degrees -/
def Complementary (α β : ℝ) : Prop := α + β = 90

/-- Two angles are adjacent if they share a common vertex and a common side,
    but have no common interior points -/
def Adjacent (α β : ℝ) : Prop := True  -- We simplify this for the statement

theorem adjacent_complementary_angles_are_complementary 
  (α β : ℝ) (h1 : Adjacent α β) (h2 : Complementary α β) : Complementary α β := by
  sorry

end adjacent_complementary_angles_are_complementary_l509_50964


namespace candy_sampling_percentage_l509_50953

theorem candy_sampling_percentage
  (caught_sampling : Real)
  (total_sampling : Real)
  (h1 : caught_sampling = 22)
  (h2 : total_sampling = 27.5)
  : total_sampling - caught_sampling = 5.5 := by
sorry

end candy_sampling_percentage_l509_50953


namespace sector_area_from_arc_and_angle_l509_50932

/-- Given an arc length of 28 cm and a central angle of 240°, 
    the area of the sector is 294/π cm² -/
theorem sector_area_from_arc_and_angle 
  (arc_length : ℝ) 
  (central_angle : ℝ) 
  (h1 : arc_length = 28) 
  (h2 : central_angle = 240) : 
  (1/2) * arc_length * (arc_length / (central_angle * (π / 180))) = 294 / π :=
by sorry

end sector_area_from_arc_and_angle_l509_50932


namespace same_solution_implies_c_value_l509_50950

theorem same_solution_implies_c_value (x : ℝ) (c : ℝ) :
  (3 * x + 9 = 6) ∧ (c * x - 15 = -5) → c = -10 := by
  sorry

end same_solution_implies_c_value_l509_50950


namespace salon_earnings_l509_50915

/-- Calculates the total earnings from hair salon services -/
def total_earnings (haircut_price style_price coloring_price treatment_price : ℕ)
                   (haircuts styles colorings treatments : ℕ) : ℕ :=
  haircut_price * haircuts +
  style_price * styles +
  coloring_price * colorings +
  treatment_price * treatments

/-- Theorem stating that given specific prices and quantities, the total earnings are 871 -/
theorem salon_earnings :
  total_earnings 12 25 35 50 8 5 10 6 = 871 := by
  sorry

end salon_earnings_l509_50915


namespace dans_marbles_l509_50924

/-- Represents the number of marbles Dan has -/
structure Marbles where
  violet : ℕ
  red : ℕ
  blue : ℕ

/-- Calculates the total number of marbles -/
def totalMarbles (m : Marbles) : ℕ := m.violet + m.red + m.blue

/-- Theorem stating the total number of marbles Dan has -/
theorem dans_marbles (x : ℕ) : 
  let initial := Marbles.mk 64 0 0
  let fromMary := Marbles.mk 0 14 0
  let fromJohn := Marbles.mk 0 0 x
  let final := Marbles.mk (initial.violet + fromMary.violet + fromJohn.violet)
                          (initial.red + fromMary.red + fromJohn.red)
                          (initial.blue + fromMary.blue + fromJohn.blue)
  totalMarbles final = 78 + x := by
  sorry

end dans_marbles_l509_50924


namespace abs_m_minus_n_equals_five_l509_50997

theorem abs_m_minus_n_equals_five (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : 
  |m - n| = 5 := by
sorry

end abs_m_minus_n_equals_five_l509_50997


namespace square_to_parallelogram_l509_50999

/-- Represents a plane figure --/
structure PlaneFigure where
  -- Add necessary fields

/-- Represents the oblique side drawing method --/
def obliqueSideDrawing (figure : PlaneFigure) : PlaneFigure :=
  sorry

/-- Predicate to check if a figure is a square --/
def isSquare (figure : PlaneFigure) : Prop :=
  sorry

/-- Predicate to check if a figure is a parallelogram --/
def isParallelogram (figure : PlaneFigure) : Prop :=
  sorry

/-- Theorem: The intuitive diagram of a square using oblique side drawing is a parallelogram --/
theorem square_to_parallelogram (figure : PlaneFigure) :
  isSquare figure → isParallelogram (obliqueSideDrawing figure) :=
by sorry

end square_to_parallelogram_l509_50999
