import Mathlib

namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2227_222754

theorem complex_fraction_sum (a b : ℂ) (h1 : a = 5 + 7*I) (h2 : b = 5 - 7*I) : 
  a / b + b / a = -23 / 37 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2227_222754


namespace NUMINAMATH_CALUDE_at_least_one_angle_not_greater_than_60_l2227_222776

-- Define a triangle as a triple of angles
def Triangle := (ℝ × ℝ × ℝ)

-- Define a predicate for a valid triangle (sum of angles is 180°)
def is_valid_triangle (t : Triangle) : Prop :=
  let (a, b, c) := t
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

-- Theorem statement
theorem at_least_one_angle_not_greater_than_60 (t : Triangle) 
  (h : is_valid_triangle t) : 
  ∃ θ, θ ∈ [t.1, t.2.1, t.2.2] ∧ θ ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_angle_not_greater_than_60_l2227_222776


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l2227_222716

theorem geometric_progression_ratio (b₁ q : ℕ) (h_sum : b₁ * q^2 + b₁ * q^4 + b₁ * q^6 = 7371 * 2^2016) :
  q = 1 ∨ q = 2 ∨ q = 3 ∨ q = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l2227_222716


namespace NUMINAMATH_CALUDE_water_remaining_l2227_222767

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 ∧ used = 11/4 → remaining = initial - used → remaining = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l2227_222767


namespace NUMINAMATH_CALUDE_total_sword_weight_l2227_222720

/-- The number of squads the Dark Lord has -/
def num_squads : ℕ := 10

/-- The number of orcs in each squad -/
def orcs_per_squad : ℕ := 8

/-- The weight of swords each orc carries (in pounds) -/
def sword_weight_per_orc : ℕ := 15

/-- Theorem stating the total weight of swords to be transported -/
theorem total_sword_weight : 
  num_squads * orcs_per_squad * sword_weight_per_orc = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_sword_weight_l2227_222720


namespace NUMINAMATH_CALUDE_symmetry_proof_l2227_222738

/-- Given two lines in the xy-plane, this function returns true if they are symmetric with respect to the line y = x -/
def are_symmetric_lines (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y ↔ line2 y x

/-- The original line: 2x + 3y + 6 = 0 -/
def original_line (x y : ℝ) : Prop :=
  2 * x + 3 * y + 6 = 0

/-- The symmetric line to be proved: 3x + 2y + 6 = 0 -/
def symmetric_line (x y : ℝ) : Prop :=
  3 * x + 2 * y + 6 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to y = x -/
theorem symmetry_proof : are_symmetric_lines original_line symmetric_line :=
sorry

end NUMINAMATH_CALUDE_symmetry_proof_l2227_222738


namespace NUMINAMATH_CALUDE_max_value_and_sum_l2227_222787

theorem max_value_and_sum (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 4050) : 
  let M := a*c + 3*b*c + 2*c*d + 8*d*e
  ∃ (a_M b_M c_M d_M e_M : ℝ),
    (∀ a' b' c' d' e' : ℝ, 
      0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 
      a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 4050 → 
      a'*c' + 3*b'*c' + 2*c'*d' + 8*d'*e' ≤ M) ∧
    M = 4050 * Real.sqrt 14 ∧
    M + a_M + b_M + c_M + d_M + e_M = 4050 * Real.sqrt 14 + 90 :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_sum_l2227_222787


namespace NUMINAMATH_CALUDE_unique_prime_sum_difference_l2227_222718

theorem unique_prime_sum_difference : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  (∃ a b : ℕ, Nat.Prime a ∧ Nat.Prime b ∧ p = a + b) ∧
  (∃ c d : ℕ, Nat.Prime c ∧ Nat.Prime d ∧ p = c - d) ∧
  p = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_difference_l2227_222718


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2227_222752

def num_math_books : ℕ := 4
def num_history_books : ℕ := 6
def total_books : ℕ := num_math_books + num_history_books

def arrange_books : ℕ := num_math_books * (num_math_books - 1) * Nat.factorial (total_books - 2)

theorem book_arrangement_count :
  arrange_books = 145152 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2227_222752


namespace NUMINAMATH_CALUDE_quadratic_value_l2227_222766

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_value (a b c : ℝ) :
  (∀ x, f a b c x ≤ 8) ∧  -- maximum value is 8
  (f a b c (-2) = 8) ∧    -- maximum occurs at x = -2
  (f a b c 1 = 4) →       -- passes through (1, 4)
  f a b c (-3) = 68/9 :=  -- value at x = -3 is 68/9
by sorry

end NUMINAMATH_CALUDE_quadratic_value_l2227_222766


namespace NUMINAMATH_CALUDE_sequence_property_l2227_222775

-- Define the sequence a_n
def a (n : ℕ) : ℚ := n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (a n)

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := (n^2 + n) / 2

-- Theorem statement
theorem sequence_property (n k : ℕ) (h1 : k > 2) :
  (∀ m : ℕ, S m = (m^2 + m) / 2) →
  (2 * b (n + 2) = b n + b (n + k)) →
  (k ≠ 4 ∧ k ≠ 10) ∧ (k = 6 ∨ k = 8) :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l2227_222775


namespace NUMINAMATH_CALUDE_equation_solution_l2227_222795

theorem equation_solution : ∃ n : ℤ, 5^3 - 7 = 6^2 + n ∧ n = 82 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2227_222795


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l2227_222791

theorem max_value_of_sum_of_squares (x y : ℝ) (h : 3 * x^2 + 2 * y^2 = 6 * x) :
  ∃ (max : ℝ), max = 1 ∧ ∀ (a b : ℝ), 3 * a^2 + 2 * b^2 = 6 * a → a^2 + b^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l2227_222791


namespace NUMINAMATH_CALUDE_domain_of_sqrt_sin_minus_cos_l2227_222755

open Real

theorem domain_of_sqrt_sin_minus_cos (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (sin x - cos x)) ↔
  (∃ k : ℤ, 2 * k * π + π / 4 ≤ x ∧ x ≤ 2 * k * π + 5 * π / 4) :=
by sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_sin_minus_cos_l2227_222755


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l2227_222746

theorem power_of_negative_cube (x : ℝ) : (-4 * x^3)^2 = 16 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l2227_222746


namespace NUMINAMATH_CALUDE_train_platform_pole_time_ratio_l2227_222734

/-- Given a train of length l traveling at constant velocity v,
    prove that the ratio of time taken to pass a platform of length 5l
    to the time taken to pass a pole is 6:1 -/
theorem train_platform_pole_time_ratio
  (l : ℝ) (v : ℝ) (t : ℝ) (h_v_pos : v > 0) (h_l_pos : l > 0) (h_t_pos : t > 0)
  (h_pole_time : l = v * t) :
  let platform_length := 5 * l
  let platform_time := (l + platform_length) / v
  platform_time / t = 6 := by
sorry

end NUMINAMATH_CALUDE_train_platform_pole_time_ratio_l2227_222734


namespace NUMINAMATH_CALUDE_domain_and_even_function_implies_a_eq_neg_one_l2227_222719

/-- A function is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem domain_and_even_function_implies_a_eq_neg_one
  (a : ℝ)
  (f : ℝ → ℝ)
  (h_domain : Set.Ioo (4*a - 3) (3 - 2*a^2) = Set.range f)
  (h_even : IsEven (fun x ↦ f (2*x - 3))) :
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_domain_and_even_function_implies_a_eq_neg_one_l2227_222719


namespace NUMINAMATH_CALUDE_no_real_roots_l2227_222741

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem no_real_roots :
  ¬∃ x : ℝ, 1 - log10 (Real.sin x) = Real.cos x :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_l2227_222741


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l2227_222714

-- Define the centers of the circles
def A : ℝ × ℝ := (-2, 2)
def B : ℝ × ℝ := (3, 3)
def C : ℝ × ℝ := (10, 4)

-- Define the radii of the circles
def r_A : ℝ := 2
def r_B : ℝ := 3
def r_C : ℝ := 4

-- Define the distance between centers
def dist_AB : ℝ := r_A + r_B
def dist_BC : ℝ := r_B + r_C

-- Theorem statement
theorem area_of_triangle_ABC :
  let triangle_area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  triangle_area = 1 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l2227_222714


namespace NUMINAMATH_CALUDE_expression_value_at_two_l2227_222789

theorem expression_value_at_two :
  let x : ℕ := 2
  x + x * (x ^ x) = 10 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_two_l2227_222789


namespace NUMINAMATH_CALUDE_total_sales_given_april_l2227_222703

/-- Bennett's window screen sales pattern --/
structure BennettSales where
  january : ℕ
  february : ℕ := 2 * january
  march : ℕ := (january + february) / 2
  april : ℕ := min (2 * march) 20000

/-- Theorem: Total sales given April sales of 18000 --/
theorem total_sales_given_april (sales : BennettSales) 
  (h_april : sales.april = 18000) : 
  sales.january + sales.february + sales.march + sales.april = 45000 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_given_april_l2227_222703


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l2227_222727

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 400 →
  hindu_percentage = 28 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 72 →
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_boys : ℚ)) / total_boys = 44 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l2227_222727


namespace NUMINAMATH_CALUDE_pet_store_cats_l2227_222750

theorem pet_store_cats (initial_birds initial_puppies initial_spiders : ℕ)
  (sold_birds adopted_puppies loose_spiders : ℕ)
  (total_left : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_spiders = 15 →
  sold_birds = initial_birds / 2 →
  adopted_puppies = 3 →
  loose_spiders = 7 →
  total_left = 25 →
  ∃ initial_cats : ℕ,
    initial_cats = 5 ∧
    total_left = initial_birds - sold_birds +
                 initial_puppies - adopted_puppies +
                 initial_cats +
                 initial_spiders - loose_spiders :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cats_l2227_222750


namespace NUMINAMATH_CALUDE_crayon_ratio_l2227_222726

def karen_crayons : ℕ := 128
def judah_crayons : ℕ := 8

def gilbert_crayons : ℕ := 4 * judah_crayons
def beatrice_crayons : ℕ := 2 * gilbert_crayons

theorem crayon_ratio :
  karen_crayons / beatrice_crayons = 2 := by
  sorry

end NUMINAMATH_CALUDE_crayon_ratio_l2227_222726


namespace NUMINAMATH_CALUDE_diff_eq_linear_solution_l2227_222730

/-- The differential equation y'' = 0 has a general solution of the form y = C₁x + C₂,
    where C₁ and C₂ are arbitrary constants. -/
theorem diff_eq_linear_solution (x : ℝ) :
  ∃ (y : ℝ → ℝ) (C₁ C₂ : ℝ), (∀ x, (deriv^[2] y) x = 0) ∧ (∀ x, y x = C₁ * x + C₂) := by
  sorry

end NUMINAMATH_CALUDE_diff_eq_linear_solution_l2227_222730


namespace NUMINAMATH_CALUDE_circle_radius_problem_l2227_222704

theorem circle_radius_problem (r : ℝ) (h : r > 0) :
  3 * (2 * 2 * Real.pi * r) = 3 * (Real.pi * r ^ 2) → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l2227_222704


namespace NUMINAMATH_CALUDE_max_value_inequality_l2227_222794

theorem max_value_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 6 + 8 * y * z ≤ Real.sqrt 22 ∧
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^2 + y^2 + z^2 = 1 ∧
    2 * x * y * Real.sqrt 6 + 8 * y * z = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2227_222794


namespace NUMINAMATH_CALUDE_walkers_speed_l2227_222762

theorem walkers_speed (speed_man2 : ℝ) (distance_apart : ℝ) (time : ℝ) (speed_man1 : ℝ) :
  speed_man2 = 12 →
  distance_apart = 2 →
  time = 1 →
  speed_man2 * time - speed_man1 * time = distance_apart →
  speed_man1 = 10 := by
sorry

end NUMINAMATH_CALUDE_walkers_speed_l2227_222762


namespace NUMINAMATH_CALUDE_parabola_equation_l2227_222700

/-- The equation of a parabola with focus at the center of x^2 + y^2 = 4x and vertex at origin -/
theorem parabola_equation (x y : ℝ) :
  (∃ (c : ℝ × ℝ), c.1^2 + c.2^2 = 4*c.1 ∧ 
   (x - c.1)^2 + (y - c.2)^2 = (x - 0)^2 + (y - 0)^2) →
  y^2 = 8*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2227_222700


namespace NUMINAMATH_CALUDE_thousandth_special_number_l2227_222783

/-- A function that returns true if n is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that returns true if n is a perfect cube --/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

/-- The sequence of positive integers that are neither perfect squares nor perfect cubes --/
def specialSequence : ℕ → ℕ :=
  fun n => sorry

theorem thousandth_special_number :
  specialSequence 1000 = 1039 := by sorry

end NUMINAMATH_CALUDE_thousandth_special_number_l2227_222783


namespace NUMINAMATH_CALUDE_jill_clothing_expenditure_l2227_222786

theorem jill_clothing_expenditure 
  (total : ℝ) 
  (food_percent : ℝ) 
  (other_percent : ℝ) 
  (clothing_tax_rate : ℝ) 
  (other_tax_rate : ℝ) 
  (total_tax_rate : ℝ) 
  (h1 : food_percent = 0.2)
  (h2 : other_percent = 0.3)
  (h3 : clothing_tax_rate = 0.04)
  (h4 : other_tax_rate = 0.1)
  (h5 : total_tax_rate = 0.05)
  (h6 : clothing_tax_rate * (1 - food_percent - other_percent) * total + 
        other_tax_rate * other_percent * total = total_tax_rate * total) :
  1 - food_percent - other_percent = 0.5 := by
sorry

end NUMINAMATH_CALUDE_jill_clothing_expenditure_l2227_222786


namespace NUMINAMATH_CALUDE_sin_690_degrees_l2227_222792

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l2227_222792


namespace NUMINAMATH_CALUDE_smallest_possible_students_l2227_222784

/-- Represents the number of students in each of the four equal-sized groups -/
def n : ℕ := 7

/-- The total number of students in the drama club -/
def total_students : ℕ := 4 * n + 2 * (n + 1)

/-- The drama club has six groups -/
axiom six_groups : ℕ

/-- Four groups have the same number of students -/
axiom four_equal_groups : ℕ

/-- Two groups have one more student than the other four -/
axiom two_larger_groups : ℕ

/-- The total number of groups is six -/
axiom total_groups : six_groups = 4 + 2

/-- The total number of students exceeds 40 -/
axiom exceeds_forty : total_students > 40

/-- 44 is the smallest number of students satisfying all conditions -/
theorem smallest_possible_students : total_students = 44 ∧ 
  ∀ m : ℕ, m < n → 4 * m + 2 * (m + 1) ≤ 40 := by sorry

end NUMINAMATH_CALUDE_smallest_possible_students_l2227_222784


namespace NUMINAMATH_CALUDE_carnival_game_cost_per_play_l2227_222731

/-- Represents the carnival game scenario -/
structure CarnivalGame where
  budget : ℚ
  red_points : ℕ
  green_points : ℕ
  games_played : ℕ
  red_buckets_hit : ℕ
  green_buckets_hit : ℕ
  total_points_possible : ℕ

/-- Calculates the cost per play for the carnival game -/
def cost_per_play (game : CarnivalGame) : ℚ :=
  game.budget / game.games_played

/-- Theorem stating that the cost per play is $1.50 for the given scenario -/
theorem carnival_game_cost_per_play :
  let game : CarnivalGame := {
    budget := 3,
    red_points := 2,
    green_points := 3,
    games_played := 2,
    red_buckets_hit := 4,
    green_buckets_hit := 5,
    total_points_possible := 38
  }
  cost_per_play game = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_carnival_game_cost_per_play_l2227_222731


namespace NUMINAMATH_CALUDE_product_equivalence_l2227_222739

theorem product_equivalence : 
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_product_equivalence_l2227_222739


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_greater_than_two_l2227_222797

-- Define set A
def A : Set ℝ := {x : ℝ | |2*x - 1| ≤ 3}

-- Define set B
def B (a : ℝ) : Set ℝ := Set.Ioo (-3) a

-- Theorem statement
theorem intersection_equality_implies_a_greater_than_two (a : ℝ) :
  A ∩ B a = A → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_greater_than_two_l2227_222797


namespace NUMINAMATH_CALUDE_students_per_van_l2227_222715

/-- Given five coaster vans transporting 60 boys and 80 girls, prove that each van carries 28 students. -/
theorem students_per_van (num_vans : ℕ) (num_boys : ℕ) (num_girls : ℕ) 
  (h1 : num_vans = 5)
  (h2 : num_boys = 60)
  (h3 : num_girls = 80) :
  (num_boys + num_girls) / num_vans = 28 := by
  sorry

end NUMINAMATH_CALUDE_students_per_van_l2227_222715


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2227_222735

theorem right_triangle_arctan_sum (a b c k : ℝ) (h1 : k ≠ 0) (h2 : a^2 + b^2 = c^2) :
  Real.arctan (a / (b + c + k)) + Real.arctan (b / (a + c + k)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2227_222735


namespace NUMINAMATH_CALUDE_dasha_flag_count_l2227_222751

/-- Represents the number of flags held by each first-grader -/
structure FlagCount where
  tata : ℕ
  yasha : ℕ
  vera : ℕ
  maxim : ℕ
  dasha : ℕ

/-- The problem statement -/
def flag_problem (fc : FlagCount) : Prop :=
  fc.tata + fc.yasha + fc.vera + fc.maxim + fc.dasha = 37 ∧
  fc.yasha + fc.vera + fc.maxim + fc.dasha = 32 ∧
  fc.vera + fc.maxim + fc.dasha = 20 ∧
  fc.maxim + fc.dasha = 14 ∧
  fc.dasha = 8

/-- The theorem to prove -/
theorem dasha_flag_count :
  ∀ fc : FlagCount, flag_problem fc → fc.dasha = 8 := by
  sorry

end NUMINAMATH_CALUDE_dasha_flag_count_l2227_222751


namespace NUMINAMATH_CALUDE_jindra_dice_count_l2227_222770

/-- Represents the number of dice in half a layer -/
def half_layer : ℕ := 18

/-- Represents the number of complete layers -/
def complete_layers : ℕ := 6

/-- Theorem stating the total number of dice Jindra had yesterday -/
theorem jindra_dice_count : 
  (2 * half_layer * complete_layers) + half_layer = 234 := by
  sorry

end NUMINAMATH_CALUDE_jindra_dice_count_l2227_222770


namespace NUMINAMATH_CALUDE_romney_value_l2227_222764

theorem romney_value (N O : ℕ) (a b c d e f : ℕ) :
  (0 < N) → (N < O) →  -- N/O is a proper fraction
  (N = 4) → (O = 7) →  -- N/O = 4/7
  (0 ≤ a) → (a ≤ 9) → (0 ≤ b) → (b ≤ 9) → (0 ≤ c) → (c ≤ 9) →
  (0 ≤ d) → (d ≤ 9) → (0 ≤ e) → (e ≤ 9) → (0 ≤ f) → (f ≤ 9) →  -- Each letter is a digit
  (a ≠ b) → (a ≠ c) → (a ≠ d) → (a ≠ e) → (a ≠ f) →
  (b ≠ c) → (b ≠ d) → (b ≠ e) → (b ≠ f) →
  (c ≠ d) → (c ≠ e) → (c ≠ f) →
  (d ≠ e) → (d ≠ f) →
  (e ≠ f) →  -- All letters are distinct
  (N : ℚ) / (O : ℚ) = 
    (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + (d : ℚ) / 10000 + (e : ℚ) / 100000 + (f : ℚ) / 1000000 +
    (a : ℚ) / 1000000 + (b : ℚ) / 10000000 + (c : ℚ) / 100000000 + (d : ℚ) / 1000000000 + (e : ℚ) / 10000000000 + (f : ℚ) / 100000000000 +
    (a : ℚ) / 100000000000 + (b : ℚ) / 1000000000000 + (c : ℚ) / 10000000000000 + (d : ℚ) / 100000000000000 + (e : ℚ) / 1000000000000000 + (f : ℚ) / 10000000000000000 +
    (a : ℚ) / 10000000000000000 + (b : ℚ) / 100000000000000000 + (c : ℚ) / 1000000000000000000 + (d : ℚ) / 10000000000000000000 + (e : ℚ) / 100000000000000000000 + (f : ℚ) / 1000000000000000000000 →  -- Decimal representation
  a = 5 ∧ b = 7 ∧ c = 1 ∧ d = 4 ∧ e = 2 ∧ f = 8 := by
  sorry

end NUMINAMATH_CALUDE_romney_value_l2227_222764


namespace NUMINAMATH_CALUDE_function_identity_l2227_222798

theorem function_identity (f : ℕ → ℕ) 
  (h1 : ∀ (m n : ℕ), f (m^2 + n^2) = (f m)^2 + (f n)^2) 
  (h2 : f 1 > 0) : 
  ∀ (n : ℕ), f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2227_222798


namespace NUMINAMATH_CALUDE_parabola_r_value_l2227_222723

/-- A parabola in the xy-plane defined by x = py^2 + qy + r -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (para : Parabola) (y : ℝ) : ℝ :=
  para.p * y^2 + para.q * y + para.r

theorem parabola_r_value (para : Parabola) :
  para.x_coord 4 = 5 →
  para.x_coord 6 = 3 →
  para.x_coord 0 = 3 →
  para.r = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_r_value_l2227_222723


namespace NUMINAMATH_CALUDE_total_dots_is_89_l2227_222799

/-- The number of ladybugs caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of dots on each ladybug caught on Monday -/
def monday_dots_per_ladybug : ℕ := 6

/-- The number of ladybugs caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The number of ladybugs caught on Wednesday -/
def wednesday_ladybugs : ℕ := 4

/-- The number of dots on each ladybug caught on Tuesday -/
def tuesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 1

/-- The number of dots on each ladybug caught on Wednesday -/
def wednesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 2

/-- The total number of dots on all ladybugs caught over three days -/
def total_dots : ℕ :=
  monday_ladybugs * monday_dots_per_ladybug +
  tuesday_ladybugs * tuesday_dots_per_ladybug +
  wednesday_ladybugs * wednesday_dots_per_ladybug

theorem total_dots_is_89 : total_dots = 89 := by
  sorry

end NUMINAMATH_CALUDE_total_dots_is_89_l2227_222799


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2227_222773

/-- Represents the repeating decimal 0.53246246246... -/
def repeating_decimal : ℚ := 0.53 + (0.246 / 999)

/-- The denominator of the target fraction -/
def target_denominator : ℕ := 999900

theorem repeating_decimal_as_fraction :
  ∃ x : ℕ, (x : ℚ) / target_denominator = repeating_decimal ∧ x = 531714 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2227_222773


namespace NUMINAMATH_CALUDE_characterize_function_l2227_222793

theorem characterize_function (f : ℤ → ℤ) :
  (∀ x y z : ℤ, x + y + z = 0 → f x + f y + f z = x * y * z) →
  ∃ c : ℤ, ∀ x : ℤ, f x = (x^3 - x) / 3 + c * x :=
sorry

end NUMINAMATH_CALUDE_characterize_function_l2227_222793


namespace NUMINAMATH_CALUDE_f_sum_lower_bound_f_squared_sum_lower_bound_l2227_222756

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1)

-- Theorem 1
theorem f_sum_lower_bound : ∀ x : ℝ, f x + f (1 - x) ≥ 1 := by sorry

-- Theorem 2
theorem f_squared_sum_lower_bound (a b : ℝ) (h : a + 2 * b = 8) : f a ^ 2 + f b ^ 2 ≥ 5 := by sorry

end NUMINAMATH_CALUDE_f_sum_lower_bound_f_squared_sum_lower_bound_l2227_222756


namespace NUMINAMATH_CALUDE_max_fraction_bound_l2227_222760

theorem max_fraction_bound (A B : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (hAB : A ≠ B) 
  (hA1000 : A < 1000) (hB1000 : B < 1000) : 
  (A : ℚ) - B ≤ 499 * ((A : ℚ) + B) / 500 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_bound_l2227_222760


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2227_222745

/-- The set of real numbers x such that x^2 - 9 > 0 -/
def A : Set ℝ := {x | x^2 - 9 > 0}

/-- The set of real numbers x such that x^2 - 5/6*x + 1/6 > 0 -/
def B : Set ℝ := {x | x^2 - 5/6*x + 1/6 > 0}

/-- Theorem stating that A is a subset of B and there exists an element in B that is not in A -/
theorem sufficient_not_necessary : A ⊆ B ∧ ∃ x, x ∈ B ∧ x ∉ A :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2227_222745


namespace NUMINAMATH_CALUDE_number_between_24_and_28_l2227_222732

def possibleNumbers : List ℕ := [20, 23, 26, 29]

theorem number_between_24_and_28 (n : ℕ) 
  (h1 : n > 24) 
  (h2 : n < 28) 
  (h3 : n ∈ possibleNumbers) : 
  n = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_between_24_and_28_l2227_222732


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2227_222743

theorem two_digit_number_property : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (∃ x y : ℕ, 
    n = 10 * x + y ∧ 
    x < 10 ∧ 
    y < 10 ∧ 
    n = x^3 + y^2) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2227_222743


namespace NUMINAMATH_CALUDE_multiple_remainder_l2227_222706

theorem multiple_remainder (x : ℕ) (h : x % 9 = 5) :
  ∃ k : ℕ, k > 0 ∧ (k * x) % 9 = 8 ∧ k = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiple_remainder_l2227_222706


namespace NUMINAMATH_CALUDE_lottery_comparison_l2227_222740

-- Define the number of red and white balls
def red_balls : ℕ := 4
def white_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := red_balls + white_balls

-- Define the probability of drawing two red balls
def prob_two_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

-- Define the probability of rolling at least one four with two dice
def prob_at_least_one_four : ℚ := 1 - (5/6) * (5/6)

-- Theorem to prove
theorem lottery_comparison : prob_two_red > prob_at_least_one_four := by
  sorry


end NUMINAMATH_CALUDE_lottery_comparison_l2227_222740


namespace NUMINAMATH_CALUDE_base_8_4513_equals_2379_l2227_222725

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4513_equals_2379 :
  base_8_to_10 [3, 1, 5, 4] = 2379 := by
  sorry

end NUMINAMATH_CALUDE_base_8_4513_equals_2379_l2227_222725


namespace NUMINAMATH_CALUDE_average_of_combined_results_l2227_222747

theorem average_of_combined_results (n₁ n₂ : ℕ) (avg₁ avg₂ : ℝ) 
  (h₁ : n₁ = 55) (h₂ : n₂ = 28) (h₃ : avg₁ = 28) (h₄ : avg₂ = 55) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = (55 * 28 + 28 * 55) / (55 + 28) := by
sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l2227_222747


namespace NUMINAMATH_CALUDE_pool_filling_rate_l2227_222749

/-- Given a pool with the following properties:
  * Capacity: 60 gallons
  * Leak rate: 0.1 gallons per minute
  * Filling time: 40 minutes
  Prove that the rate at which water is provided to fill the pool is 1.6 gallons per minute. -/
theorem pool_filling_rate 
  (capacity : ℝ) 
  (leak_rate : ℝ) 
  (filling_time : ℝ) 
  (h1 : capacity = 60) 
  (h2 : leak_rate = 0.1) 
  (h3 : filling_time = 40) : 
  ∃ (fill_rate : ℝ), 
    fill_rate = 1.6 ∧ 
    (fill_rate - leak_rate) * filling_time = capacity :=
by sorry

end NUMINAMATH_CALUDE_pool_filling_rate_l2227_222749


namespace NUMINAMATH_CALUDE_multiple_subtracted_l2227_222705

theorem multiple_subtracted (a b : ℝ) (h1 : a / b = 4 / 1) 
  (h2 : ∃ x : ℝ, (a - x * b) / (2 * a - b) = 0.14285714285714285) : 
  ∃ x : ℝ, (a - x * b) / (2 * a - b) = 0.14285714285714285 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiple_subtracted_l2227_222705


namespace NUMINAMATH_CALUDE_dan_picked_nine_apples_l2227_222796

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The total number of apples picked by Benny and Dan -/
def total_apples : ℕ := 11

/-- The number of apples Dan picked -/
def dan_apples : ℕ := total_apples - benny_apples

theorem dan_picked_nine_apples : dan_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_dan_picked_nine_apples_l2227_222796


namespace NUMINAMATH_CALUDE_sum_integers_50_to_70_l2227_222774

theorem sum_integers_50_to_70 (x y : ℕ) : 
  (x = (50 + 70) * (70 - 50 + 1) / 2) →  -- Sum of integers from 50 to 70
  (y = ((70 - 50) / 2 + 1)) →            -- Number of even integers from 50 to 70
  (x + y = 1271) → 
  (x = 1260) := by sorry

end NUMINAMATH_CALUDE_sum_integers_50_to_70_l2227_222774


namespace NUMINAMATH_CALUDE_largest_c_for_range_containing_negative_five_l2227_222711

theorem largest_c_for_range_containing_negative_five :
  let f (x c : ℝ) := x^2 + 5*x + c
  ∃ (c_max : ℝ), c_max = 5/4 ∧
    (∀ c : ℝ, (∃ x : ℝ, f x c = -5) → c ≤ c_max) ∧
    (∃ x : ℝ, f x c_max = -5) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_range_containing_negative_five_l2227_222711


namespace NUMINAMATH_CALUDE_fraction_condition_necessary_not_sufficient_l2227_222778

theorem fraction_condition_necessary_not_sufficient :
  ∀ x : ℝ, (|x - 1| < 1 → (x + 3) / (x - 2) < 0) ∧
  ¬(∀ x : ℝ, (x + 3) / (x - 2) < 0 → |x - 1| < 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_condition_necessary_not_sufficient_l2227_222778


namespace NUMINAMATH_CALUDE_complex_number_location_l2227_222724

theorem complex_number_location :
  let z : ℂ := 1 / (2 + Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l2227_222724


namespace NUMINAMATH_CALUDE_cubic_function_range_l2227_222763

/-- If f(x) = x^3 - a and the graph of f(x) does not pass through the second quadrant, then a ∈ [0, +∞) -/
theorem cubic_function_range (a : ℝ) : 
  (∀ x : ℝ, (x ≤ 0 ∧ x^3 - a ≥ 0) → False) → 
  a ∈ Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_range_l2227_222763


namespace NUMINAMATH_CALUDE_f_difference_at_5_and_neg_5_l2227_222729

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x

-- State the theorem
theorem f_difference_at_5_and_neg_5 : f 5 - f (-5) = 50 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_5_and_neg_5_l2227_222729


namespace NUMINAMATH_CALUDE_intersection_sum_l2227_222759

theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d → (x = 3 ∧ y = 3)) → 
  c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l2227_222759


namespace NUMINAMATH_CALUDE_scientific_notation_505000_l2227_222758

theorem scientific_notation_505000 :
  505000 = 5.05 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_505000_l2227_222758


namespace NUMINAMATH_CALUDE_probability_five_blue_marbles_in_eight_draws_l2227_222753

/-- The probability of drawing exactly k blue marbles in n draws with replacement -/
def probability_k_blue_marbles (total_marbles blue_marbles k n : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (blue_marbles / total_marbles : ℚ) ^ k * 
  ((total_marbles - blue_marbles) / total_marbles : ℚ) ^ (n - k)

/-- The probability of drawing exactly 5 blue marbles in 8 draws with replacement
    from a bag containing 9 blue marbles and 6 red marbles -/
theorem probability_five_blue_marbles_in_eight_draws : 
  probability_k_blue_marbles 15 9 5 8 = 108864 / 390625 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_blue_marbles_in_eight_draws_l2227_222753


namespace NUMINAMATH_CALUDE_triangle_side_length_l2227_222761

-- Define the triangle PQS
structure Triangle :=
  (P Q S : ℝ × ℝ)

-- Define the length function
def length (A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_side_length (PQS : Triangle) :
  length PQS.P PQS.Q = 2 → length PQS.P PQS.S = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2227_222761


namespace NUMINAMATH_CALUDE_ms_cole_total_students_l2227_222768

/-- The number of students in Ms. Cole's sixth-level math class -/
def S6 : ℕ := 40

/-- The number of students in Ms. Cole's fourth-level math class -/
def S4 : ℕ := 4 * S6

/-- The number of students in Ms. Cole's seventh-level math class -/
def S7 : ℕ := 2 * S4

/-- The total number of math students Ms. Cole teaches -/
def total_students : ℕ := S6 + S4 + S7

/-- Theorem stating that Ms. Cole teaches 520 math students in total -/
theorem ms_cole_total_students : total_students = 520 := by
  sorry

end NUMINAMATH_CALUDE_ms_cole_total_students_l2227_222768


namespace NUMINAMATH_CALUDE_winter_carnival_participants_l2227_222736

theorem winter_carnival_participants (total_students : ℕ) (total_participants : ℕ) 
  (girls : ℕ) (boys : ℕ) (h1 : total_students = 1500) 
  (h2 : total_participants = 900) (h3 : girls + boys = total_students) 
  (h4 : (3 * girls / 4 : ℚ) + (2 * boys / 3 : ℚ) = total_participants) : 
  3 * girls / 4 = 900 := by
  sorry

end NUMINAMATH_CALUDE_winter_carnival_participants_l2227_222736


namespace NUMINAMATH_CALUDE_school_sections_l2227_222790

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 312) :
  let section_size := Nat.gcd boys girls
  let boy_sections := boys / section_size
  let girl_sections := girls / section_size
  boy_sections + girl_sections = 30 := by
sorry

end NUMINAMATH_CALUDE_school_sections_l2227_222790


namespace NUMINAMATH_CALUDE_circle_line_intersection_l2227_222757

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 8*x + 15 = 0

-- Define the line
def line (x y k : ℝ) : Prop := y = k*x - 2

-- Define the condition for common points
def has_common_points (k : ℝ) : Prop :=
  ∃ x y : ℝ, line x y k ∧ 
    ∃ x' y' : ℝ, circle_C x' y' ∧ 
      (x - x')^2 + (y - y')^2 ≤ 4

-- The main theorem
theorem circle_line_intersection (k : ℝ) :
  has_common_points k → -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l2227_222757


namespace NUMINAMATH_CALUDE_weight_system_l2227_222701

/-- Represents the weight of birds in jin -/
structure BirdWeight where
  sparrow : ℝ
  swallow : ℝ

/-- The conditions of the sparrow and swallow weight problem -/
def weightProblem (w : BirdWeight) : Prop :=
  (5 * w.sparrow + 6 * w.swallow = 1) ∧
  (w.sparrow > w.swallow) ∧
  (4 * w.sparrow + 7 * w.swallow = 5 * w.sparrow + 6 * w.swallow)

/-- The system of equations representing the sparrow and swallow weight problem -/
theorem weight_system (w : BirdWeight) (h : weightProblem w) :
  (5 * w.sparrow + 6 * w.swallow = 1) ∧
  (4 * w.sparrow + 7 * w.swallow = 5 * w.sparrow + 6 * w.swallow) :=
by sorry

end NUMINAMATH_CALUDE_weight_system_l2227_222701


namespace NUMINAMATH_CALUDE_meaningful_expression_l2227_222779

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 5)) ↔ x > 5 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2227_222779


namespace NUMINAMATH_CALUDE_pre_bought_ticket_price_l2227_222737

/-- The price of pre-bought plane tickets is $155 -/
theorem pre_bought_ticket_price :
  ∀ (pre_bought_price : ℕ) (pre_bought_quantity : ℕ) (gate_price : ℕ) (gate_quantity : ℕ) (price_difference : ℕ),
  pre_bought_quantity = 20 →
  gate_quantity = 30 →
  gate_price = 200 →
  gate_quantity * gate_price = pre_bought_quantity * pre_bought_price + price_difference →
  price_difference = 2900 →
  pre_bought_price = 155 :=
by sorry

end NUMINAMATH_CALUDE_pre_bought_ticket_price_l2227_222737


namespace NUMINAMATH_CALUDE_swimming_club_girls_l2227_222708

theorem swimming_club_girls (total_members : ℕ) (present_members : ℕ) 
  (h1 : total_members = 30)
  (h2 : present_members = 18)
  (h3 : ∃ (boys girls : ℕ), boys + girls = total_members ∧ boys / 3 + girls = present_members) :
  ∃ (girls : ℕ), girls = 12 ∧ ∃ (boys : ℕ), boys + girls = total_members := by
  sorry

end NUMINAMATH_CALUDE_swimming_club_girls_l2227_222708


namespace NUMINAMATH_CALUDE_t_range_theorem_l2227_222769

theorem t_range_theorem (t x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  3 * x^2 + 3 * z * x + z^2 = 1 →
  3 * y^2 + 3 * y * z + z^2 = 4 →
  x^2 - x * y + y^2 = t →
  (3 - Real.sqrt 5) / 2 ≤ t ∧ t ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_t_range_theorem_l2227_222769


namespace NUMINAMATH_CALUDE_calculation_proof_l2227_222780

theorem calculation_proof :
  ((-36) * (1/3 - 1/2) + 16 / (-2)^3 = 4) ∧
  ((-5 + 2) * (1/3) + 5^2 / (-5) = -6) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2227_222780


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l2227_222742

theorem girls_to_boys_ratio (total : ℕ) (girl_boy_diff : ℕ) : 
  total = 25 → girl_boy_diff = 3 → 
  ∃ (girls boys : ℕ), 
    girls + boys = total ∧ 
    girls = boys + girl_boy_diff ∧ 
    (girls : ℚ) / (boys : ℚ) = 14 / 11 := by
  sorry

#check girls_to_boys_ratio

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l2227_222742


namespace NUMINAMATH_CALUDE_calculate_expression_l2227_222702

theorem calculate_expression : |-7| + Real.sqrt 16 - (-3)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2227_222702


namespace NUMINAMATH_CALUDE_square_area_l2227_222748

/-- Square in a coordinate plane --/
structure Square where
  B : ℝ × ℝ
  C : ℝ × ℝ
  E : ℝ × ℝ
  BC_is_side : True  -- Represents that BC is a side of the square
  E_on_line : True   -- Represents that E is on a line intersecting another vertex

/-- The area of the square ABCD is 4 --/
theorem square_area (s : Square) (h1 : s.B = (0, 0)) (h2 : s.C = (2, 0)) (h3 : s.E = (2, 1)) : 
  (s.C.1 - s.B.1) ^ 2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_square_area_l2227_222748


namespace NUMINAMATH_CALUDE_function_property_proof_l2227_222713

theorem function_property_proof (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = f (4 - x))
  (h2 : ∀ x : ℝ, f (x + 1) = -f (x + 3))
  (h3 : ∃ a b : ℝ, ∀ x ∈ Set.Icc 0 4, f x = |x - a| + b) :
  ∃ a b : ℝ, (∀ x ∈ Set.Icc 0 4, f x = |x - a| + b) ∧ a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_function_property_proof_l2227_222713


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_l2227_222733

theorem quadratic_complex_roots
  (a b c : ℝ) (x : ℂ)
  (h_a : a ≠ 0)
  (h_root : a * (1 + Complex.I)^2 + b * (1 + Complex.I) + c = 0) :
  a * (1 - Complex.I)^2 + b * (1 - Complex.I) + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_l2227_222733


namespace NUMINAMATH_CALUDE_largest_roots_ratio_l2227_222722

/-- The polynomial f(x) = 1 - x - 4x² + x⁴ -/
def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4

/-- The polynomial g(x) = 16 - 8x - 16x² + x⁴ -/
def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

/-- x₁ is the largest root of f -/
def x₁ : ℝ := sorry

/-- x₂ is the largest root of g -/
def x₂ : ℝ := sorry

theorem largest_roots_ratio :
  x₁ / x₂ = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_largest_roots_ratio_l2227_222722


namespace NUMINAMATH_CALUDE_solve_jewelry_store_problem_l2227_222765

/-- Represents the jewelry store inventory problem --/
def jewelry_store_problem (necklace_capacity ring_capacity bracelet_capacity : ℕ)
  (current_rings current_bracelets : ℕ)
  (price_necklace price_ring price_bracelet : ℕ)
  (total_cost : ℕ) : Prop :=
  let rings_needed := ring_capacity - current_rings
  let bracelets_needed := bracelet_capacity - current_bracelets
  let necklaces_on_stand := necklace_capacity - 
    ((total_cost - price_ring * rings_needed - price_bracelet * bracelets_needed) / price_necklace)
  necklaces_on_stand = 5

/-- The main theorem stating the solution to the jewelry store problem --/
theorem solve_jewelry_store_problem :
  jewelry_store_problem 12 30 15 18 8 4 10 5 183 := by
  sorry

end NUMINAMATH_CALUDE_solve_jewelry_store_problem_l2227_222765


namespace NUMINAMATH_CALUDE_pants_price_problem_l2227_222707

theorem pants_price_problem (total_cost shirt_price pants_price shoes_price : ℚ) : 
  total_cost = 340 →
  shirt_price = (3/4) * pants_price →
  shoes_price = pants_price + 10 →
  total_cost = shirt_price + pants_price + shoes_price →
  pants_price = 120 := by
sorry

end NUMINAMATH_CALUDE_pants_price_problem_l2227_222707


namespace NUMINAMATH_CALUDE_intersection_equality_subset_condition_l2227_222712

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 < 2*x + 1 ∧ 2*x + 1 < 7}
def B : Set ℝ := {x | x < -4 ∨ x > 2}
def C (a : ℝ) : Set ℝ := {x | 3*a - 2 < x ∧ x < a + 1}

-- Statement 1: A ∩ (C_R B) = {x | -2 < x ≤ 2}
theorem intersection_equality : A ∩ (Set.Icc (-4) 2) = {x : ℝ | -2 < x ∧ x ≤ 2} := by sorry

-- Statement 2: C_R (A∪B) ⊆ C if and only if -3 < a < -2/3
theorem subset_condition (a : ℝ) : Set.Icc (-4) 2 ⊆ C a ↔ -3 < a ∧ a < -2/3 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_subset_condition_l2227_222712


namespace NUMINAMATH_CALUDE_jellybean_problem_l2227_222772

theorem jellybean_problem : ∃ (n : ℕ), n ≥ 150 ∧ n % 19 = 17 ∧ ∀ (m : ℕ), m ≥ 150 ∧ m % 19 = 17 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l2227_222772


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2227_222728

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem stating that if (1,x) and (-2,3) are perpendicular, then x = 2/3 -/
theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, perpendicular (1, x) (-2, 3) → x = 2/3 := by
  sorry

#check perpendicular_vectors_x_value

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2227_222728


namespace NUMINAMATH_CALUDE_circle_center_l2227_222744

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in the form (x - h)² + (y - k)² = r² -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The given circle equation -/
def given_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem circle_center :
  ∃ (c : Circle), (∀ x y : ℝ, given_equation x y ↔ c.equation x y) ∧ c.center = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l2227_222744


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2227_222721

/-- Given points A, B, C, and D where D is the midpoint of AB, 
    prove that the sum of the slope and y-intercept of line CD is 27/10 -/
theorem line_slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 8) → 
  B = (0, -2) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let slope := (C.2 - D.2) / (C.1 - D.1)
  let y_intercept := D.2
  slope + y_intercept = 27 / 10 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2227_222721


namespace NUMINAMATH_CALUDE_unique_two_digit_prime_sum_reverse_l2227_222782

def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_prime_sum_reverse : 
  ∃! n : ℕ, is_two_digit n ∧ Nat.Prime (n + reverse_digits n) :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_prime_sum_reverse_l2227_222782


namespace NUMINAMATH_CALUDE_intersecting_chords_length_l2227_222785

/-- Power of a Point theorem for intersecting chords --/
axiom power_of_point (AP BP CP DP : ℝ) : AP * BP = CP * DP

/-- Proof that DP = 8/3 given the conditions --/
theorem intersecting_chords_length (AP BP CP DP : ℝ) 
  (h1 : AP = 4) 
  (h2 : CP = 9) 
  (h3 : BP = 6) : 
  DP = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_intersecting_chords_length_l2227_222785


namespace NUMINAMATH_CALUDE_expression_simplification_l2227_222788

theorem expression_simplification :
  let x := Real.pi / 18  -- 10 degrees in radians
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) /
  (Real.sin (17 * x) - Real.sqrt (1 - Real.sin (17 * x) ^ 2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2227_222788


namespace NUMINAMATH_CALUDE_train_length_train_length_is_240_l2227_222710

/-- The length of a train crossing a bridge -/
theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - bridge_length

/-- Proof that the train length is 240 meters -/
theorem train_length_is_240 :
  train_length 150 20 70.2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_is_240_l2227_222710


namespace NUMINAMATH_CALUDE_numerator_increase_percentage_numerator_increase_proof_l2227_222771

theorem numerator_increase_percentage (original_fraction : ℚ) 
  (denominator_decrease : ℚ) (resulting_fraction : ℚ) : ℚ :=
  let numerator_increase := 
    (resulting_fraction * (1 - denominator_decrease / 100) / original_fraction - 1) * 100
  numerator_increase

#check numerator_increase_percentage (3/4) 8 (15/16) = 15

theorem numerator_increase_proof :
  numerator_increase_percentage (3/4) 8 (15/16) = 15 := by sorry

end NUMINAMATH_CALUDE_numerator_increase_percentage_numerator_increase_proof_l2227_222771


namespace NUMINAMATH_CALUDE_unique_real_root_of_equation_l2227_222781

theorem unique_real_root_of_equation :
  ∃! x : ℝ, 2 * Real.sqrt (x - 3) + 6 = x :=
by sorry

end NUMINAMATH_CALUDE_unique_real_root_of_equation_l2227_222781


namespace NUMINAMATH_CALUDE_b_spending_percentage_l2227_222717

/-- Proves that B spends 85% of her salary given the conditions of the problem -/
theorem b_spending_percentage (total_salary : ℕ) (a_spending_rate : ℚ) (b_salary : ℕ) :
  total_salary = 14000 →
  a_spending_rate = 4/5 →
  b_salary = 8000 →
  let a_salary := total_salary - b_salary
  let a_savings := a_salary * (1 - a_spending_rate)
  let b_savings := a_savings
  let b_spending_rate := 1 - (b_savings / b_salary)
  b_spending_rate = 17/20 := by
sorry

#eval (17 : ℚ) / 20  -- Should output 0.85

end NUMINAMATH_CALUDE_b_spending_percentage_l2227_222717


namespace NUMINAMATH_CALUDE_initial_money_theorem_l2227_222777

def meat_cost : ℝ := 17
def chicken_cost : ℝ := 22
def veggie_cost : ℝ := 43
def egg_cost : ℝ := 5
def dog_food_cost : ℝ := 45
def cat_food_cost : ℝ := 18
def discount_rate : ℝ := 0.1
def money_left : ℝ := 35

def total_spent : ℝ := meat_cost + chicken_cost + veggie_cost + egg_cost + dog_food_cost + (cat_food_cost * (1 - discount_rate))

theorem initial_money_theorem :
  total_spent + money_left = 183.20 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_theorem_l2227_222777


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l2227_222709

theorem percentage_of_percentage (x : ℝ) (h : x ≠ 0) :
  (60 / 100) * (30 / 100) * x = (18 / 100) * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l2227_222709
