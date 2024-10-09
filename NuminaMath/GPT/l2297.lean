import Mathlib

namespace range_of_x_l2297_229715

theorem range_of_x (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + a * x > 4 * x + a - 3) ↔ (x < -1 ∨ x > 3) :=
by
  sorry

end range_of_x_l2297_229715


namespace combined_flock_size_l2297_229789

def original_ducks := 100
def killed_per_year := 20
def born_per_year := 30
def years_passed := 5
def another_flock := 150

theorem combined_flock_size :
  original_ducks + years_passed * (born_per_year - killed_per_year) + another_flock = 300 :=
by
  sorry

end combined_flock_size_l2297_229789


namespace largest_of_eight_consecutive_integers_l2297_229796

theorem largest_of_eight_consecutive_integers (n : ℕ) 
  (h : 8 * n + 28 = 3652) : n + 7 = 460 := by 
  sorry

end largest_of_eight_consecutive_integers_l2297_229796


namespace greatest_divisor_of_product_of_four_consecutive_integers_l2297_229778

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l2297_229778


namespace bills_equal_at_80_minutes_l2297_229708

variable (m : ℝ)

def C_U : ℝ := 8 + 0.25 * m
def C_A : ℝ := 12 + 0.20 * m

theorem bills_equal_at_80_minutes (h : C_U m = C_A m) : m = 80 :=
by {
  sorry
}

end bills_equal_at_80_minutes_l2297_229708


namespace Q_subset_P_l2297_229754

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by
  sorry

end Q_subset_P_l2297_229754


namespace charles_finishes_in_11_days_l2297_229727

theorem charles_finishes_in_11_days : 
  ∀ (total_pages : ℕ) (pages_mon : ℕ) (pages_tue : ℕ) (pages_wed : ℕ) (pages_thu : ℕ) 
    (does_not_read_on_weekend : Prop),
  total_pages = 96 →
  pages_mon = 7 →
  pages_tue = 12 →
  pages_wed = 10 →
  pages_thu = 6 →
  does_not_read_on_weekend →
  ∃ days_to_finish : ℕ, days_to_finish = 11 :=
by
  intros
  sorry

end charles_finishes_in_11_days_l2297_229727


namespace caleb_puffs_to_mom_l2297_229793

variable (initial_puffs : ℕ) (puffs_to_sister : ℕ) (puffs_to_grandmother : ℕ) (puffs_to_dog : ℕ)
variable (puffs_per_friend : ℕ) (friends : ℕ)

theorem caleb_puffs_to_mom
  (h1 : initial_puffs = 40) 
  (h2 : puffs_to_sister = 3)
  (h3 : puffs_to_grandmother = 5) 
  (h4 : puffs_to_dog = 2) 
  (h5 : puffs_per_friend = 9)
  (h6 : friends = 3)
  : initial_puffs - ( friends * puffs_per_friend + puffs_to_sister + puffs_to_grandmother + puffs_to_dog ) = 3 :=
by
  sorry

end caleb_puffs_to_mom_l2297_229793


namespace square_difference_example_l2297_229741

theorem square_difference_example : 601^2 - 599^2 = 2400 := 
by sorry

end square_difference_example_l2297_229741


namespace roots_sum_cubes_l2297_229717

theorem roots_sum_cubes (a b c d : ℝ) 
  (h_eqn : ∀ x : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d) → 
    3 * x^4 + 6 * x^3 + 1002 * x^2 + 2005 * x + 4010 = 0) :
  (a + b)^3 + (b + c)^3 + (c + d)^3 + (d + a)^3 = 9362 :=
by { sorry }

end roots_sum_cubes_l2297_229717


namespace determine_d_l2297_229721

theorem determine_d (d c f : ℚ) :
  (3 * x^3 - 2 * x^2 + x - (5/4)) * (3 * x^3 + d * x^2 + c * x + f) = 9 * x^6 - 5 * x^5 - x^4 + 20 * x^3 - (25/4) * x^2 + (15/4) * x - (5/2) →
  d = 1 / 3 :=
by
  sorry

end determine_d_l2297_229721


namespace price_of_most_expensive_book_l2297_229765

-- Define the conditions
def number_of_books := 41
def price_increment := 3

-- Define the price of the n-th book as a function of the price of the first book
def price (c : ℕ) (n : ℕ) : ℕ := c + price_increment * (n - 1)

-- Define a theorem stating the result
theorem price_of_most_expensive_book (c : ℕ) :
  c = 30 → price c number_of_books = 150 :=
by {
  sorry
}

end price_of_most_expensive_book_l2297_229765


namespace a_in_M_l2297_229797

def M : Set ℝ := { x | x ≤ 5 }
def a : ℝ := 2

theorem a_in_M : a ∈ M :=
by
  -- Proof omitted
  sorry

end a_in_M_l2297_229797


namespace new_total_lifting_capacity_is_correct_l2297_229716

-- Define the initial lifting capacities and improvements
def initial_clean_and_jerk : ℕ := 80
def initial_snatch : ℕ := 50
def clean_and_jerk_multiplier : ℕ := 2
def snatch_increment_percentage : ℕ := 80

-- Calculated values
def new_clean_and_jerk := initial_clean_and_jerk * clean_and_jerk_multiplier
def snatch_increment := initial_snatch * snatch_increment_percentage / 100
def new_snatch := initial_snatch + snatch_increment
def new_total_lifting_capacity := new_clean_and_jerk + new_snatch

-- Theorem statement to be proven
theorem new_total_lifting_capacity_is_correct :
  new_total_lifting_capacity = 250 := 
sorry

end new_total_lifting_capacity_is_correct_l2297_229716


namespace range_of_a_l2297_229737

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) := 
by 
  sorry

end range_of_a_l2297_229737


namespace all_points_equal_l2297_229742

-- Define the problem conditions and variables
variable (P : Type) -- points in the plane
variable [MetricSpace P] -- the plane is a metric space
variable (f : P → ℝ) -- assignment of numbers to points
variable (incenter : P → P → P → P) -- calculates incenter of a nondegenerate triangle

-- Condition: the value at the incenter of a triangle is the arithmetic mean of the values at the vertices
axiom incenter_mean_property : ∀ (A B C : P), 
  (A ≠ B) → (B ≠ C) → (A ≠ C) →
  f (incenter A B C) = (f A + f B + f C) / 3

-- The theorem to be proved
theorem all_points_equal : ∀ x y : P, f x = f y :=
by
  sorry

end all_points_equal_l2297_229742


namespace tomatoes_picked_l2297_229794

theorem tomatoes_picked (initial_tomatoes picked_tomatoes : ℕ)
  (h₀ : initial_tomatoes = 17)
  (h₁ : initial_tomatoes - picked_tomatoes = 8) :
  picked_tomatoes = 9 :=
by
  sorry

end tomatoes_picked_l2297_229794


namespace leaves_decrease_by_four_fold_l2297_229774

theorem leaves_decrease_by_four_fold (x y : ℝ) (h1 : y ≤ x / 4) : 
  9 * y ≤ (9 * x) / 4 := by 
  sorry

end leaves_decrease_by_four_fold_l2297_229774


namespace percentage_decrease_l2297_229775

theorem percentage_decrease (x : ℝ) (h : x > 0) : ∃ p : ℝ, p = 0.20 ∧ ((1.25 * x) * (1 - p) = x) :=
by
  sorry

end percentage_decrease_l2297_229775


namespace sum_eq_product_l2297_229784

theorem sum_eq_product (a b c : ℝ) (h1 : 1 + b * c ≠ 0) (h2 : 1 + c * a ≠ 0) (h3 : 1 + a * b ≠ 0) :
  (b - c) / (1 + b * c) + (c - a) / (1 + c * a) + (a - b) / (1 + a * b) =
  ((b - c) * (c - a) * (a - b)) / ((1 + b * c) * (1 + c * a) * (1 + a * b)) :=
by
  sorry

end sum_eq_product_l2297_229784


namespace base_h_addition_eq_l2297_229745

theorem base_h_addition_eq (h : ℕ) :
  let n1 := 7 * h^3 + 3 * h^2 + 6 * h + 4
  let n2 := 8 * h^3 + 4 * h^2 + 2 * h + 1
  let sum := 1 * h^4 + 7 * h^3 + 2 * h^2 + 8 * h + 5
  n1 + n2 = sum → h = 8 :=
by
  intros n1 n2 sum h_eq
  sorry

end base_h_addition_eq_l2297_229745


namespace min_value_polynomial_l2297_229744

theorem min_value_polynomial (a b : ℝ) : 
  ∃ c, (∀ a b, c ≤ a^2 + 2 * b^2 + 2 * a + 4 * b + 2008) ∧
       (∀ a b, a = -1 ∧ b = -1 → c = a^2 + 2 * b^2 + 2 * a + 4 * b + 2008) :=
sorry

end min_value_polynomial_l2297_229744


namespace point_in_fourth_quadrant_l2297_229799

-- Definitions of the quadrants as provided in the conditions
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Given point
def point : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem point_in_fourth_quadrant : fourth_quadrant point.fst point.snd :=
sorry

end point_in_fourth_quadrant_l2297_229799


namespace percent_calculation_l2297_229720

theorem percent_calculation (Part Whole : ℝ) (hPart : Part = 14) (hWhole : Whole = 70) : 
  (Part / Whole) * 100 = 20 := 
by 
  sorry

end percent_calculation_l2297_229720


namespace jessica_routes_count_l2297_229768

def line := Type

def valid_route_count (p q r s t u : line) : ℕ := 9 + 36 + 36

theorem jessica_routes_count (p q r s t u : line) :
  valid_route_count p q r s t u = 81 :=
by
  sorry

end jessica_routes_count_l2297_229768


namespace area_of_field_with_tomatoes_l2297_229771

theorem area_of_field_with_tomatoes :
  let length := 3.6
  let width := 2.5 * length
  let total_area := length * width
  let area_with_tomatoes := total_area / 2
  area_with_tomatoes = 16.2 :=
by
  sorry

end area_of_field_with_tomatoes_l2297_229771


namespace max_vx_minus_yz_l2297_229770

-- Define the set A
def A : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

-- Define the conditions
variables (v w x y z : ℤ)
#check v ∈ A -- v belongs to set A
#check w ∈ A -- w belongs to set A
#check x ∈ A -- x belongs to set A
#check y ∈ A -- y belongs to set A
#check z ∈ A -- z belongs to set A

-- vw = x
axiom vw_eq_x : v * w = x

-- w ≠ 0
axiom w_ne_zero : w ≠ 0

-- The target problem
theorem max_vx_minus_yz : ∃ v w x y z : ℤ, v ∈ A ∧ w ∈ A ∧ x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ v * w = x ∧ w ≠ 0 ∧ (v * x - y * z) = 150 := by
  sorry

end max_vx_minus_yz_l2297_229770


namespace factor_polynomial_l2297_229747

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l2297_229747


namespace minimum_dwarfs_l2297_229791

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l2297_229791


namespace average_weighted_score_l2297_229707

theorem average_weighted_score
  (score1 score2 score3 : ℕ)
  (weight1 weight2 weight3 : ℕ)
  (h_scores : score1 = 90 ∧ score2 = 85 ∧ score3 = 80)
  (h_weights : weight1 = 5 ∧ weight2 = 2 ∧ weight3 = 3) :
  (weight1 * score1 + weight2 * score2 + weight3 * score3) / (weight1 + weight2 + weight3) = 86 := 
by
  sorry

end average_weighted_score_l2297_229707


namespace profit_percent_300_l2297_229753

theorem profit_percent_300 (SP : ℝ) (CP : ℝ) (h : CP = 0.25 * SP) : ((SP - CP) / CP) * 100 = 300 :=
by
  sorry

end profit_percent_300_l2297_229753


namespace math_preference_related_to_gender_l2297_229786

-- Definitions for conditions
def total_students : ℕ := 100
def male_students : ℕ := 55
def female_students : ℕ := total_students - male_students -- 45
def likes_math : ℕ := 40
def female_likes_math : ℕ := 20
def female_not_like_math : ℕ := female_students - female_likes_math -- 25
def male_likes_math : ℕ := likes_math - female_likes_math -- 20
def male_not_like_math : ℕ := male_students - male_likes_math -- 35

-- Calculate Chi-square
def chi_square (a b c d : ℕ) : Float :=
  let numerator := (total_students * (a * d - b * c)^2).toFloat
  let denominator := ((a + b) * (c + d) * (a + c) * (b + d)).toFloat
  numerator / denominator

def k_square : Float := chi_square 20 35 20 25 -- Calculate with given values

-- Prove the result
theorem math_preference_related_to_gender :
  k_square > 7.879 :=
by
  sorry

end math_preference_related_to_gender_l2297_229786


namespace garden_area_l2297_229751

/-- A rectangular garden is 350 cm long and 50 cm wide. Determine its area in square meters. -/
theorem garden_area (length_cm width_cm : ℝ) (h_length : length_cm = 350) (h_width : width_cm = 50) : (length_cm / 100) * (width_cm / 100) = 1.75 :=
by
  sorry

end garden_area_l2297_229751


namespace superior_points_in_Omega_l2297_229725

-- Define the set Omega
def Omega : Set (ℝ × ℝ) := { p | let (x, y) := p; x^2 + y^2 ≤ 2008 }

-- Definition of the superior relation
def superior (P P' : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (x', y') := P'
  x ≤ x' ∧ y ≥ y'

-- Definition of the set of points Q such that no other point in Omega is superior to Q
def Q_set : Set (ℝ × ℝ) :=
  { p | let (x, y) := p; x^2 + y^2 = 2008 ∧ x ≤ 0 ∧ y ≥ 0 }

theorem superior_points_in_Omega :
  { p | p ∈ Omega ∧ ¬ (∃ q ∈ Omega, superior q p) } = Q_set :=
by
  sorry

end superior_points_in_Omega_l2297_229725


namespace problem_l2297_229734

theorem problem (a : ℝ) (h : a^2 - 5 * a - 1 = 0) : 3 * a^2 - 15 * a = 3 :=
by
  sorry

end problem_l2297_229734


namespace algae_coverage_day_21_l2297_229776

-- Let "algae_coverage n" denote the percentage of lake covered by algae on day n.
noncomputable def algaeCoverage : ℕ → ℝ
| 0 => 1 -- initial state on day 0 taken as baseline (can be adjusted accordingly)
| (n+1) => 2 * algaeCoverage n

-- Define the problem statement
theorem algae_coverage_day_21 :
  algaeCoverage 24 = 100 → algaeCoverage 21 = 12.5 :=
by
  sorry

end algae_coverage_day_21_l2297_229776


namespace fifth_digit_is_one_l2297_229783

def self_descriptive_seven_digit_number (A B C D E F G : ℕ) : Prop :=
  A = 3 ∧ B = 2 ∧ C = 2 ∧ D = 1 ∧ E = 1 ∧ [A, B, C, D, E, F, G].count 0 = A ∧
  [A, B, C, D, E, F, G].count 1 = B ∧ [A, B, C, D, E, F, G].count 2 = C ∧
  [A, B, C, D, E, F, G].count 3 = D ∧ [A, B, C, D, E, F, G].count 4 = E

theorem fifth_digit_is_one
  (A B C D E F G : ℕ) (h : self_descriptive_seven_digit_number A B C D E F G) : E = 1 := by
  sorry

end fifth_digit_is_one_l2297_229783


namespace coordinates_of_B_l2297_229714

theorem coordinates_of_B (x y : ℝ) (A : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (2, 4) ∧ a = (3, 4) ∧ (x - 2, y - 4) = (2 * a.1, 2 * a.2) → (x, y) = (8, 12) :=
by
  intros h
  sorry

end coordinates_of_B_l2297_229714


namespace problem_equiv_l2297_229728

variable (a b c d e f : ℝ)

theorem problem_equiv :
  a * b * c = 65 → 
  b * c * d = 65 → 
  c * d * e = 1000 → 
  d * e * f = 250 → 
  (a * f) / (c * d) = 1 / 4 := 
by 
  intros h1 h2 h3 h4
  sorry

end problem_equiv_l2297_229728


namespace powers_of_two_div7_l2297_229738

theorem powers_of_two_div7 (n : ℕ) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, n = 3 * k := sorry

end powers_of_two_div7_l2297_229738


namespace brick_length_l2297_229785

theorem brick_length (w h SA : ℝ) (h_w : w = 6) (h_h : h = 2) (h_SA : SA = 152) :
  ∃ l : ℝ, 2 * l * w + 2 * l * h + 2 * w * h = SA ∧ l = 8 := 
sorry

end brick_length_l2297_229785


namespace range_of_a_l2297_229740

variable {a : ℝ}

-- Proposition p: The solution set of the inequality x^2 - (a+1)x + 1 ≤ 0 is empty
def prop_p (a : ℝ) : Prop := (a + 1) ^ 2 - 4 < 0 

-- Proposition q: The function f(x) = (a+1)^x is increasing within its domain
def prop_q (a : ℝ) : Prop := a > 0 

-- The combined conditions
def combined_conditions (a : ℝ) : Prop := (prop_p a) ∨ (prop_q a) ∧ ¬(prop_p a ∧ prop_q a)

-- The range of values for a
theorem range_of_a (h : combined_conditions a) : -3 < a ∧ a ≤ 0 ∨ a ≥ 1 :=
  sorry

end range_of_a_l2297_229740


namespace parabola_distance_l2297_229730

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l2297_229730


namespace roots_cubed_l2297_229767

noncomputable def q (b c : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * x + b^2 - c^2
noncomputable def p (b c : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * (b^2 + 3 * c^2) * x + (b^2 - c^2)^3 
def x1 (b c : ℝ) := b + c
def x2 (b c : ℝ) := b - c

theorem roots_cubed (b c : ℝ) :
  (q b c (x1 b c) = 0 ∧ q b c (x2 b c) = 0) →
  (p b c ((x1 b c)^3) = 0 ∧ p b c ((x2 b c)^3) = 0) :=
by
  sorry

end roots_cubed_l2297_229767


namespace students_brought_two_plants_l2297_229790

theorem students_brought_two_plants 
  (a1 a2 a3 a4 a5 : ℕ) (p1 p2 p3 p4 p5 : ℕ)
  (h1 : a1 + a2 + a3 + a4 + a5 = 20)
  (h2 : a1 * p1 + a2 * p2 + a3 * p3 + a4 * p4 + a5 * p5 = 30)
  (h3 : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
        p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5)
  : ∃ a : ℕ, a = 1 ∧ (∃ i : ℕ, p1 = 2 ∨ p2 = 2 ∨ p3 = 2 ∨ p4 = 2 ∨ p5 = 2) :=
sorry

end students_brought_two_plants_l2297_229790


namespace buses_needed_40_buses_needed_30_l2297_229733

-- Define the number of students
def number_of_students : ℕ := 186

-- Define the function to calculate minimum buses needed
def min_buses_needed (n : ℕ) : ℕ := (number_of_students + n - 1) / n

-- Theorem statements for the specific cases
theorem buses_needed_40 : min_buses_needed 40 = 5 := 
by 
  sorry

theorem buses_needed_30 : min_buses_needed 30 = 7 := 
by 
  sorry

end buses_needed_40_buses_needed_30_l2297_229733


namespace trees_left_after_typhoon_l2297_229709

-- Define the initial count of trees and the number of trees that died
def initial_trees := 150
def trees_died := 24

-- Define the expected number of trees left
def expected_trees_left := 126

-- The statement to be proven: after trees died, the number of trees left is as expected
theorem trees_left_after_typhoon : (initial_trees - trees_died) = expected_trees_left := by
  sorry

end trees_left_after_typhoon_l2297_229709


namespace inverse_proportion_y_relation_l2297_229766

theorem inverse_proportion_y_relation (x₁ x₂ y₁ y₂ : ℝ) 
  (hA : y₁ = -4 / x₁) 
  (hB : y₂ = -4 / x₂)
  (h₁ : x₁ < 0) 
  (h₂ : 0 < x₂) : 
  y₁ > y₂ := 
sorry

end inverse_proportion_y_relation_l2297_229766


namespace students_and_confucius_same_arrival_time_l2297_229757

noncomputable def speed_of_students_walking (x : ℝ) : ℝ := x

noncomputable def speed_of_bullock_cart (x : ℝ) : ℝ := 1.5 * x

noncomputable def time_for_students_to_school (x : ℝ) : ℝ := 30 / x

noncomputable def time_for_confucius_to_school (x : ℝ) : ℝ := 30 / (1.5 * x) + 1

theorem students_and_confucius_same_arrival_time (x : ℝ) (h1 : 0 < x) :
  30 / x = 30 / (1.5 * x) + 1 :=
by
  sorry

end students_and_confucius_same_arrival_time_l2297_229757


namespace binomial_divisible_by_prime_l2297_229763

theorem binomial_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  (Nat.choose n p) - (n / p) % p = 0 := 
sorry

end binomial_divisible_by_prime_l2297_229763


namespace tariffs_impact_but_no_timeframe_l2297_229782

noncomputable def cost_of_wine_today : ℝ := 20.00
noncomputable def increase_percentage : ℝ := 0.25
noncomputable def bottles_count : ℕ := 5
noncomputable def price_increase_for_bottles : ℝ := 25.00

theorem tariffs_impact_but_no_timeframe :
  ¬ ∃ (t : ℝ), (cost_of_wine_today * (1 + increase_percentage) - cost_of_wine_today) * bottles_count = price_increase_for_bottles →
  (t = sorry) :=
by 
  sorry

end tariffs_impact_but_no_timeframe_l2297_229782


namespace multiplication_vs_subtraction_difference_l2297_229732

variable (x : ℕ)
variable (h : x = 10)

theorem multiplication_vs_subtraction_difference :
  3 * x - (26 - x) = 14 := by
  sorry

end multiplication_vs_subtraction_difference_l2297_229732


namespace three_digit_number_multiple_of_eleven_l2297_229781

theorem three_digit_number_multiple_of_eleven:
  ∃ (a b c : ℕ), (1 ≤ a) ∧ (a ≤ 9) ∧ (0 ≤ b) ∧ (b ≤ 9) ∧ (0 ≤ c) ∧ (c ≤ 9) ∧
                  (100 * a + 10 * b + c = 11 * (a + b + c) ∧ (100 * a + 10 * b + c = 198)) :=
by
  use 1
  use 9
  use 8
  sorry

end three_digit_number_multiple_of_eleven_l2297_229781


namespace jimmy_fill_bucket_time_l2297_229729

-- Definitions based on conditions
def pool_volume : ℕ := 84
def bucket_volume : ℕ := 2
def total_time_minutes : ℕ := 14
def total_time_seconds : ℕ := total_time_minutes * 60
def trips : ℕ := pool_volume / bucket_volume

-- Theorem statement
theorem jimmy_fill_bucket_time : (total_time_seconds / trips) = 20 := by
  sorry

end jimmy_fill_bucket_time_l2297_229729


namespace functional_equation_solution_l2297_229795

def odd_integers (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem functional_equation_solution (f : ℤ → ℤ)
  (h_odd : ∀ x : ℤ, odd_integers (f x))
  (h_eq : ∀ x y : ℤ, 
    f (x + f x + y) + f (x - f x - y) = f (x + y) + f (x - y)) :
  ∃ (d k : ℤ) (ell : ℕ → ℤ), 
    (∀ i : ℕ, i < d → odd_integers (ell i)) ∧
    ∀ (m : ℤ) (i : ℕ), i < d → 
      f (m * d + i) = 2 * k * m * d + ell i :=
sorry

end functional_equation_solution_l2297_229795


namespace scientific_notation_of_42_trillion_l2297_229752

theorem scientific_notation_of_42_trillion : (42.1 * 10^12) = 4.21 * 10^13 :=
by
  sorry

end scientific_notation_of_42_trillion_l2297_229752


namespace correct_average_weight_l2297_229743

theorem correct_average_weight (n : ℕ) (incorrect_avg_weight : ℝ) (initial_avg_weight : ℝ)
  (misread_weight correct_weight : ℝ) (boys_count : ℕ) :
  incorrect_avg_weight = 58.4 →
  n = 20 →
  misread_weight = 56 →
  correct_weight = 65 →
  boys_count = n →
  initial_avg_weight = (incorrect_avg_weight * n + (correct_weight - misread_weight)) / boys_count →
  initial_avg_weight = 58.85 :=
by
  intro h1 h2 h3 h4 h5 h_avg
  sorry

end correct_average_weight_l2297_229743


namespace inverse_proportion_inequality_l2297_229701

theorem inverse_proportion_inequality :
  ∀ (y : ℝ → ℝ) (y_1 y_2 y_3 : ℝ),
  (∀ x, y x = 7 / x) →
  y (-3) = y_1 →
  y (-1) = y_2 →
  y (2) = y_3 →
  y_2 < y_1 ∧ y_1 < y_3 :=
by
  intros y y_1 y_2 y_3 hy hA hB hC
  sorry

end inverse_proportion_inequality_l2297_229701


namespace orangeade_ratio_l2297_229748

theorem orangeade_ratio (O W : ℝ) (price1 price2 : ℝ) (revenue1 revenue2 : ℝ)
  (h1 : price1 = 0.30) (h2 : price2 = 0.20)
  (h3 : revenue1 = revenue2)
  (glasses1 glasses2 : ℝ)
  (V : ℝ) :
  glasses1 = (O + W) / V → glasses2 = (O + 2 * W) / V →
  revenue1 = glasses1 * price1 → revenue2 = glasses2 * price2 →
  (O + W) * price1 = (O + 2 * W) * price2 → O / W = 1 :=
by sorry

end orangeade_ratio_l2297_229748


namespace packs_of_chewing_gum_zero_l2297_229722

noncomputable def frozen_yogurt_price : ℝ := sorry
noncomputable def chewing_gum_price : ℝ := frozen_yogurt_price / 2
noncomputable def packs_of_chewing_gum : ℕ := sorry

theorem packs_of_chewing_gum_zero 
  (F : ℝ) -- Price of a pint of frozen yogurt
  (G : ℝ) -- Price of a pack of chewing gum
  (x : ℕ) -- Number of packs of chewing gum
  (H1 : G = F / 2)
  (H2 : 5 * F + x * G + 25 = 55)
  : x = 0 :=
sorry

end packs_of_chewing_gum_zero_l2297_229722


namespace ball_bounce_height_l2297_229703

theorem ball_bounce_height (b : ℕ) : 
  ∃ b : ℕ, 400 * (3 / 4 : ℝ)^b < 50 ∧ ∀ b' < b, 400 * (3 / 4 : ℝ)^b' ≥ 50 :=
sorry

end ball_bounce_height_l2297_229703


namespace range_of_given_function_l2297_229739

noncomputable def given_function (x : ℝ) : ℝ :=
  abs (Real.sin x) / (Real.sin x) + Real.cos x / abs (Real.cos x) + abs (Real.tan x) / Real.tan x

theorem range_of_given_function : Set.range given_function = {-1, 3} :=
by
  sorry

end range_of_given_function_l2297_229739


namespace subway_train_speed_l2297_229749

theorem subway_train_speed (s : ℕ) (h1 : 0 ≤ s ∧ s ≤ 7) (h2 : s^2 + 2*s = 63) : s = 7 :=
by
  sorry

end subway_train_speed_l2297_229749


namespace vector_operation_result_l2297_229762

-- Definitions of vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (2, -3)

-- The operation 2a - b
def operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem stating the result of the operation
theorem vector_operation_result : operation a b = (-4, 5) :=
by
  sorry

end vector_operation_result_l2297_229762


namespace ribbon_per_gift_l2297_229780

theorem ribbon_per_gift
  (total_ribbon : ℕ)
  (number_of_gifts : ℕ)
  (ribbon_left : ℕ)
  (used_ribbon := total_ribbon - ribbon_left)
  (ribbon_per_gift := used_ribbon / number_of_gifts)
  (h_total : total_ribbon = 18)
  (h_gifts : number_of_gifts = 6)
  (h_left : ribbon_left = 6) :
  ribbon_per_gift = 2 := by
  sorry

end ribbon_per_gift_l2297_229780


namespace sum_of_roots_is_three_l2297_229713

theorem sum_of_roots_is_three :
  ∀ (x1 x2 : ℝ), (x1^2 - 3 * x1 - 4 = 0) ∧ (x2^2 - 3 * x2 - 4 = 0) → x1 + x2 = 3 :=
by sorry

end sum_of_roots_is_three_l2297_229713


namespace december_revenue_times_average_l2297_229787

variable (D : ℝ) -- December's revenue
variable (N : ℝ) -- November's revenue
variable (J : ℝ) -- January's revenue

-- Conditions
def revenue_in_november : N = (2/5) * D := by sorry
def revenue_in_january : J = (1/2) * N := by sorry

-- Statement to be proved
theorem december_revenue_times_average :
  D = (10/3) * ((N + J) / 2) :=
by sorry

end december_revenue_times_average_l2297_229787


namespace range_of_x_when_a_eq_1_p_and_q_range_of_a_when_not_p_sufficient_for_not_q_l2297_229788

-- Define the propositions
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := -x^2 + 5 * x - 6 ≥ 0

-- Question 1: Prove that for a = 1 and p ∧ q is true, the range of x is [2, 3)
theorem range_of_x_when_a_eq_1_p_and_q : 
  ∀ x : ℝ, p 1 x ∧ q x → 2 ≤ x ∧ x < 3 := 
by sorry

-- Question 2: Prove that if ¬p is a sufficient but not necessary condition for ¬q, 
-- then the range of a is (1, 2)
theorem range_of_a_when_not_p_sufficient_for_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬p a x → ¬q x) ∧ (∃ x : ℝ, ¬(¬p a x → ¬q x)) → 1 < a ∧ a < 2 := 
by sorry

end range_of_x_when_a_eq_1_p_and_q_range_of_a_when_not_p_sufficient_for_not_q_l2297_229788


namespace gift_wrapping_combinations_l2297_229726

theorem gift_wrapping_combinations :
  (10 * 4 * 5 * 2 = 400) := by
  sorry

end gift_wrapping_combinations_l2297_229726


namespace difference_of_squares_divisible_by_9_l2297_229705

theorem difference_of_squares_divisible_by_9 (a b : ℤ) : ∃ k : ℤ, (3 * a + 2)^2 - (3 * b + 2)^2 = 9 * k := by
  sorry

end difference_of_squares_divisible_by_9_l2297_229705


namespace minimum_a_l2297_229746

noncomputable def func (t a : ℝ) := 5 * (t + 1) ^ 2 + a / (t + 1) ^ 5

theorem minimum_a (a : ℝ) (h: ∀ t ≥ 0, func t a ≥ 24) :
  a = 2 * Real.sqrt ((24 / 7) ^ 7) :=
sorry

end minimum_a_l2297_229746


namespace function_evaluation_l2297_229758

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 + 2 * x :=
by
  sorry

end function_evaluation_l2297_229758


namespace function_zero_solution_l2297_229736

-- Define the statement of the problem
theorem function_zero_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → ∀ y : ℝ, f (x ^ 2 + y) ≥ (1 / x + 1) * f y) →
  (∀ x : ℝ, f x = 0) :=
by
  -- The proof of this theorem will be inserted here.
  sorry

end function_zero_solution_l2297_229736


namespace area_of_T_shaped_region_l2297_229711

theorem area_of_T_shaped_region :
  let ABCD_area : ℝ := 48
  let EFHG_area : ℝ := 4
  let EFGI_area : ℝ := 8
  let EFCD_area : ℝ := 12
  (ABCD_area - (EFHG_area + EFGI_area + EFCD_area)) = 24 :=
by
  let ABCD_area : ℝ := 48
  let EFHG_area : ℝ := 4
  let EFGI_area : ℝ := 8
  let EFCD_area : ℝ := 12
  exact sorry

end area_of_T_shaped_region_l2297_229711


namespace total_number_of_members_l2297_229710

variables (b g : Nat)
def girls_twice_boys : Prop := g = 2 * b
def boys_twice_remaining_girls (b g : Nat) : Prop := b = 2 * (g - 24)

theorem total_number_of_members (b g : Nat) 
  (h1 : girls_twice_boys b g) 
  (h2 : boys_twice_remaining_girls b g) : 
  b + g = 48 := by
  sorry

end total_number_of_members_l2297_229710


namespace problem_l2297_229755

-- Define sets A and B
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y <= -1 }

-- Define set C as a function of a
def C (a : ℝ) : Set ℝ := { x | x < -a / 2 }

-- The statement of the problem: if B ⊆ C, then a < 2
theorem problem (a : ℝ) : (B ⊆ C a) → a < 2 :=
by sorry

end problem_l2297_229755


namespace longer_diagonal_eq_l2297_229773

variable (a b : ℝ)
variable (h_cd : CD = a) (h_bc : BC = b) (h_diag : AC = a) (h_ad : AD = 2 * b)

theorem longer_diagonal_eq (CD BC AC AD BD : ℝ) (h_cd : CD = a)
  (h_bc : BC = b) (h_diag : AC = CD) (h_ad : AD = 2 * b) :
  BD = Real.sqrt (a^2 + 3 * b^2) :=
sorry

end longer_diagonal_eq_l2297_229773


namespace negation_of_proposition_range_of_m_l2297_229712

noncomputable def proposition (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * x - m - 1 < 0

theorem negation_of_proposition (m : ℝ) : ¬ proposition m ↔ ∀ x : ℝ, x^2 + 2 * x - m - 1 ≥ 0 :=
sorry

theorem range_of_m (m : ℝ) : proposition m → m > -2 :=
sorry

end negation_of_proposition_range_of_m_l2297_229712


namespace parabola_hyperbola_tangent_l2297_229723

-- Definitions of the parabola and hyperbola
def parabola (x : ℝ) : ℝ := x^2 + 4
def hyperbola (x y : ℝ) (m : ℝ) : Prop := y^2 - m*x^2 = 1

-- Tangency condition stating that the parabola and hyperbola are tangent implies m = 8 + 2*sqrt(15)
theorem parabola_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, parabola x = y → hyperbola x y m) → m = 8 + 2 * Real.sqrt 15 :=
by
  sorry

end parabola_hyperbola_tangent_l2297_229723


namespace find_smaller_number_l2297_229759

variable (x y : ℕ)

theorem find_smaller_number (h1 : ∃ k : ℕ, x = 2 * k ∧ y = 5 * k) (h2 : x + y = 21) : x = 6 :=
by
  sorry

end find_smaller_number_l2297_229759


namespace andrey_boris_denis_eat_candies_l2297_229792

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l2297_229792


namespace solve_prime_equation_l2297_229704

theorem solve_prime_equation (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) 
(h_eq : p^3 - q^3 = 5 * r) : p = 7 ∧ q = 2 ∧ r = 67 := 
sorry

end solve_prime_equation_l2297_229704


namespace value_of_a_l2297_229724

variable (a : ℝ)

noncomputable def f (x : ℝ) := x^2 + 8
noncomputable def g (x : ℝ) := x^2 - 4

theorem value_of_a
  (h0 : a > 0)
  (h1 : f (g a) = 8) : a = 2 :=
by
  -- conditions are used as assumptions
  let f := f
  let g := g
  sorry

end value_of_a_l2297_229724


namespace calories_in_250_grams_of_lemonade_l2297_229798

theorem calories_in_250_grams_of_lemonade:
  ∀ (lemon_juice_grams sugar_grams water_grams total_grams: ℕ)
    (lemon_juice_cal_per_100 sugar_cal_per_100 total_cal: ℕ),
  lemon_juice_grams = 150 →
  sugar_grams = 150 →
  water_grams = 300 →
  total_grams = lemon_juice_grams + sugar_grams + water_grams →
  lemon_juice_cal_per_100 = 30 →
  sugar_cal_per_100 = 386 →
  total_cal = (lemon_juice_grams * lemon_juice_cal_per_100 / 100) + (sugar_grams * sugar_cal_per_100 / 100) →
  (250:ℕ) * total_cal / total_grams = 260 :=
by
  intros lemon_juice_grams sugar_grams water_grams total_grams lemon_juice_cal_per_100 sugar_cal_per_100 total_cal
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end calories_in_250_grams_of_lemonade_l2297_229798


namespace gopi_turbans_annual_salary_l2297_229706

variable (T : ℕ) (annual_salary_turbans : ℕ)
variable (annual_salary_money : ℕ := 90)
variable (months_worked : ℕ := 9)
variable (total_months_in_year : ℕ := 12)
variable (received_money : ℕ := 55)
variable (turban_price : ℕ := 50)
variable (received_turbans : ℕ := 1)
variable (servant_share_fraction : ℚ := 3 / 4)

theorem gopi_turbans_annual_salary 
    (annual_salary_turbans : ℕ)
    (H : (servant_share_fraction * (annual_salary_money + turban_price * annual_salary_turbans) = received_money + turban_price * received_turbans))
    : annual_salary_turbans = 1 :=
sorry

end gopi_turbans_annual_salary_l2297_229706


namespace algebraic_expression_l2297_229735

variable (m n x y : ℤ)

theorem algebraic_expression (h1 : x = m) (h2 : y = n) (h3 : x - y = 2) : n - m = -2 := 
by
  sorry

end algebraic_expression_l2297_229735


namespace maximize_expression_l2297_229760

-- Given the condition
theorem maximize_expression (x y : ℝ) (h : x + y = 1) : (x^3 + 1) * (y^3 + 1) ≤ (1)^3 + 1 * (0)^3 + 1 * (0)^3 + 1 :=
sorry

end maximize_expression_l2297_229760


namespace sector_area_l2297_229702

-- Given conditions
variables {l r : ℝ}

-- Definitions (conditions from the problem)
def arc_length (l : ℝ) := l
def radius (r : ℝ) := r

-- Problem statement
theorem sector_area (l r : ℝ) : 
    (1 / 2) * l * r = (1 / 2) * l * r :=
by
  sorry

end sector_area_l2297_229702


namespace find_seventh_term_l2297_229718

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- Define arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define sum of the first n terms of the sequence
def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 0) + (d * (n * (n - 1)) / 2)

-- Now state the theorem
theorem find_seventh_term
  (h_arith_seq : arithmetic_sequence a d)
  (h_nonzero_d : d ≠ 0)
  (h_sum_five : S 5 = 5)
  (h_squares_eq : a 0 ^ 2 + a 1 ^ 2 = a 2 ^ 2 + a 3 ^ 2) :
  a 6 = 9 :=
sorry

end find_seventh_term_l2297_229718


namespace range_of_a_l2297_229719

-- Conditions for sets A and B
def SetA := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def SetB (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 2}

-- Main statement to show that A ∪ B = A implies the range of a is [-2, 0]
theorem range_of_a (a : ℝ) : (SetB a ⊆ SetA) → (-2 ≤ a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l2297_229719


namespace range_of_a_l2297_229756

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l2297_229756


namespace car_drive_time_60_kmh_l2297_229777

theorem car_drive_time_60_kmh
  (t : ℝ)
  (avg_speed : ℝ := 80)
  (dist_speed_60 : ℝ := 60 * t)
  (time_speed_90 : ℝ := 2 / 3)
  (dist_speed_90 : ℝ := 90 * time_speed_90)
  (total_distance : ℝ := dist_speed_60 + dist_speed_90)
  (total_time : ℝ := t + time_speed_90)
  (avg_speed_eq : avg_speed = total_distance / total_time) :
  t = 1 / 3 := 
sorry

end car_drive_time_60_kmh_l2297_229777


namespace find_q_l2297_229761

variable (x : ℝ)

def f (x : ℝ) := (5 * x^4 + 15 * x^3 + 30 * x^2 + 10 * x + 10)
def g (x : ℝ) := (2 * x^6 + 4 * x^4 + 10 * x^2)
def q (x : ℝ) := (-2 * x^6 + x^4 + 15 * x^3 + 20 * x^2 + 10 * x + 10)

theorem find_q :
  (∀ x, q x + g x = f x) ↔ (∀ x, q x = -2 * x^6 + x^4 + 15 * x^3 + 20 * x^2 + 10 * x + 10)
:= sorry

end find_q_l2297_229761


namespace min_AP_squared_sum_value_l2297_229769

-- Definitions based on given problem conditions
def A : ℝ := 0
def B : ℝ := 2
def C : ℝ := 4
def D : ℝ := 7
def E : ℝ := 15

def distance_squared (x y : ℝ) : ℝ := (x - y)^2

noncomputable def min_AP_squared_sum (r : ℝ) : ℝ :=
  r^2 + distance_squared r B + distance_squared r C + distance_squared r D + distance_squared r E

theorem min_AP_squared_sum_value : ∃ (r : ℝ), (min_AP_squared_sum r) = 137.2 :=
by
  existsi 5.6
  sorry

end min_AP_squared_sum_value_l2297_229769


namespace triangle_formation_l2297_229700

theorem triangle_formation (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : x₁ ≠ x₂) (h₂ : x₁ ≠ x₃) (h₃ : x₁ ≠ x₄) (h₄ : x₂ ≠ x₃) (h₅ : x₂ ≠ x₄) (h₆ : x₃ ≠ x₄)
  (h₇ : 0 < x₁) (h₈ : 0 < x₂) (h₉ : 0 < x₃) (h₁₀ : 0 < x₄)
  (h₁₁ : (x₁ + x₂ + x₃ + x₄) * (1/x₁ + 1/x₂ + 1/x₃ + 1/x₄) < 17) :
  (x₁ + x₂ > x₃) ∧ (x₂ + x₃ > x₄) ∧ (x₁ + x₃ > x₂) ∧ 
  (x₁ + x₄ > x₃) ∧ (x₁ + x₂ > x₄) ∧ (x₃ + x₄ > x₁) ∧ 
  (x₂ + x₄ > x₁) ∧ (x₂ + x₃ > x₁) :=
sorry

end triangle_formation_l2297_229700


namespace cost_price_for_one_meter_l2297_229779

variable (meters_sold : Nat) (selling_price : Nat) (loss_per_meter : Nat) (total_cost_price : Nat)
variable (cost_price_per_meter : Rat)

theorem cost_price_for_one_meter (h1 : meters_sold = 200)
                                  (h2 : selling_price = 12000)
                                  (h3 : loss_per_meter = 12)
                                  (h4 : total_cost_price = selling_price + loss_per_meter * meters_sold)
                                  (h5 : cost_price_per_meter = total_cost_price / meters_sold) :
  cost_price_per_meter = 72 := by
  sorry

end cost_price_for_one_meter_l2297_229779


namespace problem_l2297_229750

noncomputable def y := 2 + Real.sqrt 3

theorem problem (c d : ℤ) (hc : c > 0) (hd : d > 0) (h : y = c + Real.sqrt d)
  (hy_eq : y^2 + 2*y + 2/y + 1/y^2 = 20) : c + d = 5 :=
  sorry

end problem_l2297_229750


namespace average_children_in_families_with_children_l2297_229731

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l2297_229731


namespace smallest_N_value_proof_l2297_229764

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l2297_229764


namespace batsman_average_after_25th_innings_l2297_229772

theorem batsman_average_after_25th_innings (A : ℝ) (runs_25th : ℝ) (increase : ℝ) (not_out_innings : ℕ) 
    (total_innings : ℕ) (average_increase_condition : 24 * A + runs_25th = 25 * (A + increase)) :       
    runs_25th = 150 ∧ increase = 3 ∧ not_out_innings = 3 ∧ total_innings = 25 → 
    ∃ avg : ℝ, avg = 88.64 := by 
  sorry

end batsman_average_after_25th_innings_l2297_229772
