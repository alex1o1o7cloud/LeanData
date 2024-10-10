import Mathlib

namespace two_digit_number_puzzle_l36_3607

theorem two_digit_number_puzzle (a b : ℕ) : 
  a < 10 → b < 10 → a ≠ 0 →
  (10 * a + b) - (10 * b + a) = 36 →
  2 * a = b →
  (a + b) - (a - b) = 16 := by
sorry

end two_digit_number_puzzle_l36_3607


namespace triangle_area_l36_3617

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    prove that its area is 15√3 under certain conditions. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b < c →
  2 * a * c * Real.cos C + 2 * c^2 * Real.cos A = a + c →
  2 * c * Real.sin A - Real.sqrt 3 * a = 0 →
  let S := (1 / 2) * a * b * Real.sin C
  S = 15 * Real.sqrt 3 := by
  sorry

end triangle_area_l36_3617


namespace student_math_percentage_l36_3610

/-- The percentage a student got in math, given their history score, third subject score,
    and desired overall average. -/
def math_percentage (history : ℝ) (third_subject : ℝ) (overall_average : ℝ) : ℝ :=
  3 * overall_average - history - third_subject

/-- Theorem stating that the student got 74% in math, given the conditions. -/
theorem student_math_percentage :
  math_percentage 81 70 75 = 74 := by
  sorry

end student_math_percentage_l36_3610


namespace power_tower_at_three_l36_3642

theorem power_tower_at_three : 
  let x : ℕ := 3
  (x^x)^(x^x) = 27^27 := by
  sorry

end power_tower_at_three_l36_3642


namespace unique_four_digit_square_l36_3605

/-- Represents a four-digit number --/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Extracts the thousands digit from a four-digit number --/
def thousandsDigit (n : FourDigitNumber) : ℕ := n.val / 1000

/-- Extracts the hundreds digit from a four-digit number --/
def hundredsDigit (n : FourDigitNumber) : ℕ := (n.val / 100) % 10

/-- Extracts the tens digit from a four-digit number --/
def tensDigit (n : FourDigitNumber) : ℕ := (n.val / 10) % 10

/-- Extracts the units digit from a four-digit number --/
def unitsDigit (n : FourDigitNumber) : ℕ := n.val % 10

/-- Checks if a natural number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem unique_four_digit_square : 
  ∃! (n : FourDigitNumber), 
    isPerfectSquare n.val ∧ 
    thousandsDigit n = tensDigit n ∧ 
    hundredsDigit n = unitsDigit n + 1 ∧
    n.val = 8281 :=
  sorry

end unique_four_digit_square_l36_3605


namespace multiplier_value_l36_3644

theorem multiplier_value (x n : ℚ) : 
  x = 40 → (x / 4) * n + 10 - 12 = 48 → n = 5 := by
  sorry

end multiplier_value_l36_3644


namespace quadratic_equations_integer_roots_l36_3611

theorem quadratic_equations_integer_roots :
  ∃ (p q : ℤ), ∀ k : ℕ, k ≤ 9 →
    ∃ (x y : ℤ), x^2 + (p + k) * x + (q + k) = 0 ∧
                 y^2 + (p + k) * y + (q + k) = 0 ∧
                 x ≠ y :=
by sorry

end quadratic_equations_integer_roots_l36_3611


namespace log_identity_l36_3608

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_identity : log10 2 ^ 2 + log10 2 * log10 5 + log10 5 = 1 := by
  sorry

end log_identity_l36_3608


namespace arithmetic_geometric_sequences_l36_3633

/-- An arithmetic sequence {a_n} with a_2 = 2 and a_5 = 8 -/
def a : ℕ → ℝ := sorry

/-- A geometric sequence {b_n} with all terms positive and b_1 = 1 -/
def b : ℕ → ℝ := sorry

/-- Sum of the first n terms of the geometric sequence {b_n} -/
def T (n : ℕ) : ℝ := sorry

theorem arithmetic_geometric_sequences :
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 2) ∧
  (a 2 = 2) ∧
  (a 5 = 8) ∧
  (∀ n : ℕ, n ≥ 1 → b n > 0) ∧
  (b 1 = 1) ∧
  (b 2 + b 3 = a 4) ∧
  (∀ n : ℕ, n ≥ 1 → T n = 2^n - 1) :=
by sorry

end arithmetic_geometric_sequences_l36_3633


namespace integral_inequality_l36_3670

theorem integral_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 1) :
  ∫ x in (0 : ℝ)..1, ((1 - a*x)^3 + (1 - b*x)^3 + (1 - c*x)^3 - 3*x) ≥ 
    a*b + b*c + c*a - 3/2*(a + b + c) - 3/4*a*b*c := by
  sorry

end integral_inequality_l36_3670


namespace odd_function_properties_l36_3602

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_properties (f : ℝ → ℝ) (h : IsOdd f) :
  (f 0 = 0) ∧
  (∀ x ≥ 0, f x ≥ -1) →
  (∃ x ≥ 0, f x = -1) →
  (∀ x ≤ 0, f x ≤ 1) ∧
  (∃ x ≤ 0, f x = 1) := by
  sorry

end odd_function_properties_l36_3602


namespace unique_solution_l36_3643

theorem unique_solution : ∃! x : ℝ, 4 * x - 3 = 9 * (x - 7) := by
  sorry

end unique_solution_l36_3643


namespace cone_properties_l36_3695

/-- A cone with vertex P, base radius √3, and lateral area 2√3π -/
structure Cone where
  vertex : Point
  base_radius : ℝ
  lateral_area : ℝ
  h_base_radius : base_radius = Real.sqrt 3
  h_lateral_area : lateral_area = 2 * Real.sqrt 3 * Real.pi

/-- The length of the generatrix of the cone -/
def generatrix_length (c : Cone) : ℝ := sorry

/-- The angle between the generatrix and the base of the cone -/
def generatrix_base_angle (c : Cone) : ℝ := sorry

theorem cone_properties (c : Cone) : 
  generatrix_length c = 2 ∧ generatrix_base_angle c = Real.pi / 6 := by
  sorry

end cone_properties_l36_3695


namespace l₃_is_symmetric_to_l₁_l36_3646

/-- The equation of line l₁ -/
def l₁ (x y : ℝ) : Prop := x - 2 * y - 2 = 0

/-- The equation of line l₂ -/
def l₂ (x y : ℝ) : Prop := x + y = 0

/-- The equation of line l₃ -/
def l₃ (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- A point is symmetric to another point with respect to l₂ -/
def symmetric_wrt_l₂ (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -y₁ ∧ y₂ = -x₁

theorem l₃_is_symmetric_to_l₁ :
  ∀ x y : ℝ, l₃ x y ↔ ∃ x₁ y₁ : ℝ, l₁ x₁ y₁ ∧ symmetric_wrt_l₂ x y x₁ y₁ :=
sorry

end l₃_is_symmetric_to_l₁_l36_3646


namespace radical_conjugate_sum_product_l36_3698

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 4 → 
  a + b = 2 := by
sorry

end radical_conjugate_sum_product_l36_3698


namespace hexagon_midpoint_area_l36_3655

-- Define the hexagon
def regular_hexagon (side_length : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the set of line segments
def line_segments (h : Set (ℝ × ℝ)) : Set (ℝ × ℝ × ℝ × ℝ) := sorry

-- Define the midpoints of the line segments
def midpoints (segments : Set (ℝ × ℝ × ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Define the area enclosed by the midpoints
def enclosed_area (points : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem hexagon_midpoint_area :
  let h := regular_hexagon 3
  let s := line_segments h
  let m := midpoints s
  let a := enclosed_area m
  ∃ ε > 0, abs (a - 1.85) < ε := by sorry

end hexagon_midpoint_area_l36_3655


namespace additional_fabric_needed_l36_3631

def yards_to_feet (yards : ℝ) : ℝ := yards * 3

def fabric_needed_for_dresses : ℝ :=
  2 * (yards_to_feet 5.5) +
  2 * (yards_to_feet 6) +
  2 * (yards_to_feet 6.5)

def current_fabric : ℝ := 10

theorem additional_fabric_needed :
  fabric_needed_for_dresses - current_fabric = 98 :=
by sorry

end additional_fabric_needed_l36_3631


namespace sqrt_square_iff_abs_l36_3641

theorem sqrt_square_iff_abs (f g : ℝ → ℝ) :
  (∀ x, Real.sqrt (f x ^ 2) ≥ Real.sqrt (g x ^ 2)) ↔ (∀ x, |f x| ≥ |g x|) := by
  sorry

end sqrt_square_iff_abs_l36_3641


namespace k_range_for_equation_solution_l36_3647

theorem k_range_for_equation_solution (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ k * 4^x - k * 2^(x + 1) + 6 * (k - 5) = 0) →
  k ∈ Set.Icc 5 6 :=
by sorry

end k_range_for_equation_solution_l36_3647


namespace complex_modulus_problem_l36_3675

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 1 + 3 * Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l36_3675


namespace gumball_probability_l36_3682

/-- Given a box of gumballs with blue, green, red, and purple colors, 
    prove that the probability of selecting either a red or a purple gumball is 0.45, 
    given that the probability of selecting a blue gumball is 0.3 
    and the probability of selecting a green gumball is 0.25. -/
theorem gumball_probability (blue green red purple : ℝ) 
  (h1 : blue = 0.3) 
  (h2 : green = 0.25) 
  (h3 : blue + green + red + purple = 1) : 
  red + purple = 0.45 := by
  sorry

end gumball_probability_l36_3682


namespace cricket_team_average_age_l36_3693

/-- The average age of a cricket team given specific conditions -/
theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (team_average : ℝ),
  team_size = 11 →
  captain_age = 26 →
  wicket_keeper_age_diff = 3 →
  (team_size : ℝ) * team_average = 
    (team_size - 2 : ℝ) * (team_average - 1) + 
    (captain_age : ℝ) + (captain_age + wicket_keeper_age_diff : ℝ) →
  team_average = 32 := by
sorry

end cricket_team_average_age_l36_3693


namespace rogers_money_l36_3632

theorem rogers_money (x : ℝ) : 
  (x + 28 - 25 = 19) → (x = 16) := by
  sorry

end rogers_money_l36_3632


namespace quadratic_inequality_l36_3694

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 5

-- State the theorem
theorem quadratic_inequality (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ < 1) 
  (h₂ : 2 ≤ x₂ ∧ x₂ < 3) 
  (hy₁ : y₁ = f x₁) 
  (hy₂ : y₂ = f x₂) : 
  y₁ ≥ y₂ := by
sorry

end quadratic_inequality_l36_3694


namespace inequality_proof_l36_3692

theorem inequality_proof (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 :=
by sorry

end inequality_proof_l36_3692


namespace power_of_729_l36_3606

theorem power_of_729 : (729 : ℝ) ^ (4/6 : ℝ) = 81 :=
by
  have h : 729 = 3^6 := by sorry
  sorry

end power_of_729_l36_3606


namespace simon_age_proof_l36_3668

/-- Alvin's age in years -/
def alvin_age : ℕ := 30

/-- Simon's age in years -/
def simon_age : ℕ := 10

/-- The difference between half of Alvin's age and Simon's age -/
def age_difference : ℕ := 5

theorem simon_age_proof :
  simon_age = alvin_age / 2 - age_difference :=
by sorry

end simon_age_proof_l36_3668


namespace cubic_fraction_equals_ten_l36_3685

theorem cubic_fraction_equals_ten (a b : ℝ) (ha : a = 7) (hb : b = 3) :
  (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := by
  sorry

end cubic_fraction_equals_ten_l36_3685


namespace cube_inequality_l36_3652

theorem cube_inequality (a b : ℝ) : a^3 > b^3 → a > b := by
  sorry

end cube_inequality_l36_3652


namespace intersection_implies_a_value_l36_3689

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- State the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a) = {x | -2 ≤ x ∧ x ≤ 1} → a = -2 := by
  sorry

end intersection_implies_a_value_l36_3689


namespace limit_of_sequence_a_l36_3674

def a (n : ℕ) : ℚ := (1 + 3 * n) / (6 - n)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-3)| < ε :=
by sorry

end limit_of_sequence_a_l36_3674


namespace usb_available_space_l36_3683

theorem usb_available_space (total_capacity : ℝ) (occupied_percentage : ℝ) 
  (h1 : total_capacity = 128)
  (h2 : occupied_percentage = 75) :
  (1 - occupied_percentage / 100) * total_capacity = 32 := by
  sorry

end usb_available_space_l36_3683


namespace tournament_games_l36_3669

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInSingleElimination (n : ℕ) : ℕ := n - 1

/-- Represents the structure of a two-stage tournament. -/
structure TwoStageTournament where
  totalTeams : ℕ
  firstStageGroups : ℕ
  teamsPerGroup : ℕ
  secondStageTeams : ℕ

/-- Calculates the total number of games in a two-stage tournament. -/
def totalGames (t : TwoStageTournament) : ℕ :=
  (t.firstStageGroups * gamesInSingleElimination t.teamsPerGroup) +
  gamesInSingleElimination t.secondStageTeams

/-- Theorem stating the total number of games in the specific tournament described. -/
theorem tournament_games :
  let t : TwoStageTournament := {
    totalTeams := 24,
    firstStageGroups := 4,
    teamsPerGroup := 6,
    secondStageTeams := 4
  }
  totalGames t = 23 := by sorry

end tournament_games_l36_3669


namespace square_vertex_locus_l36_3649

/-- Represents a line in 2D plane with equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with vertices A, B, C, D and center O -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point
  O : Point

def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem square_vertex_locus 
  (a c o : Line) 
  (h_not_parallel : a.a * c.b ≠ a.b * c.a) :
  ∃ (F G H : ℝ),
    ∀ (ABCD : Square),
      on_line ABCD.A a → 
      on_line ABCD.C c → 
      on_line ABCD.O o → 
      (on_line ABCD.B ⟨F, G, H⟩ ∧ on_line ABCD.D ⟨F, G, H⟩) :=
sorry

end square_vertex_locus_l36_3649


namespace smallest_five_digit_congruent_to_2_mod_37_l36_3667

theorem smallest_five_digit_congruent_to_2_mod_37 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- Five-digit positive integer
  (n ≡ 2 [ZMOD 37]) ∧         -- Congruent to 2 modulo 37
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m ≡ 2 [ZMOD 37]) → n ≤ m) ∧  -- Smallest such number
  n = 10027 :=                -- The number is 10027
by sorry

end smallest_five_digit_congruent_to_2_mod_37_l36_3667


namespace points_on_circle_l36_3604

theorem points_on_circle (t : ℝ) (h : t ≠ 0) :
  ∃ (a : ℝ), (((t^2 + 1) / t)^2 + ((t^2 - 1) / t)^2) = a := by
  sorry

end points_on_circle_l36_3604


namespace diagonal_passes_through_720_cubes_l36_3666

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: In a 180 × 360 × 450 rectangular solid made of unit cubes, 
    an internal diagonal passes through 720 cubes -/
theorem diagonal_passes_through_720_cubes :
  cubes_passed_by_diagonal 180 360 450 = 720 := by
  sorry

end diagonal_passes_through_720_cubes_l36_3666


namespace inequality_implies_not_six_l36_3684

theorem inequality_implies_not_six (m : ℝ) : m + 3 < (-m + 1) - (-13) → m ≠ 6 := by
  sorry

end inequality_implies_not_six_l36_3684


namespace noodle_problem_l36_3672

theorem noodle_problem (x : ℚ) : 
  (2 / 3 : ℚ) * x = 54 → x = 81 := by
  sorry

end noodle_problem_l36_3672


namespace book_pages_sum_l36_3673

/-- A book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- The total number of pages in a book -/
def total_pages (b : Book) : ℕ := b.chapter1_pages + b.chapter2_pages

/-- Theorem: A book with 13 pages in the first chapter and 68 pages in the second chapter has 81 pages in total -/
theorem book_pages_sum : 
  ∀ (b : Book), b.chapter1_pages = 13 ∧ b.chapter2_pages = 68 → total_pages b = 81 := by
  sorry

end book_pages_sum_l36_3673


namespace polygon_with_108_degree_interior_angles_is_pentagon_l36_3676

theorem polygon_with_108_degree_interior_angles_is_pentagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    interior_angle = 108 →
    (n : ℝ) * (180 - interior_angle) = 360 →
    n = 5 :=
by
  sorry

end polygon_with_108_degree_interior_angles_is_pentagon_l36_3676


namespace solution_implies_difference_l36_3678

theorem solution_implies_difference (m n : ℝ) : 
  (m - n = 2) → (n - m = -2) := by sorry

end solution_implies_difference_l36_3678


namespace line_intersects_circle_iff_abs_b_le_sqrt2_l36_3663

/-- The line y=x+b has common points with the circle x²+y²=1 if and only if |b| ≤ √2. -/
theorem line_intersects_circle_iff_abs_b_le_sqrt2 (b : ℝ) : 
  (∃ (x y : ℝ), y = x + b ∧ x^2 + y^2 = 1) ↔ |b| ≤ Real.sqrt 2 := by
  sorry

end line_intersects_circle_iff_abs_b_le_sqrt2_l36_3663


namespace units_digit_not_eight_l36_3601

theorem units_digit_not_eight (a b : Nat) :
  a ∈ Finset.range 100 → b ∈ Finset.range 100 →
  (2^a + 5^b) % 10 ≠ 8 := by
  sorry

end units_digit_not_eight_l36_3601


namespace salary_increase_l36_3627

theorem salary_increase
  (num_employees : ℕ)
  (avg_salary : ℝ)
  (manager_salary : ℝ)
  (h1 : num_employees = 20)
  (h2 : avg_salary = 1500)
  (h3 : manager_salary = 3600) :
  let total_salary := num_employees * avg_salary
  let new_total_salary := total_salary + manager_salary
  let new_avg_salary := new_total_salary / (num_employees + 1)
  new_avg_salary - avg_salary = 100 := by
sorry

end salary_increase_l36_3627


namespace shirt_to_wallet_ratio_l36_3665

/-- The cost of food Mike bought --/
def food_cost : ℚ := 30

/-- The total amount Mike spent on shopping --/
def total_spent : ℚ := 150

/-- The cost of the wallet Mike bought --/
def wallet_cost : ℚ := food_cost + 60

/-- The cost of the shirt Mike bought --/
def shirt_cost : ℚ := total_spent - wallet_cost - food_cost

/-- The theorem stating the ratio of shirt cost to wallet cost --/
theorem shirt_to_wallet_ratio : 
  shirt_cost / wallet_cost = 1 / 3 := by sorry

end shirt_to_wallet_ratio_l36_3665


namespace triangle_side_range_l36_3635

-- Define an acute-angled triangle with side lengths 2, 4, and x
def is_acute_triangle (x : ℝ) : Prop :=
  0 < x ∧ x < 2 + 4 ∧ 2 < 4 + x ∧ 4 < 2 + x ∧
  (2^2 + 4^2 > x^2) ∧ (2^2 + x^2 > 4^2) ∧ (4^2 + x^2 > 2^2)

-- Theorem statement
theorem triangle_side_range :
  ∀ x : ℝ, is_acute_triangle x → (2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 5) :=
by sorry

end triangle_side_range_l36_3635


namespace bales_in_barn_l36_3661

/-- The number of bales in the barn after Tim stacked new bales -/
def total_bales (initial_bales new_bales : ℕ) : ℕ :=
  initial_bales + new_bales

/-- Theorem stating that the total number of bales is 82 -/
theorem bales_in_barn : total_bales 54 28 = 82 := by
  sorry

end bales_in_barn_l36_3661


namespace jimmy_folders_l36_3613

-- Define the variables
def pen_cost : ℕ := 1
def notebook_cost : ℕ := 3
def folder_cost : ℕ := 5
def num_pens : ℕ := 3
def num_notebooks : ℕ := 4
def paid_amount : ℕ := 50
def change_amount : ℕ := 25

-- Define the theorem
theorem jimmy_folders :
  (paid_amount - change_amount - (num_pens * pen_cost + num_notebooks * notebook_cost)) / folder_cost = 2 :=
by sorry

end jimmy_folders_l36_3613


namespace rent_percentage_is_seven_percent_l36_3624

/-- Proves that the percentage of monthly earnings spent on rent is 7% -/
theorem rent_percentage_is_seven_percent (monthly_earnings : ℝ) 
  (rent_amount : ℝ) (savings_amount : ℝ) :
  rent_amount = 133 →
  savings_amount = 817 →
  monthly_earnings = rent_amount + savings_amount + (monthly_earnings / 2) →
  (rent_amount / monthly_earnings) * 100 = 7 := by
sorry

end rent_percentage_is_seven_percent_l36_3624


namespace g_of_2_eq_8_l36_3651

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def g (x : ℝ) : ℝ := 1 / (f.invFun x) + 7

theorem g_of_2_eq_8 : g 2 = 8 := by sorry

end g_of_2_eq_8_l36_3651


namespace square_difference_quotient_l36_3600

theorem square_difference_quotient : (245^2 - 205^2) / 40 = 450 := by
  sorry

end square_difference_quotient_l36_3600


namespace ship_speed_comparison_ship_time_comparison_l36_3664

/-- Prove that the harmonic mean of two speeds is less than their arithmetic mean -/
theorem ship_speed_comparison 
  (distance : ℝ) 
  (speed_forward : ℝ) 
  (speed_return : ℝ) 
  (h1 : 0 < distance)
  (h2 : 0 < speed_forward)
  (h3 : 0 < speed_return)
  (h4 : speed_forward ≠ speed_return) :
  (2 * speed_forward * speed_return) / (speed_forward + speed_return) < 
  (speed_forward + speed_return) / 2 := by
  sorry

/-- Prove that a ship with varying speeds takes longer than a ship with constant average speed -/
theorem ship_time_comparison 
  (distance : ℝ) 
  (speed_forward : ℝ) 
  (speed_return : ℝ) 
  (h1 : 0 < distance)
  (h2 : 0 < speed_forward)
  (h3 : 0 < speed_return)
  (h4 : speed_forward ≠ speed_return) :
  (2 * distance) / ((2 * speed_forward * speed_return) / (speed_forward + speed_return)) > 
  (2 * distance) / ((speed_forward + speed_return) / 2) := by
  sorry

end ship_speed_comparison_ship_time_comparison_l36_3664


namespace unique_intersecting_line_l36_3691

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne : point1 ≠ point2

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- A line intersects another line -/
def intersects (l1 l2 : Line3D) : Prop :=
  -- Definition of line intersection
  sorry

theorem unique_intersecting_line (a b c : Line3D) 
  (hab : are_skew a b) (hbc : are_skew b c) (hac : are_skew a c) :
  ∃! l : Line3D, intersects l a ∧ intersects l b ∧ intersects l c :=
sorry

end unique_intersecting_line_l36_3691


namespace quadratic_inequality_range_l36_3654

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) → (0 ≤ a ∧ a < 1) := by
  sorry

end quadratic_inequality_range_l36_3654


namespace grain_storage_capacity_l36_3687

theorem grain_storage_capacity (total_bins : ℕ) (large_bin_capacity : ℕ) (total_capacity : ℕ) (num_large_bins : ℕ) :
  total_bins = 30 →
  large_bin_capacity = 20 →
  total_capacity = 510 →
  num_large_bins = 12 →
  ∃ (small_bin_capacity : ℕ),
    small_bin_capacity * (total_bins - num_large_bins) + large_bin_capacity * num_large_bins = total_capacity ∧
    small_bin_capacity = 15 :=
by
  sorry

end grain_storage_capacity_l36_3687


namespace smallest_factorizable_b_l36_3630

theorem smallest_factorizable_b : ∃ (b : ℕ),
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 2016 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), b' < b →
    ¬(∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b'*x + 2016 = (x + p) * (x + q))) ∧
  b = 90 := by
sorry

end smallest_factorizable_b_l36_3630


namespace three_letter_sets_count_l36_3636

/-- The number of permutations of k elements chosen from a set of n distinct elements -/
def permutations (n k : ℕ) : ℕ := sorry

/-- The number of letters available (A through J) -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def set_size : ℕ := 3

theorem three_letter_sets_count : permutations num_letters set_size = 720 := by
  sorry

end three_letter_sets_count_l36_3636


namespace bruno_pens_l36_3618

/-- The number of pens in a dozen -/
def pens_per_dozen : ℕ := 12

/-- The total number of pens Bruno will have -/
def total_pens : ℕ := 30

/-- The number of dozens of pens Bruno wants to buy -/
def dozens_to_buy : ℚ := total_pens / pens_per_dozen

/-- Theorem stating that Bruno wants to buy 2.5 dozens of pens -/
theorem bruno_pens : dozens_to_buy = 5/2 := by sorry

end bruno_pens_l36_3618


namespace darla_books_count_l36_3658

/-- Proves that Darla has 6 books given the conditions of the problem -/
theorem darla_books_count :
  ∀ (d k g : ℕ),
  k = d / 2 →
  g = 5 * (d + k) →
  d + k + g = 54 →
  d = 6 :=
by sorry

end darla_books_count_l36_3658


namespace sector_area_l36_3621

/-- The area of a circular sector with central angle 72° and radius 5 is 5π. -/
theorem sector_area (S : ℝ) : S = 5 * Real.pi := by
  -- Given:
  -- Central angle is 72°
  -- Radius is 5
  sorry

end sector_area_l36_3621


namespace horner_method_result_l36_3615

def f (x : ℝ) : ℝ := 9 + 15*x - 8*x^2 - 20*x^3 + 6*x^4 + 3*x^5

theorem horner_method_result : f 4 = 3269 := by
  sorry

end horner_method_result_l36_3615


namespace parallel_vectors_k_value_l36_3696

def vector_a (k : ℝ) : Fin 2 → ℝ := ![1, k]
def vector_b : Fin 2 → ℝ := ![-2, 6]

theorem parallel_vectors_k_value :
  (∃ (c : ℝ), c ≠ 0 ∧ (∀ i, vector_a k i = c * vector_b i)) →
  k = -3 :=
by
  sorry

end parallel_vectors_k_value_l36_3696


namespace triangle_division_into_congruent_parts_l36_3648

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : c > 0)
  (h₄ : a + b > c)
  (h₅ : b + c > a)
  (h₆ : c + a > b)

-- Define congruence for triangles
def CongruentTriangles (t₁ t₂ : Triangle) : Prop :=
  t₁.a = t₂.a ∧ t₁.b = t₂.b ∧ t₁.c = t₂.c

-- Define a division of a triangle into five smaller triangles
structure TriangleDivision (t : Triangle) :=
  (t₁ t₂ t₃ t₄ t₅ : Triangle)

-- State the theorem
theorem triangle_division_into_congruent_parts (t : Triangle) :
  ∃ (d : TriangleDivision t), 
    CongruentTriangles d.t₁ d.t₂ ∧
    CongruentTriangles d.t₁ d.t₃ ∧
    CongruentTriangles d.t₁ d.t₄ ∧
    CongruentTriangles d.t₁ d.t₅ :=
sorry

end triangle_division_into_congruent_parts_l36_3648


namespace least_subtraction_for_divisibility_problem_solution_l36_3697

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 14 ∧ (10154 - x) % 30 = 0 ∧ ∀ (y : ℕ), y < x → (10154 - y) % 30 ≠ 0 :=
sorry

end least_subtraction_for_divisibility_problem_solution_l36_3697


namespace solution_value_l36_3612

theorem solution_value (a b : ℝ) : 
  (a * 1 - b * 2 + 3 = 0) →
  (a * (-1) - b * 1 + 3 = 0) →
  a - 3 * b = -5 := by
sorry

end solution_value_l36_3612


namespace parabola_directrix_l36_3603

/-- Given a parabola y² = 2px and a point M(1, m) on it, 
    if the distance from M to its focus is 5, 
    then the equation of its directrix is x = -4 -/
theorem parabola_directrix (p : ℝ) (m : ℝ) :
  m^2 = 2*p  -- Point M(1, m) is on the parabola y² = 2px
  → (1 - p/2)^2 + m^2 = 5^2  -- Distance from M to focus is 5
  → (-p/2 : ℝ) = -4  -- Equation of directrix is x = -4
:= by sorry

end parabola_directrix_l36_3603


namespace last_digit_of_3_power_2012_l36_3662

/-- The last digit of 3^n for any natural number n -/
def lastDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem last_digit_of_3_power_2012 :
  lastDigitOf3Power 2012 = 1 := by
  sorry

end last_digit_of_3_power_2012_l36_3662


namespace diagonal_intersection_coincidence_l36_3660

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by its four vertices -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Predicate to check if a quadrilateral is circumscribed around a circle -/
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Function to get the tangency points of a circumscribed quadrilateral -/
def tangency_points (q : Quadrilateral) (c : Circle) : 
  (Point × Point × Point × Point) := sorry

/-- Function to get the intersection point of two diagonals -/
def diagonal_intersection (q : Quadrilateral) : Point := sorry

/-- The main theorem -/
theorem diagonal_intersection_coincidence 
  (q : Quadrilateral) (c : Circle) 
  (h : is_circumscribed q c) : 
  let (E, F, G, K) := tangency_points q c
  let q' := Quadrilateral.mk E F G K
  diagonal_intersection q = diagonal_intersection q' := by sorry

end diagonal_intersection_coincidence_l36_3660


namespace partnership_profit_l36_3625

/-- Represents the profit share of a partner in a business partnership. -/
structure ProfitShare where
  investment : ℕ
  share : ℕ

/-- Calculates the total profit of a partnership business given the investments and one partner's profit share. -/
def totalProfit (a b c : ProfitShare) : ℕ :=
  sorry

/-- Theorem stating that given the specific investments and A's profit share, the total profit is 12500. -/
theorem partnership_profit (a b c : ProfitShare) 
  (ha : a.investment = 6300)
  (hb : b.investment = 4200)
  (hc : c.investment = 10500)
  (ha_share : a.share = 3750) :
  totalProfit a b c = 12500 := by
  sorry

end partnership_profit_l36_3625


namespace not_odd_implies_exists_neq_l36_3609

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem not_odd_implies_exists_neq (f : ℝ → ℝ) (h : ¬IsOdd f) : 
  ∃ x, f (-x) ≠ -f x := by
  sorry

end not_odd_implies_exists_neq_l36_3609


namespace megans_work_hours_l36_3629

/-- Megan's work problem -/
theorem megans_work_hours
  (hourly_rate : ℝ)
  (days_per_month : ℕ)
  (total_earnings : ℝ)
  (h : hourly_rate = 7.5)
  (d : days_per_month = 20)
  (e : total_earnings = 2400) :
  ∃ (hours_per_day : ℝ),
    hours_per_day * hourly_rate * (2 * days_per_month) = total_earnings ∧
    hours_per_day = 8 := by
  sorry

end megans_work_hours_l36_3629


namespace sphere_radius_equal_cylinder_surface_area_l36_3640

/-- The radius of a sphere given that its surface area is equal to the curved surface area of a right circular cylinder with height and diameter both 6 cm -/
theorem sphere_radius_equal_cylinder_surface_area (h : ℝ) (d : ℝ) (r : ℝ) : 
  h = 6 →
  d = 6 →
  4 * Real.pi * r^2 = 2 * Real.pi * (d/2) * h →
  r = 3 :=
by sorry

end sphere_radius_equal_cylinder_surface_area_l36_3640


namespace interior_triangle_area_l36_3681

theorem interior_triangle_area (a b c : ℝ) (ha : a = 16) (hb : b = 324) (hc : c = 100) :
  (1/2 : ℝ) * Real.sqrt a * Real.sqrt b = 36 := by
  sorry

end interior_triangle_area_l36_3681


namespace product_of_three_numbers_l36_3626

theorem product_of_three_numbers (a b c : ℚ) : 
  a + b + c = 30 →
  a = 3 * (b + c) →
  b = 6 * c →
  a * b * c = 10125 / 14 := by
sorry

end product_of_three_numbers_l36_3626


namespace bananas_permutations_l36_3619

/-- The number of distinct permutations of a word with repeated letters -/
def permutationsWithRepeats (total : ℕ) (repeats : List ℕ) : ℕ :=
  Nat.factorial total / (repeats.map Nat.factorial).prod

/-- The word "BANANAS" has 7 letters -/
def totalLetters : ℕ := 7

/-- The repetition pattern of letters in "BANANAS" -/
def letterRepeats : List ℕ := [3, 2]  -- 3 'A's and 2 'N's

theorem bananas_permutations :
  permutationsWithRepeats totalLetters letterRepeats = 420 := by
  sorry


end bananas_permutations_l36_3619


namespace jerry_cases_jerry_cases_proof_l36_3628

/-- The number of cases Jerry has, given the following conditions:
  - Each case has 3 shelves
  - Each shelf can hold 20 records
  - Each vinyl record has 60 ridges
  - The shelves are 60% full
  - There are 8640 ridges on all records
-/
theorem jerry_cases : ℕ :=
  let shelves_per_case : ℕ := 3
  let records_per_shelf : ℕ := 20
  let ridges_per_record : ℕ := 60
  let shelf_fullness : ℚ := 3/5
  let total_ridges : ℕ := 8640
  
  4

/-- Proof that Jerry has 4 cases -/
theorem jerry_cases_proof : jerry_cases = 4 := by
  sorry

end jerry_cases_jerry_cases_proof_l36_3628


namespace max_blocks_fit_l36_3638

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- The dimensions of the small block -/
def smallBlock : Dimensions := ⟨3, 2, 1⟩

/-- The dimensions of the box -/
def box : Dimensions := ⟨4, 6, 2⟩

/-- The maximum number of small blocks that can fit in the box -/
def maxBlocks : ℕ := 8

theorem max_blocks_fit :
  volume box / volume smallBlock = maxBlocks ∧
  maxBlocks * volume smallBlock ≤ volume box ∧
  (maxBlocks + 1) * volume smallBlock > volume box :=
sorry

end max_blocks_fit_l36_3638


namespace ratio_odd_even_divisors_N_l36_3656

/-- The number N as defined in the problem -/
def N : ℕ := 46 * 46 * 81 * 450

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the ratio of sum of odd divisors to sum of even divisors of N -/
theorem ratio_odd_even_divisors_N :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 14 := by sorry

end ratio_odd_even_divisors_N_l36_3656


namespace probability_ellipse_x_foci_value_l36_3645

/-- The probability that x²/m² + y²/n² = 1 represents an ellipse with foci on the x-axis,
    given m ∈ [1,5] and n ∈ [2,4] -/
def probability_ellipse_x_foci (m n : ℝ) : ℝ :=
  sorry

/-- Theorem stating the probability is equal to some value P -/
theorem probability_ellipse_x_foci_value :
  ∃ P, ∀ m n, m ∈ Set.Icc 1 5 → n ∈ Set.Icc 2 4 →
    probability_ellipse_x_foci m n = P :=
  sorry

end probability_ellipse_x_foci_value_l36_3645


namespace girls_to_boys_ratio_l36_3659

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) : 
  total = 36 → difference = 6 → 
  ∃ (girls boys : ℕ), 
    girls + boys = total ∧ 
    girls = boys + difference ∧
    girls * 5 = boys * 7 := by
  sorry

end girls_to_boys_ratio_l36_3659


namespace polynomial_division_theorem_l36_3690

theorem polynomial_division_theorem (x : ℝ) :
  (x - 2) * (x^5 + 2*x^4 + 4*x^3 + 13*x^2 + 26*x + 52) + 96 = x^6 + 5*x^3 - 8 := by
  sorry

end polynomial_division_theorem_l36_3690


namespace circle_equation_l36_3653

/-- The equation of a circle with center (-2, 1) passing through the point (2, -2) -/
theorem circle_equation :
  let center : ℝ × ℝ := (-2, 1)
  let point : ℝ × ℝ := (2, -2)
  ∀ x y : ℝ,
  (x - center.1)^2 + (y - center.2)^2 = (point.1 - center.1)^2 + (point.2 - center.2)^2 ↔
  (x + 2)^2 + (y - 1)^2 = 25 :=
by sorry

end circle_equation_l36_3653


namespace probability_of_correct_dial_l36_3688

def first_three_digits : ℕ := 3
def last_four_digits : ℕ := 24
def total_combinations : ℕ := first_three_digits * last_four_digits
def correct_numbers : ℕ := 1

theorem probability_of_correct_dial :
  (correct_numbers : ℚ) / total_combinations = 1 / 72 := by
  sorry

end probability_of_correct_dial_l36_3688


namespace not_divisible_by_4p_l36_3620

theorem not_divisible_by_4p (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ¬ (4 * p ∣ (2 * p - 1)^(p - 1) + 1) := by
  sorry

end not_divisible_by_4p_l36_3620


namespace problem_statement_l36_3699

theorem problem_statement (x : ℝ) : 3 * x - 1 = 8 → 150 * (1 / x) + 2 = 52 := by
  sorry

end problem_statement_l36_3699


namespace carolyn_practice_days_l36_3650

/-- Represents the practice schedule of a musician --/
structure PracticeSchedule where
  piano_time : ℕ  -- Daily piano practice time in minutes
  violin_ratio : ℕ  -- Ratio of violin practice time to piano practice time
  total_monthly_time : ℕ  -- Total practice time in a month (in minutes)
  weeks_in_month : ℕ  -- Number of weeks in a month

/-- Calculates the number of practice days per week --/
def practice_days_per_week (schedule : PracticeSchedule) : ℚ :=
  let daily_total := schedule.piano_time * (1 + schedule.violin_ratio)
  let monthly_days := schedule.total_monthly_time / daily_total
  monthly_days / schedule.weeks_in_month

/-- Theorem stating that Carolyn practices 6 days a week --/
theorem carolyn_practice_days (schedule : PracticeSchedule) 
  (h1 : schedule.piano_time = 20)
  (h2 : schedule.violin_ratio = 3)
  (h3 : schedule.total_monthly_time = 1920)
  (h4 : schedule.weeks_in_month = 4) :
  practice_days_per_week schedule = 6 := by
  sorry

#eval practice_days_per_week ⟨20, 3, 1920, 4⟩

end carolyn_practice_days_l36_3650


namespace washing_machines_removed_l36_3634

theorem washing_machines_removed (
  num_containers : ℕ) (crates_per_container : ℕ) (boxes_per_crate : ℕ)
  (machines_per_box : ℕ) (num_workers : ℕ) (machines_removed_per_box : ℕ)
  (h1 : num_containers = 100)
  (h2 : crates_per_container = 30)
  (h3 : boxes_per_crate = 15)
  (h4 : machines_per_box = 10)
  (h5 : num_workers = 6)
  (h6 : machines_removed_per_box = 4)
  : (num_containers * crates_per_container * boxes_per_crate * machines_removed_per_box * num_workers) = 180000 := by
  sorry

#check washing_machines_removed

end washing_machines_removed_l36_3634


namespace factorial_ten_base_twelve_zeros_l36_3637

theorem factorial_ten_base_twelve_zeros (n : ℕ) (h : n = 10) :
  ∃ k : ℕ, k = 4 ∧ 12^k ∣ n! ∧ ¬(12^(k+1) ∣ n!) :=
sorry

end factorial_ten_base_twelve_zeros_l36_3637


namespace max_sqrt_sum_l36_3680

theorem max_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (x + 31) + Real.sqrt (17 - x) + Real.sqrt x ≤ 12 ∧
  ∃ x₀, x₀ = 13 ∧ Real.sqrt (x₀ + 31) + Real.sqrt (17 - x₀) + Real.sqrt x₀ = 12 :=
by sorry

end max_sqrt_sum_l36_3680


namespace fourth_side_length_l36_3677

/-- A quadrilateral inscribed in a circle with radius 150√3, where three sides are 150 units long -/
structure InscribedQuadrilateral where
  -- The radius of the circle
  r : ℝ
  -- The lengths of the four sides of the quadrilateral
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  s₄ : ℝ
  -- Conditions
  h_radius : r = 150 * Real.sqrt 3
  h_three_sides : s₁ = 150 ∧ s₂ = 150 ∧ s₃ = 150

/-- The theorem stating that the fourth side of the quadrilateral is 450 units long -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.s₄ = 450 := by
  sorry

end fourth_side_length_l36_3677


namespace point_in_fourth_quadrant_l36_3614

theorem point_in_fourth_quadrant (a b : ℝ) : 
  let A : ℝ × ℝ := (a^2 + 1, -1 - b^2)
  A.1 > 0 ∧ A.2 < 0 :=
by sorry

end point_in_fourth_quadrant_l36_3614


namespace consecutive_integers_sum_divisible_by_12_l36_3639

theorem consecutive_integers_sum_divisible_by_12 (a b c d : ℤ) :
  (b = a + 1) → (c = b + 1) → (d = c + 1) →
  ∃ k : ℤ, ab + ac + ad + bc + bd + cd + 1 = 12 * k :=
by
  sorry
where
  ab := a * b
  ac := a * c
  ad := a * d
  bc := b * c
  bd := b * d
  cd := c * d

end consecutive_integers_sum_divisible_by_12_l36_3639


namespace restaurant_sales_tax_rate_l36_3657

theorem restaurant_sales_tax_rate 
  (total_bill : ℝ) 
  (striploin_cost : ℝ) 
  (wine_cost : ℝ) 
  (gratuities : ℝ) 
  (h1 : total_bill = 140)
  (h2 : striploin_cost = 80)
  (h3 : wine_cost = 10)
  (h4 : gratuities = 41) :
  (total_bill - striploin_cost - wine_cost - gratuities) / (striploin_cost + wine_cost) = 0.1 := by
  sorry

end restaurant_sales_tax_rate_l36_3657


namespace walking_time_calculation_l36_3671

/-- Given a man who walks and runs at different speeds, this theorem proves
    the time taken to walk a distance that he can run in 1.5 hours. -/
theorem walking_time_calculation (walk_speed run_speed : ℝ) (run_time : ℝ) 
    (h1 : walk_speed = 8)
    (h2 : run_speed = 16)
    (h3 : run_time = 1.5) : 
  (run_speed * run_time) / walk_speed = 3 := by
  sorry

end walking_time_calculation_l36_3671


namespace beth_shopping_theorem_l36_3679

def cans_of_peas : ℕ := 35

def cans_of_corn : ℕ := 10

theorem beth_shopping_theorem :
  cans_of_peas = 2 * cans_of_corn + 15 ∧ cans_of_corn = 10 := by
  sorry

end beth_shopping_theorem_l36_3679


namespace jamies_coins_l36_3622

/-- Represents the number of coins of each type -/
structure CoinCounts where
  quarters : ℚ
  nickels : ℚ
  dimes : ℚ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCounts) : ℚ :=
  25 * coins.quarters + 5 * coins.nickels + 10 * coins.dimes

/-- Theorem stating the solution to Jamie's coin problem -/
theorem jamies_coins :
  ∃ (coins : CoinCounts),
    coins.nickels = 2 * coins.quarters ∧
    coins.dimes = coins.quarters ∧
    totalValue coins = 1520 ∧
    coins.quarters = 304/9 ∧
    coins.nickels = 608/9 ∧
    coins.dimes = 304/9 := by
  sorry

end jamies_coins_l36_3622


namespace coffee_maker_price_l36_3616

def original_price (sale_price : ℝ) (discount : ℝ) : ℝ :=
  sale_price + discount

theorem coffee_maker_price :
  let sale_price : ℝ := 70
  let discount : ℝ := 20
  original_price sale_price discount = 90 :=
by sorry

end coffee_maker_price_l36_3616


namespace contrapositive_equivalence_l36_3686

/-- The proposition "If a and b are both even, then the sum of a and b is even" -/
def original_proposition (a b : ℤ) : Prop :=
  (Even a ∧ Even b) → Even (a + b)

/-- The contrapositive of the original proposition -/
def contrapositive (a b : ℤ) : Prop :=
  ¬Even (a + b) → ¬(Even a ∧ Even b)

/-- Theorem stating that the contrapositive is equivalent to "If the sum of a and b is not even, then a and b are not both even" -/
theorem contrapositive_equivalence :
  ∀ a b : ℤ, contrapositive a b ↔ (¬Even (a + b) → ¬(Even a ∧ Even b)) :=
by sorry

end contrapositive_equivalence_l36_3686


namespace prime_sum_theorem_l36_3623

theorem prime_sum_theorem (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h_eq : 2 * p + 3 * q = 6 * r) : p + q + r = 7 := by
  sorry

end prime_sum_theorem_l36_3623
