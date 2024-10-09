import Mathlib

namespace find_vidya_age_l2125_212590

theorem find_vidya_age (V M : ℕ) (h1: M = 3 * V + 5) (h2: M = 44) : V = 13 :=
by {
  sorry
}

end find_vidya_age_l2125_212590


namespace largest_non_representable_integer_l2125_212511

theorem largest_non_representable_integer (n a b : ℕ) (h₁ : n = 42 * a + b)
  (h₂ : 0 ≤ b) (h₃ : b < 42) (h₄ : ¬ (b % 6 = 0)) :
  n ≤ 252 :=
sorry

end largest_non_representable_integer_l2125_212511


namespace symmetry_proof_l2125_212586

-- Define the coordinates of point A
def A : ℝ × ℝ := (-1, 8)

-- Define the reflection property across the y-axis
def is_reflection_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

-- Define the point B which we need to prove
def B : ℝ × ℝ := (1, 8)

-- The proof statement
theorem symmetry_proof :
  is_reflection_y_axis A B :=
by
  sorry

end symmetry_proof_l2125_212586


namespace solve_equation1_solve_equation2_l2125_212574

-- Define the first equation and state the theorem that proves its roots
def equation1 (x : ℝ) : Prop := 2 * x^2 + 1 = 3 * x

theorem solve_equation1 (x : ℝ) : equation1 x ↔ (x = 1 ∨ x = 1/2) :=
by sorry

-- Define the second equation and state the theorem that proves its roots
def equation2 (x : ℝ) : Prop := (2 * x - 1)^2 = (3 - x)^2

theorem solve_equation2 (x : ℝ) : equation2 x ↔ (x = -2 ∨ x = 4 / 3) :=
by sorry

end solve_equation1_solve_equation2_l2125_212574


namespace find_b10_l2125_212519

def seq (b : ℕ → ℕ) :=
  (b 1 = 2)
  ∧ (∀ m n, b (m + n) = b m + b n + 2 * m * n)

theorem find_b10 (b : ℕ → ℕ) (h : seq b) : b 10 = 110 :=
by 
  -- Proof omitted, as requested.
  sorry

end find_b10_l2125_212519


namespace friends_gift_l2125_212502

-- Define the original number of balloons and the final number of balloons
def original_balloons := 8
def final_balloons := 10

-- The main theorem: Joan's friend gave her 2 orange balloons.
theorem friends_gift : (final_balloons - original_balloons) = 2 := by
  sorry

end friends_gift_l2125_212502


namespace sufficient_but_not_necessary_not_necessary_l2125_212530

-- Conditions
def condition_1 (x : ℝ) : Prop := x > 3
def condition_2 (x : ℝ) : Prop := x^2 - 5 * x + 6 > 0

-- Theorem statement
theorem sufficient_but_not_necessary (x : ℝ) : condition_1 x → condition_2 x :=
sorry

theorem not_necessary (x : ℝ) : condition_2 x → ∃ y : ℝ, ¬ condition_1 y ∧ condition_2 y :=
sorry

end sufficient_but_not_necessary_not_necessary_l2125_212530


namespace minimum_people_l2125_212558

def num_photos : ℕ := 10
def num_center_men : ℕ := 10
def num_people_per_photo : ℕ := 3

theorem minimum_people (n : ℕ) (h : n = num_photos) :
  (∃ total_people, total_people = 16) :=
sorry

end minimum_people_l2125_212558


namespace ratio_of_members_l2125_212553

theorem ratio_of_members (f m c : ℕ) 
  (h1 : (35 * f + 30 * m + 10 * c) / (f + m + c) = 25) :
  2 * f + m = 3 * c :=
by
  sorry

end ratio_of_members_l2125_212553


namespace angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l2125_212518

-- Define the concept of a cube and the diagonals of its faces.
structure Cube :=
  (faces : Fin 6 → (Fin 4 → ℝ × ℝ × ℝ))    -- Representing each face as a set of four vertices in 3D space

def is_square_face (face : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  -- A function that checks if a given set of four vertices forms a square face.
  sorry

def are_adjacent_faces_perpendicular_diagonals 
  (face1 face2 : Fin 4 → ℝ × ℝ × ℝ) (c : Cube) : Prop :=
  -- A function that checks if the diagonals of two given adjacent square faces of a cube are perpendicular.
  sorry

-- The theorem stating the required proof:
theorem angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees
  (c : Cube)
  (h1 : is_square_face (c.faces 0))
  (h2 : is_square_face (c.faces 1))
  (h_adj: are_adjacent_faces_perpendicular_diagonals (c.faces 0) (c.faces 1) c) :
  ∃ q : ℝ, q = 90 :=
by
  sorry

end angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l2125_212518


namespace difference_of_numbers_l2125_212587

noncomputable def larger_num : ℕ := 1495
noncomputable def quotient : ℕ := 5
noncomputable def remainder : ℕ := 4

theorem difference_of_numbers :
  ∃ S : ℕ, larger_num = quotient * S + remainder ∧ (larger_num - S = 1197) :=
by 
  sorry

end difference_of_numbers_l2125_212587


namespace smallest_a_no_inverse_mod_72_90_l2125_212515

theorem smallest_a_no_inverse_mod_72_90 :
  ∃ (a : ℕ), a > 0 ∧ ∀ b : ℕ, (b > 0 → gcd b 72 > 1 ∧ gcd b 90 > 1 → b ≥ a) ∧ gcd a 72 > 1 ∧ gcd a 90 > 1 ∧ a = 6 :=
by sorry

end smallest_a_no_inverse_mod_72_90_l2125_212515


namespace tangent_eq_tangent_intersect_other_l2125_212560

noncomputable def curve (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

/-- Equation of the tangent line to curve C at x = 1 is y = -12x + 8 --/
theorem tangent_eq (tangent_line : ℝ → ℝ) (x : ℝ):
  tangent_line x = -12 * x + 8 :=
by
  sorry

/-- Apart from the tangent point (1, -4), the tangent line intersects the curve C at the points
    (-2, 32) and (2 / 3, 0) --/
theorem tangent_intersect_other (tangent_line : ℝ → ℝ) x:
  curve x = tangent_line x →
  (x = -2 ∧ curve (-2) = 32) ∨ (x = 2 / 3 ∧ curve (2 / 3) = 0) :=
by
  sorry

end tangent_eq_tangent_intersect_other_l2125_212560


namespace nancy_carrots_next_day_l2125_212526

-- Definitions based on conditions
def carrots_picked_on_first_day : Nat := 12
def carrots_thrown_out : Nat := 2
def total_carrots_after_two_days : Nat := 31

-- Problem statement
theorem nancy_carrots_next_day :
  let carrots_left_after_first_day := carrots_picked_on_first_day - carrots_thrown_out
  let carrots_picked_next_day := total_carrots_after_two_days - carrots_left_after_first_day
  carrots_picked_next_day = 21 :=
by
  sorry

end nancy_carrots_next_day_l2125_212526


namespace rowan_distance_downstream_l2125_212575

-- Conditions
def speed_still : ℝ := 9.75
def downstream_time : ℝ := 2
def upstream_time : ℝ := 4

-- Statement to prove
theorem rowan_distance_downstream : ∃ (d : ℝ) (c : ℝ), 
  d / (speed_still + c) = downstream_time ∧
  d / (speed_still - c) = upstream_time ∧
  d = 26 := by
    sorry

end rowan_distance_downstream_l2125_212575


namespace calculate_new_measure_l2125_212578

noncomputable def equilateral_triangle_side_length : ℝ := 7.5

theorem calculate_new_measure :
  3 * (equilateral_triangle_side_length ^ 2) = 168.75 :=
by
  sorry

end calculate_new_measure_l2125_212578


namespace product_approximation_l2125_212520

theorem product_approximation :
  (3.05 * 7.95 * (6.05 + 3.95)) = 240 := by
  sorry

end product_approximation_l2125_212520


namespace apples_left_l2125_212592

def Mike_apples : ℝ := 7.0
def Nancy_apples : ℝ := 3.0
def Keith_ate_apples : ℝ := 6.0

theorem apples_left : Mike_apples + Nancy_apples - Keith_ate_apples = 4.0 := by
  sorry

end apples_left_l2125_212592


namespace find_f_one_l2125_212500

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f_defined_for_neg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x = 2 * x^2 - 1

-- Statement that needs to be proven
theorem find_f_one (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : f_defined_for_neg f) :
  f 1 = -1 :=
  sorry

end find_f_one_l2125_212500


namespace sum_common_divisors_60_18_l2125_212534

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m => n % m = 0)

noncomputable def sum (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_common_divisors_60_18 :
  sum (List.filter (λ d => d ∈ divisors 18) (divisors 60)) = 12 :=
by
  sorry

end sum_common_divisors_60_18_l2125_212534


namespace geometric_sequence_a4_a5_l2125_212546

open BigOperators

theorem geometric_sequence_a4_a5 (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 9) : 
  a 4 + a 5 = 27 ∨ a 4 + a 5 = -27 :=
sorry

end geometric_sequence_a4_a5_l2125_212546


namespace distinct_points_count_l2125_212595

-- Definitions based on conditions
def eq1 (x y : ℝ) : Prop := (x + y = 7) ∨ (2 * x - 3 * y = -7)
def eq2 (x y : ℝ) : Prop := (x - y = 3) ∨ (3 * x + 2 * y = 18)

-- The statement combining conditions and requiring the proof of 3 distinct solutions
theorem distinct_points_count : 
  ∃! p1 p2 p3 : ℝ × ℝ, 
    (eq1 p1.1 p1.2 ∧ eq2 p1.1 p1.2) ∧ 
    (eq1 p2.1 p2.2 ∧ eq2 p2.1 p2.2) ∧ 
    (eq1 p3.1 p3.2 ∧ eq2 p3.1 p3.2) ∧ 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 :=
sorry

end distinct_points_count_l2125_212595


namespace parabola_directrix_l2125_212567

theorem parabola_directrix (y x : ℝ) : y^2 = -8 * x → x = -1 :=
by
  sorry

end parabola_directrix_l2125_212567


namespace cos_double_angle_of_parallel_vectors_l2125_212516

theorem cos_double_angle_of_parallel_vectors
  (α : ℝ)
  (a : ℝ × ℝ := (1/3, Real.tan α))
  (b : ℝ × ℝ := (Real.cos α, 2))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  Real.cos (2 * α) = 1 / 9 := 
sorry

end cos_double_angle_of_parallel_vectors_l2125_212516


namespace bowling_team_score_ratio_l2125_212577

theorem bowling_team_score_ratio :
  ∀ (F S T : ℕ),
  F + S + T = 810 →
  F = (1 / 3 : ℚ) * S →
  T = 162 →
  S / T = 3 := 
by
  intros F S T h1 h2 h3
  sorry

end bowling_team_score_ratio_l2125_212577


namespace solve_for_r_l2125_212525

variable (n : ℝ) (r : ℝ)

theorem solve_for_r (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : r = (n * (1 + Real.sqrt 3)) / 2 :=
by
  sorry

end solve_for_r_l2125_212525


namespace son_father_age_sum_l2125_212564

theorem son_father_age_sum
    (S F : ℕ)
    (h1 : F - 6 = 3 * (S - 6))
    (h2 : F = 2 * S) :
    S + F = 36 :=
sorry

end son_father_age_sum_l2125_212564


namespace product_of_differing_inputs_equal_l2125_212585

theorem product_of_differing_inputs_equal (a b : ℝ) (h₁ : a ≠ b)
(h₂ : |Real.log a - (1 / 2)| = |Real.log b - (1 / 2)|) : a * b = Real.exp 1 :=
sorry

end product_of_differing_inputs_equal_l2125_212585


namespace circle_intersects_cells_l2125_212509

/-- On a grid with 1 cm x 1 cm cells, a circle with a radius of 100 cm is drawn.
    The circle does not pass through any vertices of the cells and does not touch the sides of the cells.
    Prove that the number of cells the circle can intersect is either 800 or 799. -/
theorem circle_intersects_cells (r : ℝ) (gsize : ℝ) (cells : ℕ) :
  r = 100 ∧ gsize = 1 ∧ cells = 800 ∨ cells = 799 :=
by
  sorry

end circle_intersects_cells_l2125_212509


namespace triple_square_side_area_l2125_212549

theorem triple_square_side_area (s : ℝ) : (3 * s) ^ 2 ≠ 3 * (s ^ 2) :=
by {
  sorry
}

end triple_square_side_area_l2125_212549


namespace max_absolute_difference_l2125_212543

theorem max_absolute_difference (a b c d e : ℤ) (p : ℤ) :
  0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ 100 ∧ p = (a + b + c + d + e) / 5 →
  (|p - c| ≤ 40) :=
by
  sorry

end max_absolute_difference_l2125_212543


namespace speed_conversion_l2125_212529

theorem speed_conversion (speed_kmph : ℝ) (h : speed_kmph = 18) : speed_kmph * (1000 / 3600) = 5 := by
  sorry

end speed_conversion_l2125_212529


namespace TV_height_l2125_212537

theorem TV_height (area : ℝ) (width : ℝ) (height : ℝ) (h1 : area = 21) (h2 : width = 3) : height = 7 :=
  by
  sorry

end TV_height_l2125_212537


namespace tire_circumference_l2125_212536

theorem tire_circumference 
  (rev_per_min : ℝ) -- revolutions per minute
  (car_speed_kmh : ℝ) -- car speed in km/h
  (conversion_factor : ℝ) -- conversion factor for speed from km/h to m/min
  (min_to_meter : ℝ) -- multiplier to convert minutes to meters
  (C : ℝ) -- circumference of the tire in meters
  : rev_per_min = 400 ∧ car_speed_kmh = 120 ∧ conversion_factor = 1000 / 60 ∧ min_to_meter = 1000 / 60 ∧ (C * rev_per_min = car_speed_kmh * min_to_meter) → C = 5 :=
by
  sorry

end tire_circumference_l2125_212536


namespace thomas_needs_more_money_l2125_212517

-- Define the conditions in Lean
def weeklyAllowance : ℕ := 50
def hourlyWage : ℕ := 9
def hoursPerWeek : ℕ := 30
def weeklyExpenses : ℕ := 35
def weeksInYear : ℕ := 52
def carCost : ℕ := 15000

-- Define the total earnings for the first year
def firstYearEarnings : ℕ :=
  weeklyAllowance * weeksInYear

-- Define the weekly earnings from the second year job
def secondYearWeeklyEarnings : ℕ :=
  hourlyWage * hoursPerWeek

-- Define the total earnings for the second year
def secondYearEarnings : ℕ :=
  secondYearWeeklyEarnings * weeksInYear

-- Define the total earnings over two years
def totalEarnings : ℕ :=
  firstYearEarnings + secondYearEarnings

-- Define the total expenses over two years
def totalExpenses : ℕ :=
  weeklyExpenses * (2 * weeksInYear)

-- Define the net savings after two years
def netSavings : ℕ :=
  totalEarnings - totalExpenses

-- Define the amount more needed for the car
def amountMoreNeeded : ℕ :=
  carCost - netSavings

-- The theorem to prove
theorem thomas_needs_more_money : amountMoreNeeded = 2000 := by
  sorry

end thomas_needs_more_money_l2125_212517


namespace possible_initial_triangles_l2125_212565

-- Define the triangle types by their angles in degrees
inductive TriangleType
| T45T45T90
| T30T60T90
| T30T30T120
| T60T60T60

-- Define a Lean statement to express the problem
theorem possible_initial_triangles (T : TriangleType) :
  T = TriangleType.T45T45T90 ∨
  T = TriangleType.T30T60T90 ∨
  T = TriangleType.T30T30T120 ∨
  T = TriangleType.T60T60T60 :=
sorry

end possible_initial_triangles_l2125_212565


namespace remainder_98_mul_102_div_11_l2125_212556

theorem remainder_98_mul_102_div_11 : (98 * 102) % 11 = 7 := by
  sorry

end remainder_98_mul_102_div_11_l2125_212556


namespace circle_ring_ratio_l2125_212533

theorem circle_ring_ratio
  (r R c d : ℝ)
  (hr : 0 < r)
  (hR : 0 < R)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_areas : π * R^2 = (c / d) * (π * R^2 - π * r^2)) :
  R / r = (Real.sqrt c) / (Real.sqrt (d - c)) := 
by 
  sorry

end circle_ring_ratio_l2125_212533


namespace inequality_chain_l2125_212584

theorem inequality_chain (a b : ℝ) (h₁ : a < 0) (h₂ : -1 < b) (h₃ : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_chain_l2125_212584


namespace area_triangle_QCA_l2125_212540

noncomputable def area_of_triangle_QCA (p : ℝ) : ℝ :=
  let Q := (0, 12)
  let A := (3, 12)
  let C := (0, p)
  let QA := 3
  let QC := 12 - p
  (1/2) * QA * QC

theorem area_triangle_QCA (p : ℝ) : area_of_triangle_QCA p = (3/2) * (12 - p) :=
  sorry

end area_triangle_QCA_l2125_212540


namespace fraction_to_decimal_l2125_212531

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fraction_to_decimal_l2125_212531


namespace complex_division_l2125_212532

-- Define the complex numbers in Lean
def i : ℂ := Complex.I

-- Claim to be proved
theorem complex_division :
  (1 + i) / (3 - i) = (1 + 2 * i) / 5 :=
by
  sorry

end complex_division_l2125_212532


namespace subcommittee_count_l2125_212580

theorem subcommittee_count :
  let total_members := 12
  let teachers := 5
  let total_subcommittees := (Nat.choose total_members 4)
  let subcommittees_with_zero_teachers := (Nat.choose 7 4)
  let subcommittees_with_one_teacher := (Nat.choose teachers 1) * (Nat.choose 7 3)
  let subcommittees_with_fewer_than_two_teachers := subcommittees_with_zero_teachers + subcommittees_with_one_teacher
  let subcommittees_with_at_least_two_teachers := total_subcommittees - subcommittees_with_fewer_than_two_teachers
  subcommittees_with_at_least_two_teachers = 285 := by
  sorry

end subcommittee_count_l2125_212580


namespace min_value_l2125_212542

noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin (x / 2018) + (2019 ^ x - 1) / (2019 ^ x + 1)

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : f (2 * a) + f (b - 4) = 0) :
  2 * a + b = 4 → (1 / a + 2 / b) = 2 :=
by sorry

end min_value_l2125_212542


namespace value_of_5a_l2125_212545

variable (a : ℕ)

theorem value_of_5a (h : 5 * (a - 3) = 25) : 5 * a = 40 :=
sorry

end value_of_5a_l2125_212545


namespace race_distance_l2125_212550

/-
In a race, the ratio of the speeds of two contestants A and B is 3 : 4.
A has a start of 140 m.
A wins by 20 m.
Prove that the total distance of the race is 360 times the common speed factor.
-/
theorem race_distance (x D : ℕ)
  (ratio_A_B : ∀ (speed_A speed_B : ℕ), speed_A / speed_B = 3 / 4)
  (start_A : ∀ (start : ℕ), start = 140) 
  (win_A : ∀ (margin : ℕ), margin = 20) :
  D = 360 * x := 
sorry

end race_distance_l2125_212550


namespace find_x_l2125_212506

def operation (x y : ℕ) : ℕ := 2 * x * y

theorem find_x : 
  (operation 4 5 = 40) ∧ (operation x 40 = 480) → x = 6 :=
by
  sorry

end find_x_l2125_212506


namespace number_of_single_rooms_l2125_212583

theorem number_of_single_rooms (S : ℕ) : 
  (S + 13 * 2 = 40) ∧ (S * 10 + 13 * 2 * 10 = 400) → S = 14 :=
by 
  sorry

end number_of_single_rooms_l2125_212583


namespace abs_h_of_roots_sum_squares_eq_34_l2125_212548

theorem abs_h_of_roots_sum_squares_eq_34 
  (h : ℝ)
  (h_eq : ∀ r s : ℝ, (2 * r^2 + 4 * h * r + 6 = 0) ∧ (2 * s^2 + 4 * h * s + 6 = 0)) 
  (sum_of_squares_eq : ∀ r s : ℝ, (2 * r^2 + 4 * h * r + 6 = 0) ∧ (2 * s^2 + 4 * h * s + 6 = 0) → r^2 + s^2 = 34) :
  |h| = Real.sqrt 10 :=
by
  sorry

end abs_h_of_roots_sum_squares_eq_34_l2125_212548


namespace hockey_games_per_month_calculation_l2125_212596

-- Define the given conditions
def months_in_season : Nat := 14
def total_hockey_games : Nat := 182

-- Prove the number of hockey games played each month
theorem hockey_games_per_month_calculation :
  total_hockey_games / months_in_season = 13 := by
  sorry

end hockey_games_per_month_calculation_l2125_212596


namespace sum_of_transformed_numbers_l2125_212528

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  let a' := a + 4
  let b' := b + 4
  let a'' := 3 * a'
  let b'' := 3 * b'
  a'' + b'' = 3 * S + 24 := 
by
  let a' := a + 4
  let b' := b + 4
  let a'' := 3 * a'
  let b'' := 3 * b'
  sorry

end sum_of_transformed_numbers_l2125_212528


namespace flat_fee_is_65_l2125_212508

-- Define the problem constants
def George_nights : ℕ := 3
def Noah_nights : ℕ := 6
def George_cost : ℤ := 155
def Noah_cost : ℤ := 290

-- Prove that the flat fee for the first night is 65, given the costs and number of nights stayed.
theorem flat_fee_is_65 
  (f n : ℤ)
  (h1 : f + (George_nights - 1) * n = George_cost)
  (h2 : f + (Noah_nights - 1) * n = Noah_cost) :
  f = 65 := 
sorry

end flat_fee_is_65_l2125_212508


namespace weight_of_each_pack_l2125_212591

-- Definitions based on conditions
def total_sugar : ℕ := 3020
def leftover_sugar : ℕ := 20
def number_of_packs : ℕ := 12

-- Definition of sugar used for packs
def sugar_used_for_packs : ℕ := total_sugar - leftover_sugar

-- Proof statement to be verified
theorem weight_of_each_pack : sugar_used_for_packs / number_of_packs = 250 := by
  sorry

end weight_of_each_pack_l2125_212591


namespace greatest_odd_factors_below_200_l2125_212593

theorem greatest_odd_factors_below_200 : ∃ n : ℕ, (n < 200) ∧ (n = 196) ∧ (∃ k : ℕ, n = k^2) ∧ ∀ m : ℕ, (m < 200) ∧ (∃ j : ℕ, m = j^2) → m ≤ n := by
  sorry

end greatest_odd_factors_below_200_l2125_212593


namespace quadratic_has_two_distinct_real_roots_l2125_212507

-- Definitions of the conditions
def a : ℝ := 1
def b (k : ℝ) : ℝ := -3 * k
def c : ℝ := -2

-- Definition of the discriminant function
def discriminant (k : ℝ) : ℝ := (b k) ^ 2 - 4 * a * c

-- Logical statement to be proved
theorem quadratic_has_two_distinct_real_roots (k : ℝ) : discriminant k > 0 :=
by
  unfold discriminant
  unfold b a c
  simp
  sorry

end quadratic_has_two_distinct_real_roots_l2125_212507


namespace bhanu_income_problem_l2125_212598

-- Define the total income
def total_income (I : ℝ) : Prop :=
  let petrol_spent := 300
  let house_rent := 70
  (0.10 * (I - petrol_spent) = house_rent)

-- Define the percentage of income spent on petrol
def petrol_percentage (P : ℝ) (I : ℝ) : Prop :=
  0.01 * P * I = 300

-- The theorem we aim to prove
theorem bhanu_income_problem : 
  ∃ I P, total_income I ∧ petrol_percentage P I ∧ P = 30 :=
by
  sorry

end bhanu_income_problem_l2125_212598


namespace male_population_half_total_l2125_212597

theorem male_population_half_total (total_population : ℕ) (segments : ℕ) (male_segment : ℕ) :
  total_population = 800 ∧ segments = 4 ∧ male_segment = 1 ∧ male_segment = segments / 2 →
  total_population / 2 = 400 :=
by
  intro h
  sorry

end male_population_half_total_l2125_212597


namespace base_of_parallelogram_l2125_212505

theorem base_of_parallelogram (area height base : ℝ) 
  (h_area : area = 320)
  (h_height : height = 16) :
  base = area / height :=
by 
  rw [h_area, h_height]
  norm_num
  sorry

end base_of_parallelogram_l2125_212505


namespace find_ratio_of_arithmetic_sequences_l2125_212557

variable {a_n b_n : ℕ → ℕ}
variable {A_n B_n : ℕ → ℝ}

def arithmetic_sums (a_n b_n : ℕ → ℕ) (A_n B_n : ℕ → ℝ) : Prop :=
  ∀ n, A_n n = (n * (2 * a_n 1 + (n - 1) * (a_n 8 - a_n 7))) / 2 ∧
         B_n n = (n * (2 * b_n 1 + (n - 1) * (b_n 8 - b_n 7))) / 2

theorem find_ratio_of_arithmetic_sequences 
    (h : ∀ n, A_n n / B_n n = (5 * n - 3) / (n + 9)) :
    ∃ r : ℝ, r = 3 := by
  sorry

end find_ratio_of_arithmetic_sequences_l2125_212557


namespace solve_for_m_l2125_212523

-- Define the conditions for the lines being parallel
def condition_one (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + m * y + 3 = 0

def condition_two (m : ℝ) : Prop :=
  ∃ x y : ℝ, (m - 1) * x + 2 * m * y + 2 * m = 0

def are_parallel (A B C D : ℝ) : Prop :=
  A * D = B * C

theorem solve_for_m :
  ∀ (m : ℝ),
    (condition_one m) → 
    (condition_two m) → 
    (are_parallel 1 m 3 (2 * m)) →
    (m = 0) :=
by
  intro m h1 h2 h_parallel
  sorry

end solve_for_m_l2125_212523


namespace integer_solutions_l2125_212552

theorem integer_solutions (x y : ℤ) : 
  x^2 * y = 10000 * x + y ↔ 
  (x, y) = (-9, -1125) ∨ 
  (x, y) = (-3, -3750) ∨ 
  (x, y) = (0, 0) ∨ 
  (x, y) = (3, 3750) ∨ 
  (x, y) = (9, 1125) := 
by
  sorry

end integer_solutions_l2125_212552


namespace people_in_each_van_l2125_212554

theorem people_in_each_van
  (cars : ℕ) (taxis : ℕ) (vans : ℕ)
  (people_per_car : ℕ) (people_per_taxi : ℕ) (total_people : ℕ) 
  (people_per_van : ℕ) :
  cars = 3 → taxis = 6 → vans = 2 →
  people_per_car = 4 → people_per_taxi = 6 → total_people = 58 →
  3 * people_per_car + 6 * people_per_taxi + 2 * people_per_van = total_people →
  people_per_van = 5 :=
by sorry

end people_in_each_van_l2125_212554


namespace relationship_among_a_b_c_l2125_212588

noncomputable def a : ℝ := Real.log (7 / 2) / Real.log 3
noncomputable def b : ℝ := (1 / 4)^(1 / 3)
noncomputable def c : ℝ := -Real.log 5 / Real.log 3

theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_among_a_b_c_l2125_212588


namespace find_b_vector_l2125_212501

-- Define input vectors a, b, and their sum.
def vec_a : ℝ × ℝ × ℝ := (1, -2, 1)
def vec_b : ℝ × ℝ × ℝ := (-2, 4, -2)
def vec_sum : ℝ × ℝ × ℝ := (-1, 2, -1)

-- The theorem statement to prove that b is calculated correctly.
theorem find_b_vector :
  vec_a + vec_b = vec_sum →
  vec_b = (-2, 4, -2) :=
by
  sorry

end find_b_vector_l2125_212501


namespace simplify_expression1_simplify_expression2_l2125_212562

/-- Proof Problem 1: Simplify the expression (a+2b)^2 - 4b(a+b) -/
theorem simplify_expression1 (a b : ℝ) : 
  (a + 2 * b)^2 - 4 * b * (a + b) = a^2 :=
sorry

/-- Proof Problem 2: Simplify the expression ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) ÷ (x - 1) / (x^2 - 4) -/
theorem simplify_expression2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) : 
  ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 :=
sorry

end simplify_expression1_simplify_expression2_l2125_212562


namespace sum_of_distinct_roots_l2125_212541

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l2125_212541


namespace ab_fraction_l2125_212573

theorem ab_fraction (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 9) (h2 : a * b = 20) : 
  (1 / a + 1 / b) = 9 / 20 := 
by 
  sorry

end ab_fraction_l2125_212573


namespace find_m_l2125_212570

theorem find_m (m : ℤ) : m < 2 * Real.sqrt 3 ∧ 2 * Real.sqrt 3 < m + 1 → m = 3 :=
sorry

end find_m_l2125_212570


namespace range_f_x_negative_l2125_212504

-- We define the conditions: f is an even function, increasing on (-∞, 0), and f(2) = 0.
variables {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_neg_infinity_to_zero (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → x < 0 ∧ y < 0 → f x ≤ f y

def f_at_2_is_zero (f : ℝ → ℝ) : Prop :=
  f 2 = 0

-- The theorem to be proven.
theorem range_f_x_negative (hf_even : even_function f)
  (hf_incr : increasing_on_neg_infinity_to_zero f)
  (hf_at2 : f_at_2_is_zero f) :
  ∀ x, f x < 0 ↔ x < -2 ∨ x > 2 :=
by
  sorry

end range_f_x_negative_l2125_212504


namespace canada_population_l2125_212576

theorem canada_population 
    (M : ℕ) (B : ℕ) (H : ℕ)
    (hM : M = 1000000)
    (hB : B = 2 * M)
    (hH : H = 19 * B) : 
    H = 38000000 := by
  sorry

end canada_population_l2125_212576


namespace smallest_sum_xy_min_45_l2125_212535

theorem smallest_sum_xy_min_45 (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y) (h4 : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 10) :
  x + y = 45 :=
by
  sorry

end smallest_sum_xy_min_45_l2125_212535


namespace binomial_coeff_and_coeff_of_x8_l2125_212563

theorem binomial_coeff_and_coeff_of_x8 (x : ℂ) :
  let expr := (x^2 + 4*x + 4)^5
  let expansion := (x + 2)^10
  ∃ (binom_coeff_x8 coeff_x8 : ℤ),
    binom_coeff_x8 = 45 ∧ coeff_x8 = 180 :=
by
  sorry

end binomial_coeff_and_coeff_of_x8_l2125_212563


namespace proof_of_arithmetic_sequence_l2125_212527

theorem proof_of_arithmetic_sequence 
  (x y z : ℕ) 
  (h1 : x + y + z = 15) 
  (h2 : x < y) 
  (h3 : y < z)
  (h4 : (x + 1) * (z + 9) = (y + 3) ^ 2) : 
  (x, y, z) = (3, 5, 7) :=
sorry

end proof_of_arithmetic_sequence_l2125_212527


namespace chemistry_class_students_l2125_212589

theorem chemistry_class_students (total_students both_classes biology_class only_chemistry_class : ℕ)
    (h1: total_students = 100)
    (h2 : both_classes = 10)
    (h3 : total_students = biology_class + only_chemistry_class + both_classes)
    (h4 : only_chemistry_class = 4 * (biology_class + both_classes)) : 
    only_chemistry_class = 80 :=
by
  sorry

end chemistry_class_students_l2125_212589


namespace sum_of_other_endpoint_l2125_212599

theorem sum_of_other_endpoint (x y : ℝ) (h₁ : (9 + x) / 2 = 5) (h₂ : (-6 + y) / 2 = -8) :
  x + y = -9 :=
sorry

end sum_of_other_endpoint_l2125_212599


namespace roots_of_quadratic_l2125_212568

theorem roots_of_quadratic (b c : ℝ) (h1 : 1 + -2 = -b) (h2 : 1 * -2 = c) : b = 1 ∧ c = -2 :=
by
  sorry

end roots_of_quadratic_l2125_212568


namespace milk_packet_volume_l2125_212551

theorem milk_packet_volume :
  ∃ (m : ℕ), (150 * m = 1250 * 30) ∧ m = 250 :=
by
  sorry

end milk_packet_volume_l2125_212551


namespace percentage_of_x_is_2x_minus_y_l2125_212582

variable (x y : ℝ)
variable (h1 : x / y = 4)
variable (h2 : y ≠ 0)

theorem percentage_of_x_is_2x_minus_y :
  (2 * x - y) / x * 100 = 175 := by
  sorry

end percentage_of_x_is_2x_minus_y_l2125_212582


namespace rectangle_dimensions_l2125_212581

theorem rectangle_dimensions (a b : ℝ) 
  (h_area : a * b = 12) 
  (h_perimeter : 2 * (a + b) = 26) : 
  (a = 1 ∧ b = 12) ∨ (a = 12 ∧ b = 1) :=
sorry

end rectangle_dimensions_l2125_212581


namespace red_card_value_l2125_212538

theorem red_card_value (credits : ℕ) (total_cards : ℕ) (blue_card_value : ℕ) (red_cards : ℕ) (blue_cards : ℕ) 
    (condition1 : blue_card_value = 5)
    (condition2 : total_cards = 20)
    (condition3 : credits = 84)
    (condition4 : red_cards = 8)
    (condition5 : blue_cards = total_cards - red_cards) :
  (credits - blue_cards * blue_card_value) / red_cards = 3 :=
by
  sorry

end red_card_value_l2125_212538


namespace find_abc_squared_sum_l2125_212569

theorem find_abc_squared_sum (a b c : ℕ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a^3 + 32 * b + 2 * c = 2018) (h₃ : b^3 + 32 * a + 2 * c = 1115) :
  a^2 + b^2 + c^2 = 226 :=
sorry

end find_abc_squared_sum_l2125_212569


namespace combinedAgeIn5Years_l2125_212555

variable (Amy Mark Emily : ℕ)

-- Conditions
def amyAge : ℕ := 15
def markAge : ℕ := amyAge + 7
def emilyAge : ℕ := 2 * amyAge

-- Proposition to be proved
theorem combinedAgeIn5Years :
  Amy = amyAge →
  Mark = markAge →
  Emily = emilyAge →
  (Amy + 5) + (Mark + 5) + (Emily + 5) = 82 :=
by
  intros hAmy hMark hEmily
  sorry

end combinedAgeIn5Years_l2125_212555


namespace sugar_flour_ratio_10_l2125_212559

noncomputable def sugar_to_flour_ratio (sugar flour : ℕ) : ℕ :=
  sugar / flour

theorem sugar_flour_ratio_10 (sugar flour : ℕ) (hs : sugar = 50) (hf : flour = 5) : sugar_to_flour_ratio sugar flour = 10 :=
by
  rw [hs, hf]
  unfold sugar_to_flour_ratio
  norm_num
  -- sorry

end sugar_flour_ratio_10_l2125_212559


namespace arithmetic_seq_a1_l2125_212547

theorem arithmetic_seq_a1 (a_1 d : ℝ) (h1 : a_1 + 4 * d = 9) (h2 : 2 * (a_1 + 2 * d) = (a_1 + d) + 6) : a_1 = -3 := by
  sorry

end arithmetic_seq_a1_l2125_212547


namespace roots_of_quadratic_eq_l2125_212566

theorem roots_of_quadratic_eq (h : ∀ x : ℝ, x^2 - 3 * x = 0 → x = 0 ∨ x = 3) :
  ∀ x : ℝ, x^2 - 3 * x = 0 → x = 0 ∨ x = 3 :=
by sorry

end roots_of_quadratic_eq_l2125_212566


namespace smallest_integer_with_20_divisors_l2125_212510

theorem smallest_integer_with_20_divisors : ∃ n : ℕ, 
  (0 < n) ∧ 
  (∀ m : ℕ, (0 < m ∧ ∃ k : ℕ, m = n * k) ↔ (∃ d : ℕ, d.succ * (20 / d.succ) = 20)) ∧ 
  n = 240 := 
sorry

end smallest_integer_with_20_divisors_l2125_212510


namespace range_of_m_l2125_212512

noncomputable def inequality_has_solutions (x m : ℝ) :=
  |x + 2| - |x + 3| > m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, inequality_has_solutions x m) → m < 1 :=
by
  sorry

end range_of_m_l2125_212512


namespace time_ratio_l2125_212521

theorem time_ratio (A : ℝ) (B : ℝ) (h1 : B = 18) (h2 : 1 / A + 1 / B = 1 / 3) : A / B = 1 / 5 :=
by
  sorry

end time_ratio_l2125_212521


namespace v2_correct_at_2_l2125_212522

def poly (x : ℕ) : ℕ := x^5 + x^4 + 2 * x^3 + 3 * x^2 + 4 * x + 1

def horner_v2 (x : ℕ) : ℕ :=
  let v0 := 1
  let v1 := v0 * x + 4
  let v2 := v1 * x + 3
  v2

theorem v2_correct_at_2 : horner_v2 2 = 15 := by
  sorry

end v2_correct_at_2_l2125_212522


namespace number_of_outliers_l2125_212514

def data_set : List ℕ := [10, 24, 36, 36, 42, 45, 45, 46, 58, 64]
def Q1 : ℕ := 36
def Q3 : ℕ := 46
def IQR : ℕ := Q3 - Q1
def low_threshold : ℕ := Q1 - 15
def high_threshold : ℕ := Q3 + 15
def outliers : List ℕ := data_set.filter (λ x => x < low_threshold ∨ x > high_threshold)

theorem number_of_outliers : outliers.length = 3 :=
  by
    -- Proof would go here
    sorry

end number_of_outliers_l2125_212514


namespace exists_positive_real_u_l2125_212524

theorem exists_positive_real_u (n : ℕ) (h_pos : n > 0) : 
  ∃ u : ℝ, u > 0 ∧ ∀ n : ℕ, n > 0 → (⌊u^n⌋ - n) % 2 = 0 :=
sorry

end exists_positive_real_u_l2125_212524


namespace geometric_progression_ratio_l2125_212513

theorem geometric_progression_ratio (q : ℝ) (h : |q| < 1 ∧ ∀a : ℝ, a = 4 * (a * q / (1 - q) - a * q)) :
  q = 1 / 5 :=
by
  sorry

end geometric_progression_ratio_l2125_212513


namespace find_g5_l2125_212503

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l2125_212503


namespace circle_radius_l2125_212579

theorem circle_radius
  (area_sector : ℝ)
  (arc_length : ℝ)
  (h_area : area_sector = 8.75)
  (h_arc : arc_length = 3.5) :
  ∃ r : ℝ, r = 5 :=
by
  let r := 5
  use r
  sorry

end circle_radius_l2125_212579


namespace no_solution_in_natural_numbers_l2125_212594

theorem no_solution_in_natural_numbers :
  ¬ ∃ (x y : ℕ), 2^x + 21^x = y^3 :=
sorry

end no_solution_in_natural_numbers_l2125_212594


namespace min_value_fraction_l2125_212539

theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : 
  ∀ x, (x = (1 / m + 8 / n)) → x ≥ 18 :=
by
  sorry

end min_value_fraction_l2125_212539


namespace parabola_latus_rectum_l2125_212572

theorem parabola_latus_rectum (x p y : ℝ) (hp : p > 0) (h_eq : x^2 = 2 * p * y) (hl : y = -3) :
  p = 6 :=
by
  sorry

end parabola_latus_rectum_l2125_212572


namespace triangle_is_isosceles_l2125_212571

theorem triangle_is_isosceles {a b c : ℝ} {A B C : ℝ} (h1 : b * Real.cos A = a * Real.cos B) : 
  a = b ∨ a = c ∨ b = c :=
sorry

end triangle_is_isosceles_l2125_212571


namespace root_of_quadratic_gives_value_l2125_212561

theorem root_of_quadratic_gives_value (a : ℝ) (h : a^2 + 3 * a - 5 = 0) : a^2 + 3 * a + 2021 = 2026 :=
by {
  -- We will skip the proof here.
  sorry
}

end root_of_quadratic_gives_value_l2125_212561


namespace quadratic_polynomial_divisible_by_3_l2125_212544

theorem quadratic_polynomial_divisible_by_3
  (a b c : ℤ)
  (h : ∀ x : ℤ, 3 ∣ (a * x^2 + b * x + c)) :
  3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c :=
sorry

end quadratic_polynomial_divisible_by_3_l2125_212544
