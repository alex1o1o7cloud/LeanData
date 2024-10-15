import Mathlib

namespace NUMINAMATH_GPT_arrow_sequence_correct_l2059_205951

variable (A B C D E F G : ℕ)
variable (square : ℕ → ℕ)

-- Definitions based on given conditions
def conditions : Prop :=
  square 1 = 1 ∧ square 9 = 9 ∧
  square A = 6 ∧ square B = 2 ∧ square C = 4 ∧
  square D = 5 ∧ square E = 3 ∧ square F = 8 ∧ square G = 7 ∧
  (∀ x, (x = 1 → square 2 = B) ∧ (x = 2 → square 3 = E) ∧
       (x = 3 → square 4 = C) ∧ (x = 4 → square 5 = D) ∧
       (x = 5 → square 6 = A) ∧ (x = 6 → square 7 = G) ∧
       (x = 7 → square 8 = F) ∧ (x = 8 → square 9 = 9))

theorem arrow_sequence_correct :
  conditions A B C D E F G square → 
  ∀ x, square (x + 1) = 1 + x :=
by sorry

end NUMINAMATH_GPT_arrow_sequence_correct_l2059_205951


namespace NUMINAMATH_GPT_problem_1_problem_2_l2059_205981

-- Definitions of conditions
variables {a b : ℝ}
axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_sum : a + b = 1

-- The statements to prove
theorem problem_1 : 
  (1 / (a^2)) + (1 / (b^2)) ≥ 8 := 
sorry

theorem problem_2 : 
  (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2059_205981


namespace NUMINAMATH_GPT_four_disjoint_subsets_with_equal_sums_l2059_205945

theorem four_disjoint_subsets_with_equal_sums :
  ∀ (S : Finset ℕ), 
  (∀ x ∈ S, 100 ≤ x ∧ x ≤ 999) ∧ S.card = 117 → 
  ∃ A B C D : Finset ℕ, 
    (A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧ D ⊆ S) ∧ 
    (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ A ∩ D = ∅ ∧ B ∩ C = ∅ ∧ B ∩ D = ∅ ∧ C ∩ D = ∅) ∧ 
    (A.sum id = B.sum id ∧ B.sum id = C.sum id ∧ C.sum id = D.sum id) := by
  sorry

end NUMINAMATH_GPT_four_disjoint_subsets_with_equal_sums_l2059_205945


namespace NUMINAMATH_GPT_nat_number_of_the_form_l2059_205972

theorem nat_number_of_the_form (a b : ℕ) (h : ∃ (a b : ℕ), a * a * 3 + b * b * 32 = n) :
  ∃ (a' b' : ℕ), a' * a' * 3 + b' * b' * 32 = 97 * n  :=
  sorry

end NUMINAMATH_GPT_nat_number_of_the_form_l2059_205972


namespace NUMINAMATH_GPT_sector_area_150_degrees_l2059_205914

def sector_area (radius : ℝ) (central_angle : ℝ) : ℝ :=
  0.5 * radius^2 * central_angle

theorem sector_area_150_degrees (r : ℝ) (angle_rad : ℝ) (h1 : r = Real.sqrt 3) (h2 : angle_rad = (5 * Real.pi) / 6) : 
  sector_area r angle_rad = (5 * Real.pi) / 4 :=
by
  simp [sector_area, h1, h2]
  sorry

end NUMINAMATH_GPT_sector_area_150_degrees_l2059_205914


namespace NUMINAMATH_GPT_cars_already_parked_l2059_205933

-- Define the levels and their parking spaces based on given conditions
def first_level_spaces : Nat := 90
def second_level_spaces : Nat := first_level_spaces + 8
def third_level_spaces : Nat := second_level_spaces + 12
def fourth_level_spaces : Nat := third_level_spaces - 9

-- Compute total spaces in the garage
def total_spaces : Nat := first_level_spaces + second_level_spaces + third_level_spaces + fourth_level_spaces

-- Define the available spaces for more cars
def available_spaces : Nat := 299

-- Prove the number of cars already parked
theorem cars_already_parked : total_spaces - available_spaces = 100 :=
by
  exact Nat.sub_eq_of_eq_add sorry -- Fill in with the actual proof step

end NUMINAMATH_GPT_cars_already_parked_l2059_205933


namespace NUMINAMATH_GPT_real_distance_between_cities_l2059_205906

-- Condition: the map distance between Goteborg and Jonkoping
def map_distance_cm : ℝ := 88

-- Condition: the map scale
def map_scale_km_per_cm : ℝ := 15

-- The real distance to be proven
theorem real_distance_between_cities :
  (map_distance_cm * map_scale_km_per_cm) = 1320 := by
  sorry

end NUMINAMATH_GPT_real_distance_between_cities_l2059_205906


namespace NUMINAMATH_GPT_xiao_wang_ways_to_make_8_cents_l2059_205984

theorem xiao_wang_ways_to_make_8_cents :
  let one_cent_coins := 8
  let two_cent_coins := 4
  let five_cent_coin := 1
  ∃ ways, ways = 7 ∧ (
       (ways = 8 ∧ one_cent_coins >= 8) ∨
       (ways = 4 ∧ two_cent_coins >= 4) ∨
       (ways = 2 ∧ one_cent_coins >= 2 ∧ two_cent_coins >= 3) ∨
       (ways = 4 ∧ one_cent_coins >= 4 ∧ two_cent_coins >= 2) ∨
       (ways = 6 ∧ one_cent_coins >= 6 ∧ two_cent_coins >= 1) ∨
       (ways = 3 ∧ one_cent_coins >= 3 ∧ five_cent_coin >= 1) ∨
       (ways = 1 ∧ one_cent_coins >= 1 ∧ two_cent_coins >= 1 ∧ five_cent_coin >= 1)
   ) :=
  sorry

end NUMINAMATH_GPT_xiao_wang_ways_to_make_8_cents_l2059_205984


namespace NUMINAMATH_GPT_max_sum_a_b_c_d_l2059_205964

theorem max_sum_a_b_c_d (a c d : ℤ) (b : ℕ) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  a + b + c + d = -5 := 
sorry

end NUMINAMATH_GPT_max_sum_a_b_c_d_l2059_205964


namespace NUMINAMATH_GPT_vertex_of_parabola_l2059_205917

theorem vertex_of_parabola :
  (∃ x y : ℝ, y = -3*x^2 + 6*x + 1 ∧ (x, y) = (1, 4)) :=
sorry

end NUMINAMATH_GPT_vertex_of_parabola_l2059_205917


namespace NUMINAMATH_GPT_calculate_final_speed_l2059_205930

noncomputable def final_speed : ℝ :=
  let v1 : ℝ := (150 * 1.60934 * 1000) / 3600
  let v2 : ℝ := (170 * 1000) / 3600
  let v_decreased : ℝ := v1 - v2
  let a : ℝ := (500000 * 0.01) / 60
  v_decreased + a * (30 * 60)

theorem calculate_final_speed : final_speed = 150013.45 :=
by
  sorry

end NUMINAMATH_GPT_calculate_final_speed_l2059_205930


namespace NUMINAMATH_GPT_solution_of_loginequality_l2059_205990

-- Define the conditions as inequalities
def condition1 (x : ℝ) : Prop := 2 * x - 1 > 0
def condition2 (x : ℝ) : Prop := -x + 5 > 0
def condition3 (x : ℝ) : Prop := 2 * x - 1 > -x + 5

-- Define the final solution set
def solution_set (x : ℝ) : Prop := (2 < x) ∧ (x < 5)

-- The theorem stating that under the given conditions, the solution set holds
theorem solution_of_loginequality (x : ℝ) : condition1 x ∧ condition2 x ∧ condition3 x → solution_set x :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solution_of_loginequality_l2059_205990


namespace NUMINAMATH_GPT_proof_problem_l2059_205926

noncomputable def g (x : ℝ) : ℝ := 2^(2*x - 1) + x - 1

theorem proof_problem
  (x1 x2 : ℝ)
  (h1 : g x1 = 0)  -- x1 is the root of the equation
  (h2 : 2 * x2 - 1 = 0)  -- x2 is the zero point of f(x) = 2x - 1
  : |x1 - x2| ≤ 1/4 :=
sorry

end NUMINAMATH_GPT_proof_problem_l2059_205926


namespace NUMINAMATH_GPT_tommy_nickels_l2059_205935

-- Definitions of given conditions
def pennies (quarters : Nat) : Nat := 10 * quarters  -- Tommy has 10 times as many pennies as quarters
def dimes (pennies : Nat) : Nat := pennies + 10      -- Tommy has 10 more dimes than pennies
def nickels (dimes : Nat) : Nat := 2 * dimes         -- Tommy has twice as many nickels as dimes

theorem tommy_nickels (quarters : Nat) (P : Nat) (D : Nat) (N : Nat) 
  (h1 : quarters = 4) 
  (h2 : P = pennies quarters) 
  (h3 : D = dimes P) 
  (h4 : N = nickels D) : 
  N = 100 := 
by
  -- sorry allows us to skip the proof
  sorry

end NUMINAMATH_GPT_tommy_nickels_l2059_205935


namespace NUMINAMATH_GPT_problem_statement_l2059_205901

noncomputable def AB2_AC2_BC2_eq_4 (l m n k : ℝ) : Prop :=
  let D := (l+k, 0, 0)
  let E := (0, m+k, 0)
  let F := (0, 0, n+k)
  let AB_sq := 4 * (n+k)^2
  let AC_sq := 4 * (m+k)^2
  let BC_sq := 4 * (l+k)^2
  AB_sq + AC_sq + BC_sq = 4 * ((l+k)^2 + (m+k)^2 + (n+k)^2)

theorem problem_statement (l m n k : ℝ) : 
  AB2_AC2_BC2_eq_4 l m n k :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2059_205901


namespace NUMINAMATH_GPT_point_on_line_l2059_205962

theorem point_on_line :
  ∃ a b : ℝ, (a ≠ 0) ∧
  (∀ x y : ℝ, (x = 4 ∧ y = 5) ∨ (x = 8 ∧ y = 17) ∨ (x = 12 ∧ y = 29) → y = a * x + b) →
  (∃ t : ℝ, (15, t) ∈ {(x, y) | y = a * x + b} ∧ t = 38) :=
by
  sorry

end NUMINAMATH_GPT_point_on_line_l2059_205962


namespace NUMINAMATH_GPT_part1_part2_l2059_205928

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 6)

theorem part1 (k : ℝ) :
  (∀ x : ℝ, (f x > k) ↔ (x < -3 ∨ x > -2)) ↔ k = -2/5 :=
by
  sorry

theorem part2 (t : ℝ) :
  (∀ x : ℝ, (x > 0) → (f x ≤ t)) ↔ t ∈ (Set.Ici (Real.sqrt 6 / 6)) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2059_205928


namespace NUMINAMATH_GPT_area_of_inscribed_square_l2059_205977

theorem area_of_inscribed_square
    (r : ℝ)
    (h : ∀ A : ℝ × ℝ, (A.1 = r - 1 ∨ A.1 = -(r - 1)) ∧ (A.2 = r - 2 ∨ A.2 = -(r - 2)) → A.1^2 + A.2^2 = r^2) :
    4 * r^2 = 100 := by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_area_of_inscribed_square_l2059_205977


namespace NUMINAMATH_GPT_gcd_282_470_l2059_205949

theorem gcd_282_470 : Int.gcd 282 470 = 94 := by
  sorry

end NUMINAMATH_GPT_gcd_282_470_l2059_205949


namespace NUMINAMATH_GPT_numberOfBags_l2059_205988

-- Define the given conditions
def totalCookies : Nat := 33
def cookiesPerBag : Nat := 11

-- Define the statement to prove
theorem numberOfBags : totalCookies / cookiesPerBag = 3 := by
  sorry

end NUMINAMATH_GPT_numberOfBags_l2059_205988


namespace NUMINAMATH_GPT_age_of_youngest_child_l2059_205976

theorem age_of_youngest_child
  (total_bill : ℝ)
  (mother_charge : ℝ)
  (child_charge_per_year : ℝ)
  (children_total_years : ℝ)
  (twins_age : ℕ)
  (youngest_child_age : ℕ)
  (h_total_bill : total_bill = 13.00)
  (h_mother_charge : mother_charge = 6.50)
  (h_child_charge_per_year : child_charge_per_year = 0.65)
  (h_children_bill : total_bill - mother_charge = children_total_years * child_charge_per_year)
  (h_children_age : children_total_years = 10)
  (h_youngest_child : youngest_child_age = 10 - 2 * twins_age) :
  youngest_child_age = 2 ∨ youngest_child_age = 4 :=
by
  sorry

end NUMINAMATH_GPT_age_of_youngest_child_l2059_205976


namespace NUMINAMATH_GPT_sum_of_values_l2059_205987

theorem sum_of_values (N : ℝ) (R : ℝ) (h : N ≠ 0) (h_eq : N + 5 / N = R) : N = R := 
sorry

end NUMINAMATH_GPT_sum_of_values_l2059_205987


namespace NUMINAMATH_GPT_solution_opposite_numbers_l2059_205923

theorem solution_opposite_numbers (x y : ℤ) (h1 : 2 * x + 3 * y - 4 = 0) (h2 : x = -y) : x = -4 ∧ y = 4 :=
by
  sorry

end NUMINAMATH_GPT_solution_opposite_numbers_l2059_205923


namespace NUMINAMATH_GPT_find_b_exists_l2059_205939

theorem find_b_exists (N : ℕ) (hN : N ≠ 1) : ∃ (a c d : ℕ), a > 1 ∧ c > 1 ∧ d > 1 ∧
  (N : ℝ) ^ (1/a + 1/(a*4) + 1/(a*4*c) + 1/(a*4*c*d)) = (N : ℝ) ^ (37/48) :=
by
  sorry

end NUMINAMATH_GPT_find_b_exists_l2059_205939


namespace NUMINAMATH_GPT_find_numbers_l2059_205980

theorem find_numbers (a b : ℕ) 
  (h1 : a / b * 6 = 10)
  (h2 : a - b + 4 = 10) :
  a = 15 ∧ b = 9 := by
  sorry

end NUMINAMATH_GPT_find_numbers_l2059_205980


namespace NUMINAMATH_GPT_inequality_proof_l2059_205903

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (z + x) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2059_205903


namespace NUMINAMATH_GPT_ratio_to_percent_l2059_205982

theorem ratio_to_percent :
  (9 / 5 * 100) = 180 :=
by
  sorry

end NUMINAMATH_GPT_ratio_to_percent_l2059_205982


namespace NUMINAMATH_GPT_second_coloring_book_pictures_l2059_205991

-- Let P1 be the number of pictures in the first coloring book.
def P1 := 23

-- Let P2 be the number of pictures in the second coloring book.
variable (P2 : Nat)

-- Let colored_pics be the number of pictures Rachel colored.
def colored_pics := 44

-- Let remaining_pics be the number of pictures Rachel still has to color.
def remaining_pics := 11

-- Total number of pictures in both coloring books.
def total_pics := colored_pics + remaining_pics

theorem second_coloring_book_pictures :
  P2 = total_pics - P1 :=
by
  -- We need to prove that P2 = 32.
  sorry

end NUMINAMATH_GPT_second_coloring_book_pictures_l2059_205991


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l2059_205912

theorem quadratic_no_real_roots (m : ℝ) : (4 + 4 * m < 0) → (m < -1) :=
by
  intro h
  linarith

end NUMINAMATH_GPT_quadratic_no_real_roots_l2059_205912


namespace NUMINAMATH_GPT_product_mk_through_point_l2059_205944

theorem product_mk_through_point (k m : ℝ) (h : (2 : ℝ) ^ m * k = (1/4 : ℝ)) : m * k = -2 := 
sorry

end NUMINAMATH_GPT_product_mk_through_point_l2059_205944


namespace NUMINAMATH_GPT_largest_t_value_l2059_205927

theorem largest_t_value : 
  ∃ t : ℝ, 
    (∃ s : ℝ, s > 0 ∧ t = 3 ∧
    ∀ u : ℝ, 
      (u = 3 →
        (15 * u^2 - 40 * u + 18) / (4 * u - 3) + 3 * u = 4 * u + 2 ∧
        u ≤ 3) ∧
      (u ≠ 3 → 
        (15 * u^2 - 40 * u + 18) / (4 * u - 3) + 3 * u = 4 * u + 2 → 
        u ≤ 3)) :=
sorry

end NUMINAMATH_GPT_largest_t_value_l2059_205927


namespace NUMINAMATH_GPT_alcohol_percentage_after_adding_water_l2059_205940

variables (initial_volume : ℕ) (initial_percentage : ℕ) (added_volume : ℕ)
def initial_alcohol_volume := initial_volume * initial_percentage / 100
def final_volume := initial_volume + added_volume
def final_percentage := initial_alcohol_volume * 100 / final_volume

theorem alcohol_percentage_after_adding_water :
  initial_volume = 15 →
  initial_percentage = 20 →
  added_volume = 5 →
  final_percentage = 15 := by
sorry

end NUMINAMATH_GPT_alcohol_percentage_after_adding_water_l2059_205940


namespace NUMINAMATH_GPT_soccer_lineup_count_l2059_205989

theorem soccer_lineup_count :
  let total_players : ℕ := 16
  let total_starters : ℕ := 7
  let m_j_players : ℕ := 2 -- Michael and John
  let other_players := total_players - m_j_players
  let total_ways : ℕ :=
    2 * Nat.choose other_players (total_starters - 1) + Nat.choose other_players (total_starters - 2)
  total_ways = 8008
:= sorry

end NUMINAMATH_GPT_soccer_lineup_count_l2059_205989


namespace NUMINAMATH_GPT_kyle_money_l2059_205998

theorem kyle_money (dave_money : ℕ) (kyle_initial : ℕ) (kyle_remaining : ℕ)
  (h1 : dave_money = 46)
  (h2 : kyle_initial = 3 * dave_money - 12)
  (h3 : kyle_remaining = kyle_initial - kyle_initial / 3) :
  kyle_remaining = 84 :=
by
  -- Define Dave's money and provide the assumption
  let dave_money := 46
  have h1 : dave_money = 46 := rfl

  -- Define Kyle's initial money based on Dave's money
  let kyle_initial := 3 * dave_money - 12
  have h2 : kyle_initial = 3 * dave_money - 12 := rfl

  -- Define Kyle's remaining money after spending one third on snowboarding
  let kyle_remaining := kyle_initial - kyle_initial / 3
  have h3 : kyle_remaining = kyle_initial - kyle_initial / 3 := rfl

  -- Now we prove that Kyle's remaining money is 84
  sorry -- Proof steps omitted

end NUMINAMATH_GPT_kyle_money_l2059_205998


namespace NUMINAMATH_GPT_possible_values_l2059_205948

theorem possible_values (a : ℝ) (h : a > 1) : ∃ (v : ℝ), (v = 5 ∨ v = 6 ∨ v = 7) ∧ (a + 4 / (a - 1) = v) :=
sorry

end NUMINAMATH_GPT_possible_values_l2059_205948


namespace NUMINAMATH_GPT_gcd_f_x_l2059_205967

def f (x : ℤ) : ℤ := (5 * x + 3) * (11 * x + 2) * (14 * x + 7) * (3 * x + 8)

theorem gcd_f_x (x : ℤ) (hx : x % 3456 = 0) : Int.gcd (f x) x = 48 := by
  sorry

end NUMINAMATH_GPT_gcd_f_x_l2059_205967


namespace NUMINAMATH_GPT_second_vote_difference_l2059_205904

-- Define the total number of members
def total_members : ℕ := 300

-- Define the votes for and against in the initial vote
structure votes_initial :=
  (a : ℕ) (b : ℕ) (h : a + b = total_members) (rejected : b > a)

-- Define the votes for and against in the second vote
structure votes_second :=
  (a' : ℕ) (b' : ℕ) (h : a' + b' = total_members)

-- Define the margin and condition of passage by three times the margin
def margin (vi : votes_initial) : ℕ := vi.b - vi.a

def passage_by_margin (vi : votes_initial) (vs : votes_second) : Prop :=
  vs.a' - vs.b' = 3 * margin vi

-- Define the condition that a' is 7/6 times b
def proportion (vs : votes_second) (vi : votes_initial) : Prop :=
  vs.a' = (7 * vi.b) / 6

-- The final proof statement
theorem second_vote_difference (vi : votes_initial) (vs : votes_second)
  (h_margin : passage_by_margin vi vs)
  (h_proportion : proportion vs vi) :
  vs.a' - vi.a = 55 :=
by
  sorry  -- This is where the proof would go

end NUMINAMATH_GPT_second_vote_difference_l2059_205904


namespace NUMINAMATH_GPT_jaclyn_constant_term_l2059_205994

variable {R : Type*} [CommRing R] (P Q : Polynomial R)

theorem jaclyn_constant_term (hP : P.leadingCoeff = 1) (hQ : Q.leadingCoeff = 1)
  (deg_P : P.degree = 4) (deg_Q : Q.degree = 4)
  (constant_terms_eq : P.coeff 0 = Q.coeff 0)
  (coeff_z_eq : P.coeff 1 = Q.coeff 1)
  (product_eq : P * Q = Polynomial.C 1 * 
    Polynomial.C 1 * Polynomial.C 1 * Polynomial.C (-1) *
    Polynomial.C 1) :
  Jaclyn's_constant_term = 3 :=
sorry

end NUMINAMATH_GPT_jaclyn_constant_term_l2059_205994


namespace NUMINAMATH_GPT_always_negative_l2059_205955

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (x ^ 2 + 1) - x) - Real.sin x

theorem always_negative (a b : ℝ) (ha : a ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
                     (hb : b ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
                     (hab : a + b ≠ 0) : 
  (f a + f b) / (a + b) < 0 := 
sorry

end NUMINAMATH_GPT_always_negative_l2059_205955


namespace NUMINAMATH_GPT_compute_g_five_times_l2059_205969

def g (x : ℤ) : ℤ :=
  if x ≥ 0 then -x^3 else x + 6

theorem compute_g_five_times :
  g (g (g (g (g 1)))) = -113 :=
  by sorry

end NUMINAMATH_GPT_compute_g_five_times_l2059_205969


namespace NUMINAMATH_GPT_find_total_shaded_area_l2059_205978

/-- Definition of the rectangles' dimensions and overlap conditions -/
def rect1_length : ℕ := 4
def rect1_width : ℕ := 15
def rect2_length : ℕ := 5
def rect2_width : ℕ := 10
def rect3_length : ℕ := 3
def rect3_width : ℕ := 18
def shared_side_length : ℕ := 4
def trip_overlap_width : ℕ := 3

/-- Calculation of the rectangular overlap using given conditions -/
theorem find_total_shaded_area : (rect1_length * rect1_width + rect2_length * rect2_width + rect3_length * rect3_width - shared_side_length * shared_side_length - trip_overlap_width * shared_side_length) = 136 :=
    by sorry

end NUMINAMATH_GPT_find_total_shaded_area_l2059_205978


namespace NUMINAMATH_GPT_students_wearing_blue_lipstick_l2059_205900

theorem students_wearing_blue_lipstick
  (total_students : ℕ)
  (half_students_wore_lipstick : total_students / 2 = 180)
  (red_fraction : ℚ)
  (pink_fraction : ℚ)
  (purple_fraction : ℚ)
  (green_fraction : ℚ)
  (students_wearing_red : red_fraction * 180 = 45)
  (students_wearing_pink : pink_fraction * 180 = 60)
  (students_wearing_purple : purple_fraction * 180 = 30)
  (students_wearing_green : green_fraction * 180 = 15)
  (total_red_fraction : red_fraction = 1 / 4)
  (total_pink_fraction : pink_fraction = 1 / 3)
  (total_purple_fraction : purple_fraction = 1 / 6)
  (total_green_fraction : green_fraction = 1 / 12) :
  (180 - (45 + 60 + 30 + 15) = 30) :=
by sorry

end NUMINAMATH_GPT_students_wearing_blue_lipstick_l2059_205900


namespace NUMINAMATH_GPT_problem_l2059_205922

theorem problem 
  (k a b c : ℝ)
  (h1 : (3 : ℝ)^2 - 7 * 3 + k = 0)
  (h2 : (a : ℝ)^2 - 7 * a + k = 0)
  (h3 : (b : ℝ)^2 - 8 * b + (k + 1) = 0)
  (h4 : (c : ℝ)^2 - 8 * c + (k + 1) = 0) :
  a + b * c = 17 := sorry

end NUMINAMATH_GPT_problem_l2059_205922


namespace NUMINAMATH_GPT_integer_solutions_l2059_205920

def satisfies_equation (x y : ℤ) : Prop := x^2 = y^2 * (x + y^4 + 2 * y^2)

theorem integer_solutions :
  {p : ℤ × ℤ | satisfies_equation p.1 p.2} = { (0, 0), (12, 2), (-8, 2) } :=
by sorry

end NUMINAMATH_GPT_integer_solutions_l2059_205920


namespace NUMINAMATH_GPT_specific_clothing_choice_probability_l2059_205902

noncomputable def probability_of_specific_clothing_choice : ℚ :=
  let total_clothing := 4 + 5 + 6
  let total_ways_to_choose_3 := Nat.choose 15 3
  let ways_to_choose_specific_3 := 4 * 5 * 6
  let probability := ways_to_choose_specific_3 / total_ways_to_choose_3
  probability

theorem specific_clothing_choice_probability :
  probability_of_specific_clothing_choice = 24 / 91 :=
by
  -- proof here 
  sorry

end NUMINAMATH_GPT_specific_clothing_choice_probability_l2059_205902


namespace NUMINAMATH_GPT_percentage_of_Muscovy_ducks_l2059_205966

theorem percentage_of_Muscovy_ducks
  (N : ℕ) (M : ℝ) (female_percentage : ℝ) (female_Muscovy : ℕ)
  (hN : N = 40)
  (hfemale_percentage : female_percentage = 0.30)
  (hfemale_Muscovy : female_Muscovy = 6)
  (hcondition : female_percentage * M * N = female_Muscovy) 
  : M = 0.5 := 
sorry

end NUMINAMATH_GPT_percentage_of_Muscovy_ducks_l2059_205966


namespace NUMINAMATH_GPT_croissants_left_l2059_205993

-- Definitions based on conditions
def total_croissants : ℕ := 17
def vegans : ℕ := 3
def allergic_to_chocolate : ℕ := 2
def any_type : ℕ := 2
def guests : ℕ := 7
def plain_needed : ℕ := vegans + allergic_to_chocolate
def plain_baked : ℕ := plain_needed
def choc_baked : ℕ := total_croissants - plain_baked

-- Assuming choc_baked > plain_baked as given
axiom croissants_greater_condition : choc_baked > plain_baked

-- Theorem to prove
theorem croissants_left (total_croissants vegans allergic_to_chocolate any_type guests : ℕ) 
    (plain_needed plain_baked choc_baked : ℕ) 
    (croissants_greater_condition : choc_baked > plain_baked) : 
    (choc_baked - guests + any_type) = 3 := 
by sorry

end NUMINAMATH_GPT_croissants_left_l2059_205993


namespace NUMINAMATH_GPT_rate_of_interest_l2059_205965

/-
Let P be the principal amount, SI be the simple interest paid, R be the rate of interest, and N be the number of years. 
The problem states:
- P = 1200
- SI = 432
- R = N

We need to prove that R = 6.
-/

theorem rate_of_interest (P SI R N : ℝ) (h1 : P = 1200) (h2 : SI = 432) (h3 : R = N) :
  R = 6 :=
  sorry

end NUMINAMATH_GPT_rate_of_interest_l2059_205965


namespace NUMINAMATH_GPT_intersection_complement_is_singleton_l2059_205911

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 2, 5}

theorem intersection_complement_is_singleton : (U \ M) ∩ N = {1} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_is_singleton_l2059_205911


namespace NUMINAMATH_GPT_polygon_sides_l2059_205974

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l2059_205974


namespace NUMINAMATH_GPT_sum_of_decimals_l2059_205924

theorem sum_of_decimals : 5.46 + 2.793 + 3.1 = 11.353 := by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l2059_205924


namespace NUMINAMATH_GPT_total_calories_in_jerrys_breakfast_l2059_205918

theorem total_calories_in_jerrys_breakfast :
  let pancakes := 7 * 120
  let bacon := 3 * 100
  let orange_juice := 2 * 300
  let cereal := 1 * 200
  let chocolate_muffin := 1 * 350
  pancakes + bacon + orange_juice + cereal + chocolate_muffin = 2290 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_calories_in_jerrys_breakfast_l2059_205918


namespace NUMINAMATH_GPT_nested_fraction_value_l2059_205954

theorem nested_fraction_value : 
  let expr := 1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))))
  expr = 21 / 55 :=
by 
  sorry

end NUMINAMATH_GPT_nested_fraction_value_l2059_205954


namespace NUMINAMATH_GPT_Roger_years_to_retire_l2059_205960

noncomputable def Peter : ℕ := 12
noncomputable def Robert : ℕ := Peter - 4
noncomputable def Mike : ℕ := Robert - 2
noncomputable def Tom : ℕ := 2 * Robert
noncomputable def Roger : ℕ := Peter + Tom + Robert + Mike

theorem Roger_years_to_retire :
  Roger = 42 → 50 - Roger = 8 := by
sorry

end NUMINAMATH_GPT_Roger_years_to_retire_l2059_205960


namespace NUMINAMATH_GPT_revenue_from_full_price_tickets_l2059_205999

noncomputable def full_price_ticket_revenue (f h p : ℕ) : ℕ := f * p

theorem revenue_from_full_price_tickets (f h p : ℕ) (total_tickets total_revenue : ℕ) 
  (tickets_eq : f + h = total_tickets)
  (revenue_eq : f * p + h * (p / 2) = total_revenue) 
  (total_tickets_value : total_tickets = 180)
  (total_revenue_value : total_revenue = 2652) :
  full_price_ticket_revenue f h p = 984 :=
by {
  sorry
}

end NUMINAMATH_GPT_revenue_from_full_price_tickets_l2059_205999


namespace NUMINAMATH_GPT_floyd_infinite_jumps_l2059_205913

def sum_of_digits (n: Nat) : Nat := 
  n.digits 10 |>.sum 

noncomputable def jumpable (a b: Nat) : Prop := 
  b > a ∧ b ≤ 2 * a 

theorem floyd_infinite_jumps :
  ∃ f : ℕ → ℕ, 
    (∀ n : ℕ, jumpable (f n) (f (n + 1))) ∧
    (∀ m n : ℕ, m ≠ n → sum_of_digits (f m) ≠ sum_of_digits (f n)) :=
sorry

end NUMINAMATH_GPT_floyd_infinite_jumps_l2059_205913


namespace NUMINAMATH_GPT_problem_discussion_organization_l2059_205942

theorem problem_discussion_organization 
    (students : Fin 20 → Finset (Fin 20))
    (problems : Fin 20 → Finset (Fin 20))
    (h1 : ∀ s, (students s).card = 2)
    (h2 : ∀ p, (problems p).card = 2)
    (h3 : ∀ s p, s ∈ problems p ↔ p ∈ students s) : 
    ∃ (discussion : Fin 20 → Fin 20), 
        (∀ s, discussion s ∈ students s) ∧ 
        (Finset.univ.image discussion).card = 20 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_problem_discussion_organization_l2059_205942


namespace NUMINAMATH_GPT_work_done_in_one_day_l2059_205938

theorem work_done_in_one_day (A_days B_days : ℝ) (hA : A_days = 6) (hB : B_days = A_days / 2) : 
  (1 / A_days + 1 / B_days) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_work_done_in_one_day_l2059_205938


namespace NUMINAMATH_GPT_max_min_values_l2059_205946

open Real

noncomputable def circle_condition (x y : ℝ) :=
  (x - 3) ^ 2 + (y - 3) ^ 2 = 6

theorem max_min_values (x y : ℝ) (hx : circle_condition x y) :
  ∃ k k' d d', 
    k = 3 + 2 * sqrt 2 ∧
    k' = 3 - 2 * sqrt 2 ∧
    k = y / x ∧
    k' = y / x ∧
    d = sqrt ((x - 2) ^ 2 + y ^ 2) ∧
    d' = sqrt ((x - 2) ^ 2 + y ^ 2) ∧
    d = sqrt (10) + sqrt (6) ∧
    d' = sqrt (10) - sqrt (6) :=
sorry

end NUMINAMATH_GPT_max_min_values_l2059_205946


namespace NUMINAMATH_GPT_train_speed_kmph_l2059_205919

theorem train_speed_kmph (length : ℝ) (time : ℝ) (speed_conversion : ℝ) (speed_kmph : ℝ) :
  length = 100.008 → time = 4 → speed_conversion = 3.6 →
  speed_kmph = (length / time) * speed_conversion → speed_kmph = 90.0072 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_kmph_l2059_205919


namespace NUMINAMATH_GPT_min_coins_for_less_than_1_dollar_l2059_205956

theorem min_coins_for_less_than_1_dollar :
  ∃ (p n q h : ℕ), 1*p + 5*n + 25*q + 50*h ≥ 1 ∧ 1*p + 5*n + 25*q + 50*h < 100 ∧ p + n + q + h = 8 :=
by 
  sorry

end NUMINAMATH_GPT_min_coins_for_less_than_1_dollar_l2059_205956


namespace NUMINAMATH_GPT_expression_parity_l2059_205943

theorem expression_parity (p m : ℤ) (hp : Odd p) : (Odd (p^3 + m * p)) ↔ Even m := by
  sorry

end NUMINAMATH_GPT_expression_parity_l2059_205943


namespace NUMINAMATH_GPT_part1_part2_l2059_205934

def f (x a : ℝ) : ℝ := |x + a| + |x - a^2|

theorem part1 (x : ℝ) : f x 1 ≥ 4 ↔ x ≤ -2 ∨ x ≥ 2 := sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, ∃ a : ℝ, -1 < a ∧ a < 3 ∧ m < f x a) ↔ m < 12 := sorry

end NUMINAMATH_GPT_part1_part2_l2059_205934


namespace NUMINAMATH_GPT_average_price_per_book_l2059_205909

-- Definitions of the conditions
def books_shop1 := 65
def cost_shop1 := 1480
def books_shop2 := 55
def cost_shop2 := 920

-- Definition of total values
def total_books := books_shop1 + books_shop2
def total_cost := cost_shop1 + cost_shop2

-- Proof statement
theorem average_price_per_book : (total_cost / total_books) = 20 := by
  sorry

end NUMINAMATH_GPT_average_price_per_book_l2059_205909


namespace NUMINAMATH_GPT_stamp_distribution_correct_l2059_205941

variables {W : ℕ} -- We use ℕ (natural numbers) for simplicity but this can be any type representing weight.

-- Number of envelopes that weigh less than W and need 2 stamps each
def envelopes_lt_W : ℕ := 6

-- Number of stamps per envelope if the envelope weighs less than W
def stamps_lt_W : ℕ := 2

-- Number of envelopes in total
def total_envelopes : ℕ := 14

-- Number of stamps for the envelopes that weigh less
def total_stamps_lt_W : ℕ := envelopes_lt_W * stamps_lt_W

-- Total stamps bought by Micah
def total_stamps_bought : ℕ := 52

-- Stamps left for envelopes that weigh more than W
def stamps_remaining : ℕ := total_stamps_bought - total_stamps_lt_W

-- Remaining envelopes that need stamps (those that weigh more than W)
def envelopes_gt_W : ℕ := total_envelopes - envelopes_lt_W

-- Number of stamps required per envelope that weighs more than W
def stamps_gt_W : ℕ := 5

-- Total stamps needed for the envelopes that weigh more than W
def total_stamps_needed_gt_W : ℕ := envelopes_gt_W * stamps_gt_W

theorem stamp_distribution_correct :
  total_stamps_bought = (total_stamps_lt_W + total_stamps_needed_gt_W) :=
by
  sorry

end NUMINAMATH_GPT_stamp_distribution_correct_l2059_205941


namespace NUMINAMATH_GPT_proof_problem_l2059_205975

def diamond (a b : ℚ) := a - (1 / b)

theorem proof_problem :
  ((diamond (diamond 2 4) 5) - (diamond 2 (diamond 4 5))) = (-71 / 380) := by
  sorry

end NUMINAMATH_GPT_proof_problem_l2059_205975


namespace NUMINAMATH_GPT_paint_cost_l2059_205957

theorem paint_cost {width height : ℕ} (price_per_quart coverage_area : ℕ) (total_cost : ℕ) :
  width = 5 → height = 4 → price_per_quart = 2 → coverage_area = 4 → total_cost = 20 :=
by
  intros h1 h2 h3 h4
  have area_one_side : ℕ := width * height
  have total_area : ℕ := 2 * area_one_side
  have quarts_needed : ℕ := total_area / coverage_area
  have cost : ℕ := quarts_needed * price_per_quart
  sorry

end NUMINAMATH_GPT_paint_cost_l2059_205957


namespace NUMINAMATH_GPT_fly_travel_time_to_opposite_vertex_l2059_205983

noncomputable def cube_side_length (a : ℝ) := 
  a

noncomputable def fly_travel_time_base := 4 -- minutes

noncomputable def fly_speed (a : ℝ) := 
  4 * a / fly_travel_time_base

noncomputable def space_diagonal_length (a : ℝ) := 
  a * Real.sqrt 3

theorem fly_travel_time_to_opposite_vertex (a : ℝ) : 
  fly_speed a ≠ 0 -> 
  space_diagonal_length a / fly_speed a = Real.sqrt 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fly_travel_time_to_opposite_vertex_l2059_205983


namespace NUMINAMATH_GPT_problem_ABC_sum_l2059_205971

-- Let A, B, and C be positive integers such that A and C, B and C, and A and B
-- have no common factor greater than 1.
-- If they satisfy the equation A * log_100 5 + B * log_100 4 = C,
-- then we need to prove that A + B + C = 4.

theorem problem_ABC_sum (A B C : ℕ) (h1 : 1 < A ∧ 1 < B ∧ 1 < C)
    (h2 : A.gcd B = 1 ∧ B.gcd C = 1 ∧ A.gcd C = 1)
    (h3 : A * Real.log 5 / Real.log 100 + B * Real.log 4 / Real.log 100 = C) :
    A + B + C = 4 :=
sorry

end NUMINAMATH_GPT_problem_ABC_sum_l2059_205971


namespace NUMINAMATH_GPT_domain_of_function_l2059_205925

theorem domain_of_function :
  ∀ x, (x - 2 > 0) ∧ (3 - x ≥ 0) ↔ 2 < x ∧ x ≤ 3 :=
by 
  intros x 
  simp only [and_imp, gt_iff_lt, sub_lt_iff_lt_add, sub_nonneg, le_iff_eq_or_lt, add_comm]
  exact sorry

end NUMINAMATH_GPT_domain_of_function_l2059_205925


namespace NUMINAMATH_GPT_find_m_l2059_205952

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 2}
noncomputable def N (m : ℝ) : Set ℝ := {x | x*x - m*x < 0}
noncomputable def M_inter_N (m : ℝ) : Set ℝ := {x | 0 < x ∧ x < 1}

theorem find_m (m : ℝ) (h : M ∩ (N m) = M_inter_N m) : m = 1 :=
by sorry

end NUMINAMATH_GPT_find_m_l2059_205952


namespace NUMINAMATH_GPT_number_of_undeveloped_sections_l2059_205915

def undeveloped_sections (total_area section_area : ℕ) : ℕ :=
  total_area / section_area

theorem number_of_undeveloped_sections :
  undeveloped_sections 7305 2435 = 3 :=
by
  unfold undeveloped_sections
  exact rfl

end NUMINAMATH_GPT_number_of_undeveloped_sections_l2059_205915


namespace NUMINAMATH_GPT_remainder_equality_l2059_205905

variables (A B D : ℕ) (S S' s s' : ℕ)

theorem remainder_equality 
  (h1 : A > B) 
  (h2 : (A + 3) % D = S) 
  (h3 : (B - 2) % D = S') 
  (h4 : ((A + 3) * (B - 2)) % D = s) 
  (h5 : (S * S') % D = s') : 
  s = s' := 
sorry

end NUMINAMATH_GPT_remainder_equality_l2059_205905


namespace NUMINAMATH_GPT_twenty_second_entry_l2059_205979

-- Definition of r_9 which is the remainder left when n is divided by 9
def r_9 (n : ℕ) : ℕ := n % 9

-- Statement to prove that the 22nd entry in the ordered list of all nonnegative integers
-- that satisfy r_9(5n) ≤ 4 is 38
theorem twenty_second_entry (n : ℕ) (hn : 5 * n % 9 ≤ 4) :
  ∃ m : ℕ, m = 22 ∧ n = 38 :=
sorry

end NUMINAMATH_GPT_twenty_second_entry_l2059_205979


namespace NUMINAMATH_GPT_num_four_digit_integers_divisible_by_7_l2059_205970

theorem num_four_digit_integers_divisible_by_7 :
  ∃ n : ℕ, n = 1286 ∧ ∀ k : ℕ, (1000 ≤ k ∧ k ≤ 9999) → (k % 7 = 0 ↔ ∃ m : ℕ, k = m * 7) :=
by {
  sorry
}

end NUMINAMATH_GPT_num_four_digit_integers_divisible_by_7_l2059_205970


namespace NUMINAMATH_GPT_particular_solution_exists_l2059_205997

noncomputable def general_solution (C : ℝ) (x : ℝ) : ℝ := C * x + 1

def differential_equation (x y y' : ℝ) : Prop := x * y' = y - 1

def initial_condition (y : ℝ) : Prop := y = 5

theorem particular_solution_exists :
  (∀ C x y, y = general_solution C x → differential_equation x y (C : ℝ)) →
  (∃ C, initial_condition (general_solution C 1)) →
  (∀ x, ∃ y, y = general_solution 4 x) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_particular_solution_exists_l2059_205997


namespace NUMINAMATH_GPT_derivative_correct_l2059_205937

noncomputable def f (x : ℝ) : ℝ := 
  (1 / (2 * Real.sqrt 2)) * (Real.sin (Real.log x) - (Real.sqrt 2 - 1) * Real.cos (Real.log x)) * x^(Real.sqrt 2 + 1)

noncomputable def df (x : ℝ) : ℝ := 
  (x^(Real.sqrt 2)) / (2 * Real.sqrt 2) * (2 * Real.cos (Real.log x) - Real.sqrt 2 * Real.cos (Real.log x) + 2 * Real.sqrt 2 * Real.sin (Real.log x))

theorem derivative_correct (x : ℝ) (hx : 0 < x) :
  deriv f x = df x := by
  sorry

end NUMINAMATH_GPT_derivative_correct_l2059_205937


namespace NUMINAMATH_GPT_find_a_in_triangle_l2059_205959

variable (a b c B : ℝ)

theorem find_a_in_triangle (h1 : b = Real.sqrt 3) (h2 : c = 3) (h3 : B = 30) :
    a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_find_a_in_triangle_l2059_205959


namespace NUMINAMATH_GPT_friends_came_over_later_l2059_205995

def original_friends : ℕ := 4
def total_people : ℕ := 7

theorem friends_came_over_later : (total_people - original_friends = 3) :=
sorry

end NUMINAMATH_GPT_friends_came_over_later_l2059_205995


namespace NUMINAMATH_GPT_twenty_cows_twenty_days_l2059_205996

-- Defining the initial conditions as constants
def num_cows : ℕ := 20
def days_one_cow_eats_one_bag : ℕ := 20
def bags_eaten_by_one_cow_in_days (d : ℕ) : ℕ := if d = days_one_cow_eats_one_bag then 1 else 0

-- Defining the total bags eaten by all cows
def total_bags_eaten_by_cows (cows : ℕ) (days : ℕ) : ℕ :=
  cows * (days / days_one_cow_eats_one_bag)

-- Statement to be proved: In 20 days, 20 cows will eat 20 bags of husk
theorem twenty_cows_twenty_days :
  total_bags_eaten_by_cows num_cows days_one_cow_eats_one_bag = 20 := sorry

end NUMINAMATH_GPT_twenty_cows_twenty_days_l2059_205996


namespace NUMINAMATH_GPT_proportion_solution_l2059_205947

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_proportion_solution_l2059_205947


namespace NUMINAMATH_GPT_hyperbola_real_axis_length_l2059_205973

theorem hyperbola_real_axis_length :
  (∃ (a b : ℝ), (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧ a = 3) →
  2 * 3 = 6 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_real_axis_length_l2059_205973


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l2059_205958

theorem trigonometric_identity_proof (α : ℝ) :
  3.3998 * (Real.cos α) ^ 4 - 4 * (Real.cos α) ^ 3 - 8 * (Real.cos α) ^ 2 + 3 * Real.cos α + 1 =
  -2 * Real.sin (7 * α / 2) * Real.sin (α / 2) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l2059_205958


namespace NUMINAMATH_GPT_ab_bc_ca_max_le_l2059_205963

theorem ab_bc_ca_max_le (a b c : ℝ) :
  ab + bc + ca + max (abs (a - b)) (max (abs (b - c)) (abs (c - a))) ≤
  1 + (1 / 3) * (a + b + c)^2 :=
sorry

end NUMINAMATH_GPT_ab_bc_ca_max_le_l2059_205963


namespace NUMINAMATH_GPT_sum_of_coordinates_of_C_parallelogram_l2059_205950

-- Definitions that encapsulate the given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨-1, 0⟩
def D : Point := ⟨5, -4⟩

-- The theorem we need to prove
theorem sum_of_coordinates_of_C_parallelogram :
  ∃ C : Point, C.x + C.y = 7 ∧
  ∃ M : Point, M = ⟨(A.x + D.x) / 2, (A.y + D.y) / 2⟩ ∧
  (M = ⟨(B.x + C.x) / 2, (B.y + C.y) / 2⟩) :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_C_parallelogram_l2059_205950


namespace NUMINAMATH_GPT_least_three_digit_multiple_of_8_l2059_205907

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end NUMINAMATH_GPT_least_three_digit_multiple_of_8_l2059_205907


namespace NUMINAMATH_GPT_starting_number_l2059_205932

theorem starting_number (n : ℕ) (h1 : n % 11 = 3) (h2 : (n + 11) % 11 = 3) (h3 : (n + 22) % 11 = 3) 
  (h4 : (n + 33) % 11 = 3) (h5 : (n + 44) % 11 = 3) (h6 : n + 44 ≤ 50) : n = 3 := 
sorry

end NUMINAMATH_GPT_starting_number_l2059_205932


namespace NUMINAMATH_GPT_farmer_rows_of_tomatoes_l2059_205936

def num_rows (total_tomatoes yield_per_plant plants_per_row : ℕ) : ℕ :=
  (total_tomatoes / yield_per_plant) / plants_per_row

theorem farmer_rows_of_tomatoes (total_tomatoes yield_per_plant plants_per_row : ℕ)
    (ht : total_tomatoes = 6000)
    (hy : yield_per_plant = 20)
    (hp : plants_per_row = 10) :
    num_rows total_tomatoes yield_per_plant plants_per_row = 30 := 
by
  sorry

end NUMINAMATH_GPT_farmer_rows_of_tomatoes_l2059_205936


namespace NUMINAMATH_GPT_total_number_of_subjects_l2059_205929

-- Definitions from conditions
def average_marks_5_subjects (total_marks : ℕ) : Prop :=
  74 * 5 = total_marks

def marks_in_last_subject (marks : ℕ) : Prop :=
  marks = 74

def total_average_marks (n : ℕ) (total_marks : ℕ) : Prop :=
  74 * n = total_marks

-- Lean 4 statement
theorem total_number_of_subjects (n total_marks total_marks_5 last_subject_marks : ℕ)
  (h1 : total_average_marks n total_marks)
  (h2 : average_marks_5_subjects total_marks_5)
  (h3 : marks_in_last_subject last_subject_marks)
  (h4 : total_marks = total_marks_5 + last_subject_marks) :
  n = 6 :=
sorry

end NUMINAMATH_GPT_total_number_of_subjects_l2059_205929


namespace NUMINAMATH_GPT_curve_touches_x_axis_at_most_three_times_l2059_205961

theorem curve_touches_x_axis_at_most_three_times
  (a b c d : ℝ) :
  ∃ (x : ℝ), (x^4 - x^5 + a * x^3 + b * x^2 + c * x + d = 0) → ∃ (y : ℝ), (y = 0) → 
  ∃(n : ℕ), (n ≤ 3) :=
by sorry

end NUMINAMATH_GPT_curve_touches_x_axis_at_most_three_times_l2059_205961


namespace NUMINAMATH_GPT_number_of_sets_l2059_205968

theorem number_of_sets (weight_per_rep reps total_weight : ℕ) 
  (h_weight_per_rep : weight_per_rep = 15)
  (h_reps : reps = 10)
  (h_total_weight : total_weight = 450) :
  (total_weight / (weight_per_rep * reps)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sets_l2059_205968


namespace NUMINAMATH_GPT_min_omega_l2059_205921

noncomputable def f (ω φ : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem min_omega (ω φ : ℝ) (hω : ω > 0)
  (h_sym : ∀ x : ℝ, f ω φ (2 * (π / 3) - x) = f ω φ x)
  (h_val : f ω φ (π / 12) = 0) :
  ω = 2 :=
sorry

end NUMINAMATH_GPT_min_omega_l2059_205921


namespace NUMINAMATH_GPT_trigonometric_identity_l2059_205992

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin θ * Real.sin (π / 2 - θ)) / (Real.sin θ ^ 2 + Real.cos (2 * θ) + Real.cos θ ^ 2) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2059_205992


namespace NUMINAMATH_GPT_number_of_students_l2059_205985

theorem number_of_students (total_students : ℕ) :
  (total_students = 19 * 6 + 4) ∧ 
  (∃ (x y : ℕ), x + y = 22 ∧ x > 7 ∧ total_students = x * 6 + y * 5) →
  total_students = 118 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l2059_205985


namespace NUMINAMATH_GPT_crayons_given_proof_l2059_205908

def initial_crayons : ℕ := 110
def total_lost_crayons : ℕ := 412
def more_lost_than_given : ℕ := 322

def G : ℕ := 45 -- This is the given correct answer to prove.

theorem crayons_given_proof :
  ∃ G : ℕ, (G + (G + more_lost_than_given)) = total_lost_crayons ∧ G = 45 :=
by
  sorry

end NUMINAMATH_GPT_crayons_given_proof_l2059_205908


namespace NUMINAMATH_GPT_angle_BAC_is_105_or_35_l2059_205910

-- Definitions based on conditions
def arcAB : ℝ := 110
def arcAC : ℝ := 40
def arcBC_major : ℝ := 360 - (arcAB + arcAC)
def arcBC_minor : ℝ := arcAB - arcAC

-- The conjecture: proving that the inscribed angle ∠BAC is 105° or 35° given the conditions.
theorem angle_BAC_is_105_or_35
  (h1 : 0 < arcAB ∧ arcAB < 360)
  (h2 : 0 < arcAC ∧ arcAC < 360)
  (h3 : arcAB + arcAC < 360) :
  (arcBC_major / 2 = 105) ∨ (arcBC_minor / 2 = 35) :=
  sorry

end NUMINAMATH_GPT_angle_BAC_is_105_or_35_l2059_205910


namespace NUMINAMATH_GPT_tan_negative_angle_l2059_205986

theorem tan_negative_angle (m : ℝ) (h1 : m = Real.cos (80 * Real.pi / 180)) (h2 : m = Real.sin (10 * Real.pi / 180)) :
  Real.tan (-440 * Real.pi / 180) = - (Real.sqrt (1 - m^2)) / m :=
by
  sorry

end NUMINAMATH_GPT_tan_negative_angle_l2059_205986


namespace NUMINAMATH_GPT_smallest_k_remainder_2_l2059_205916

theorem smallest_k_remainder_2 (k : ℕ) :
  k > 1 ∧
  k % 13 = 2 ∧
  k % 7 = 2 ∧
  k % 3 = 2 →
  k = 275 :=
by sorry

end NUMINAMATH_GPT_smallest_k_remainder_2_l2059_205916


namespace NUMINAMATH_GPT_modulus_of_z_l2059_205931

open Complex

theorem modulus_of_z 
  (z : ℂ) 
  (h : (1 - I) * z = 2 * I) : 
  abs z = Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_modulus_of_z_l2059_205931


namespace NUMINAMATH_GPT_range_omega_l2059_205953

noncomputable def f (ω x : ℝ) := Real.cos (ω * x + Real.pi / 6)

theorem range_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi → -1 ≤ f ω x ∧ f ω x ≤ Real.sqrt 3 / 2) →
  ω ∈ Set.Icc (5 / 6) (5 / 3) :=
  sorry

end NUMINAMATH_GPT_range_omega_l2059_205953
