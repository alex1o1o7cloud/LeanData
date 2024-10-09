import Mathlib

namespace strawberry_pancakes_l2307_230779

theorem strawberry_pancakes (total blueberry banana chocolate : ℕ) (h_total : total = 150) (h_blueberry : blueberry = 45) (h_banana : banana = 60) (h_chocolate : chocolate = 25) :
  total - (blueberry + banana + chocolate) = 20 :=
by
  sorry

end strawberry_pancakes_l2307_230779


namespace jorge_land_fraction_clay_rich_soil_l2307_230703

theorem jorge_land_fraction_clay_rich_soil 
  (total_acres : ℕ) 
  (yield_good_soil_per_acre : ℕ) 
  (yield_clay_soil_factor : ℕ)
  (total_yield : ℕ) 
  (fraction_clay_rich_soil : ℚ) :
  total_acres = 60 →
  yield_good_soil_per_acre = 400 →
  yield_clay_soil_factor = 2 →
  total_yield = 20000 →
  fraction_clay_rich_soil = 1/3 :=
by
  intro h_total_acres h_yield_good_soil_per_acre h_yield_clay_soil_factor h_total_yield
  -- math proof will be here
  sorry

end jorge_land_fraction_clay_rich_soil_l2307_230703


namespace problem_a_problem_b_problem_c_l2307_230734

open Real

noncomputable def conditions (x : ℝ) := x >= 1 / 2

/-- 
a) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = \sqrt{2} \)
valid if and only if x in [1/2, 1].
-/
theorem problem_a (x : ℝ) (h : conditions x) :
  (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = sqrt 2) ↔ (1 / 2 ≤ x ∧ x ≤ 1) :=
  sorry

/-- 
b) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = 1 \)
has no solution.
-/
theorem problem_b (x : ℝ) (h : conditions x) :
  sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 1 → False :=
  sorry

/-- 
c) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = 2 \)
if and only if x = 3/2.
-/
theorem problem_c (x : ℝ) (h : conditions x) :
  (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 2) ↔ (x = 3 / 2) :=
  sorry

end problem_a_problem_b_problem_c_l2307_230734


namespace curve_is_circle_l2307_230784

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b r : ℝ), (r > 0) ∧ ((x + a)^2 + (y + b)^2 = r^2) :=
by
  sorry

end curve_is_circle_l2307_230784


namespace carlos_laundry_time_l2307_230771

def washing_time1 := 30
def washing_time2 := 45
def washing_time3 := 40
def washing_time4 := 50
def washing_time5 := 35
def drying_time1 := 85
def drying_time2 := 95

def total_laundry_time := washing_time1 + washing_time2 + washing_time3 + washing_time4 + washing_time5 + drying_time1 + drying_time2

theorem carlos_laundry_time : total_laundry_time = 380 :=
by
  sorry

end carlos_laundry_time_l2307_230771


namespace length_of_square_side_l2307_230714

-- Definitions based on conditions
def perimeter_of_triangle : ℝ := 46
def total_perimeter : ℝ := 78
def perimeter_of_square : ℝ := total_perimeter - perimeter_of_triangle

-- Lean statement for the problem
theorem length_of_square_side : perimeter_of_square / 4 = 8 := by
  sorry

end length_of_square_side_l2307_230714


namespace expression_constant_value_l2307_230720

theorem expression_constant_value (a b x y : ℝ) 
  (h_a : a = Real.sqrt (1 + x^2))
  (h_b : b = Real.sqrt (1 + y^2)) 
  (h_xy : x + y = 1) : 
  (a + b + 1) * (a + b - 1) * (a - b + 1) * (-a + b + 1) = 4 := 
by 
  sorry

end expression_constant_value_l2307_230720


namespace part1_part2_part3_l2307_230789

noncomputable def y1 (x : ℝ) : ℝ := 0.1 * x + 15
noncomputable def y2 (x : ℝ) : ℝ := 0.15 * x

-- Prove that the functions are as described
theorem part1 : ∀ x : ℝ, y1 x = 0.1 * x + 15 ∧ y2 x = 0.15 * x :=
by sorry

-- Prove that x = 300 results in equal charges for Packages A and B
theorem part2 : y1 300 = y2 300 :=
by sorry

-- Prove that Package A is more cost-effective when x > 300
theorem part3 : ∀ x : ℝ, x > 300 → y1 x < y2 x :=
by sorry

end part1_part2_part3_l2307_230789


namespace find_S3m_l2307_230710
  
-- Arithmetic sequence with given properties
variable (m : ℕ)
variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

-- Define the conditions
axiom Sm : S m = 30
axiom S2m : S (2 * m) = 100

-- Problem statement to prove
theorem find_S3m : S (3 * m) = 170 :=
by
  sorry

end find_S3m_l2307_230710


namespace convert_sq_meters_to_hectares_convert_hours_to_hours_and_minutes_l2307_230704

theorem convert_sq_meters_to_hectares :
  (123000 / 10000) = 12.3 :=
by
  sorry

theorem convert_hours_to_hours_and_minutes :
  (4 + 0.25 * 60) = 4 * 60 + 15 :=
by
  sorry

end convert_sq_meters_to_hectares_convert_hours_to_hours_and_minutes_l2307_230704


namespace unique_digit_sum_l2307_230768

theorem unique_digit_sum (A B C D : ℕ) (h1 : A + B + C + D = 20) (h2 : B + A + 1 = 11) (uniq : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D)) : D = 8 :=
sorry

end unique_digit_sum_l2307_230768


namespace most_probable_hits_l2307_230706

variable (n : ℕ) (p : ℝ) (q : ℝ) (k : ℕ)
variable (h1 : n = 5) (h2 : p = 0.6) (h3 : q = 1 - p)

theorem most_probable_hits : k = 3 := by
  -- Define the conditions
  have hp : p = 0.6 := h2
  have hn : n = 5 := h1
  have hq : q = 1 - p := h3

  -- Set the expected value for the number of hits
  let expected := n * p

  -- Use the bounds for the most probable number of successes (k_0)
  have bounds := expected - q ≤ k ∧ k ≤ expected + p

  -- Proof step analysis can go here
  sorry

end most_probable_hits_l2307_230706


namespace number_of_students_in_all_events_l2307_230796

variable (T A B : ℕ)

-- Defining given conditions
-- Total number of students in the class
def total_students : ℕ := 45
-- Number of students participating in the Soccer event
def soccer_students : ℕ := 39
-- Number of students participating in the Basketball event
def basketball_students : ℕ := 28

-- Main theorem to prove
theorem number_of_students_in_all_events
  (h_total : T = total_students)
  (h_soccer : A = soccer_students)
  (h_basketball : B = basketball_students) :
  ∃ x : ℕ, x = A + B - T := sorry

end number_of_students_in_all_events_l2307_230796


namespace area_and_perimeter_l2307_230777

-- Given a rectangle R with length l and width w
variables (l w : ℝ)
-- Define the area of R
def area_R : ℝ := l * w

-- Define a smaller rectangle that is cut out, with an area A_cut
variables (A_cut : ℝ)
-- Define the area of the resulting figure S
def area_S : ℝ := area_R l w - A_cut

-- Define the perimeter of R
def perimeter_R : ℝ := 2 * l + 2 * w

-- perimeter_R remains the same after cutting out the smaller rectangle
theorem area_and_perimeter (h_cut : 0 < A_cut) (h_cut_le : A_cut ≤ area_R l w) : 
  (area_S l w A_cut < area_R l w) ∧ (perimeter_R l w = perimeter_R l w) :=
by
  sorry

end area_and_perimeter_l2307_230777


namespace car_rental_budget_l2307_230715

def daily_rental_cost : ℝ := 30.0
def cost_per_mile : ℝ := 0.18
def total_miles : ℝ := 250.0

theorem car_rental_budget : daily_rental_cost + (cost_per_mile * total_miles) = 75.0 :=
by 
  sorry

end car_rental_budget_l2307_230715


namespace john_spent_fraction_at_arcade_l2307_230725

theorem john_spent_fraction_at_arcade 
  (allowance : ℝ) (spent_arcade : ℝ) (spent_candy_store : ℝ) 
  (h1 : allowance = 3.45)
  (h2 : spent_candy_store = 0.92)
  (h3 : 3.45 - spent_arcade - (1/3) * (3.45 - spent_arcade) = spent_candy_store) :
  spent_arcade / allowance = 2.07 / 3.45 :=
by
  sorry

end john_spent_fraction_at_arcade_l2307_230725


namespace time_on_wednesday_is_40_minutes_l2307_230742

def hours_to_minutes (h : ℚ) : ℚ := h * 60

def time_monday : ℚ := hours_to_minutes (3 / 4)
def time_tuesday : ℚ := hours_to_minutes (1 / 2)
def time_wednesday (w : ℚ) : ℚ := w
def time_thursday : ℚ := hours_to_minutes (5 / 6)
def time_friday : ℚ := 75
def total_time : ℚ := hours_to_minutes 4

theorem time_on_wednesday_is_40_minutes (w : ℚ) 
    (h1 : time_monday = 45) 
    (h2 : time_tuesday = 30) 
    (h3 : time_thursday = 50) 
    (h4 : time_friday = 75)
    (h5 : total_time = 240) 
    (h6 : total_time = time_monday + time_tuesday + time_wednesday w + time_thursday + time_friday) 
    : w = 40 := 
by 
  sorry

end time_on_wednesday_is_40_minutes_l2307_230742


namespace polynomial_root_transformation_l2307_230781

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem polynomial_root_transformation :
  let P (a b c d e : ℝ) (x : ℂ) := (x^6 : ℂ) + (a : ℂ) * x^5 + (b : ℂ) * x^4 + (c : ℂ) * x^3 + (d : ℂ) * x^2 + (e : ℂ) * x + 4096
  (∀ r : ℂ, P 0 0 0 0 0 r = 0 → P 0 0 0 0 0 (ω * r) = 0) →
  ∃ a b c d e : ℝ, ∃ p : ℕ, p = 3 := sorry

end polynomial_root_transformation_l2307_230781


namespace cakes_left_l2307_230747

def initial_cakes : ℕ := 62
def additional_cakes : ℕ := 149
def cakes_sold : ℕ := 144

theorem cakes_left : (initial_cakes + additional_cakes) - cakes_sold = 67 :=
by
  sorry

end cakes_left_l2307_230747


namespace work_efficiency_ratio_l2307_230790

theorem work_efficiency_ratio (a b k : ℝ) (ha : a = k * b) (hb : b = 1/15)
  (hab : a + b = 1/5) : k = 2 :=
by sorry

end work_efficiency_ratio_l2307_230790


namespace cone_base_circumference_l2307_230773

theorem cone_base_circumference
  (V : ℝ) (h : ℝ) (C : ℝ)
  (volume_eq : V = 18 * Real.pi)
  (height_eq : h = 3) :
  C = 6 * Real.sqrt 2 * Real.pi :=
sorry

end cone_base_circumference_l2307_230773


namespace problem_1_l2307_230723

theorem problem_1 :
  (5 / ((1 / (1 * 2)) + (1 / (2 * 3)) + (1 / (3 * 4)) + (1 / (4 * 5)) + (1 / (5 * 6)))) = 6 := by
  sorry

end problem_1_l2307_230723


namespace inconsistent_b_positive_l2307_230793

theorem inconsistent_b_positive
  (a b c : ℝ)
  (h_solution_set : ∀ x : ℝ, -2 < x ∧ x < 3 / 2 → ax^2 + bx + c > 0) :
  ¬ b > 0 :=
sorry

end inconsistent_b_positive_l2307_230793


namespace driver_total_distance_is_148_l2307_230783

-- Definitions of the distances traveled according to the given conditions
def distance_MWF : ℕ := 12 * 3
def total_distance_MWF : ℕ := distance_MWF * 3
def distance_T : ℕ := 9 * 5 / 2  -- using ℕ for 2.5 hours as 5/2
def distance_Th : ℕ := 7 * 5 / 2

-- Statement of the total distance calculation
def total_distance_week : ℕ :=
  total_distance_MWF + distance_T + distance_Th

-- Theorem stating the total distance traveled during the week
theorem driver_total_distance_is_148 : total_distance_week = 148 := by
  sorry

end driver_total_distance_is_148_l2307_230783


namespace fraction_equivalence_l2307_230769

theorem fraction_equivalence : 
  (∀ (a b : ℕ), (a ≠ 0 ∧ b ≠ 0) → (15 * b = 25 * a ↔ a = 3 ∧ b = 5)) ∧
  (15 * 4 ≠ 25 * 3) ∧
  (15 * 3 ≠ 25 * 2) ∧
  (15 * 2 ≠ 25 * 1) ∧
  (15 * 7 ≠ 25 * 5) :=
by
  sorry

end fraction_equivalence_l2307_230769


namespace repeating_decimal_sum_l2307_230753

theorem repeating_decimal_sum :
  let x := (0.3333333333333333 : ℚ) -- 0.\overline{3}
  let y := (0.0707070707070707 : ℚ) -- 0.\overline{07}
  let z := (0.008008008008008 : ℚ)  -- 0.\overline{008}
  x + y + z = 418 / 999 := by
sorry

end repeating_decimal_sum_l2307_230753


namespace decrease_in_length_l2307_230722

theorem decrease_in_length (L B : ℝ) (h₀ : L ≠ 0) (h₁ : B ≠ 0)
  (h₂ : ∃ (A' : ℝ), A' = 0.72 * L * B)
  (h₃ : ∃ B' : ℝ, B' = B * 0.9) :
  ∃ (x : ℝ), x = 20 :=
by
  sorry

end decrease_in_length_l2307_230722


namespace sodium_bicarbonate_moles_l2307_230739

theorem sodium_bicarbonate_moles (HCl NaHCO3 CO2 : ℕ) (h1 : HCl = 1) (h2 : CO2 = 1) :
  NaHCO3 = 1 :=
by sorry

end sodium_bicarbonate_moles_l2307_230739


namespace cos_formula_of_tan_l2307_230756

theorem cos_formula_of_tan (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi) :
  Real.cos (5 * Real.pi / 2 + 2 * α) = -4 / 5 := 
  sorry

end cos_formula_of_tan_l2307_230756


namespace part1_part2_part3_l2307_230702

noncomputable def f : ℝ → ℝ := sorry -- Given f is a function on ℝ with domain (0, +∞)

axiom domain_pos (x : ℝ) : 0 < x
axiom pos_condition (x : ℝ) (h : 1 < x) : 0 < f x
axiom functional_eq (x y : ℝ) : f (x * y) = f x + f y
axiom specific_value : f (1/3) = -1

-- (1) Prove: f(1/x) = -f(x)
theorem part1 (x : ℝ) (hx : 0 < x) : f (1 / x) = - f x := sorry

-- (2) Prove: f(x) is an increasing function on its domain
theorem part2 (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (h : x1 < x2) : f x1 < f x2 := sorry

-- (3) Prove the range of x for the inequality
theorem part3 (x : ℝ) (hx : 0 < x) (hx2 : 0 < x - 2) : 
  f x - f (1 / (x - 2)) ≥ 2 ↔ 1 + Real.sqrt 10 ≤ x := sorry

end part1_part2_part3_l2307_230702


namespace solution_set_of_inequality_l2307_230772

theorem solution_set_of_inequality {x : ℝ} : 
  (|2 * x - 1| - |x - 2| < 0) → (-1 < x ∧ x < 1) :=
by
  sorry

end solution_set_of_inequality_l2307_230772


namespace password_probability_l2307_230749

theorem password_probability : 
  (5/10) * (51/52) * (9/10) = 459 / 1040 := by
  sorry

end password_probability_l2307_230749


namespace sum_of_solutions_comparison_l2307_230732

variable (a a' b b' c c' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0)

theorem sum_of_solutions_comparison :
  ( (c - b) / a > (c' - b') / a' ) ↔ ( (c'-b') / a' < (c-b) / a ) :=
by sorry

end sum_of_solutions_comparison_l2307_230732


namespace area_of_circle_l2307_230700

/-- Given a circle with circumference 36π, prove that the area is 324π. -/
theorem area_of_circle (C : ℝ) (hC : C = 36 * π) 
  (h1 : ∀ r : ℝ, C = 2 * π * r → 0 ≤ r)
  (h2 : ∀ r : ℝ, 0 ≤ r → ∃ (A : ℝ), A = π * r^2) :
  ∃ k : ℝ, (A = 324 * π → k = 324) := 
sorry


end area_of_circle_l2307_230700


namespace jonah_first_intermission_lemonade_l2307_230760

theorem jonah_first_intermission_lemonade :
  ∀ (l1 l2 l3 l_total : ℝ)
  (h1 : l2 = 0.42)
  (h2 : l3 = 0.25)
  (h3 : l_total = 0.92)
  (h4 : l_total = l1 + l2 + l3),
  l1 = 0.25 :=
by sorry

end jonah_first_intermission_lemonade_l2307_230760


namespace max_stories_on_odd_pages_l2307_230775

theorem max_stories_on_odd_pages 
    (stories : Fin 30 -> Fin 31) 
    (h_unique : Function.Injective stories) 
    (h_bounds : ∀ i, stories i < 31)
    : ∃ n, n = 23 ∧ ∃ f : Fin n -> Fin 30, ∀ j, f j % 2 = 1 := 
sorry

end max_stories_on_odd_pages_l2307_230775


namespace total_towels_folded_in_one_hour_l2307_230737

-- Define the conditions for folding rates and breaks of each person
def Jane_folding_rate (minutes : ℕ) : ℕ :=
  if minutes % 8 < 5 then 5 * (minutes / 8 + 1) else 5 * (minutes / 8)

def Kyla_folding_rate (minutes : ℕ) : ℕ :=
  if minutes < 30 then 12 * (minutes / 10 + 1) else 36 + 6 * ((minutes - 30) / 10)

def Anthony_folding_rate (minutes : ℕ) : ℕ :=
  if minutes <= 40 then 14 * (minutes / 20)
  else if minutes <= 50 then 28
  else 28 + 14 * ((minutes - 50) / 20)

def David_folding_rate (minutes : ℕ) : ℕ :=
  let sets := minutes / 15
  let additional := sets / 3
  4 * (sets - additional) + 5 * additional

-- Definitions are months passing given in the questions
def hours_fold_towels (minutes : ℕ) : ℕ :=
  Jane_folding_rate minutes + Kyla_folding_rate minutes + Anthony_folding_rate minutes + David_folding_rate minutes

theorem total_towels_folded_in_one_hour : hours_fold_towels 60 = 134 := sorry

end total_towels_folded_in_one_hour_l2307_230737


namespace tetrahedron_min_green_edges_l2307_230761

theorem tetrahedron_min_green_edges : 
  ∃ (green_edges : Finset (Fin 6)), 
  (∀ face : Finset (Fin 6), face.card = 3 → ∃ edge ∈ face, edge ∈ green_edges) ∧ green_edges.card = 3 :=
by sorry

end tetrahedron_min_green_edges_l2307_230761


namespace inequality_proof_l2307_230797

theorem inequality_proof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  (x * y / Real.sqrt (x * y + y * z) + y * z / Real.sqrt (y * z + z * x) + z * x / Real.sqrt (z * x + x * y)) 
  ≤ (Real.sqrt 2) / 2 :=
by
  sorry

end inequality_proof_l2307_230797


namespace solve_x_eqns_solve_y_eqns_l2307_230776

theorem solve_x_eqns : ∀ x : ℝ, 2 * x^2 = 8 * x ↔ (x = 0 ∨ x = 4) :=
by
  intro x
  sorry

theorem solve_y_eqns : ∀ y : ℝ, y^2 - 10 * y - 1 = 0 ↔ (y = 5 + Real.sqrt 26 ∨ y = 5 - Real.sqrt 26) :=
by
  intro y
  sorry

end solve_x_eqns_solve_y_eqns_l2307_230776


namespace fence_perimeter_l2307_230705

noncomputable def posts (n : ℕ) := 36
noncomputable def space_between_posts (d : ℕ) := 6
noncomputable def length_is_twice_width (l w : ℕ) := l = 2 * w

theorem fence_perimeter (n d w l perimeter : ℕ)
  (h1 : posts n = 36)
  (h2 : space_between_posts d = 6)
  (h3 : length_is_twice_width l w)
  : perimeter = 216 :=
sorry

end fence_perimeter_l2307_230705


namespace division_of_converted_values_l2307_230731

theorem division_of_converted_values 
  (h : 144 * 177 = 25488) : 
  254.88 / 0.177 = 1440 := by
  sorry

end division_of_converted_values_l2307_230731


namespace angle_ratio_l2307_230746

theorem angle_ratio (A B C : ℝ) (hA : A = 60) (hB : B = 80) (h_sum : A + B + C = 180) : B / C = 2 := by
  sorry

end angle_ratio_l2307_230746


namespace volume_uncovered_is_correct_l2307_230738

-- Define the volumes of the shoebox and the objects
def volume_shoebox : ℕ := 12 * 6 * 4
def volume_object1 : ℕ := 5 * 3 * 1
def volume_object2 : ℕ := 2 * 2 * 3
def volume_object3 : ℕ := 4 * 2 * 4

-- Define the total volume of the objects
def total_volume_objects : ℕ := volume_object1 + volume_object2 + volume_object3

-- Define the volume left uncovered
def volume_uncovered : ℕ := volume_shoebox - total_volume_objects

-- Prove that the volume left uncovered is 229 cubic inches
theorem volume_uncovered_is_correct : volume_uncovered = 229 := by
  -- This is where the proof would be written
  sorry

end volume_uncovered_is_correct_l2307_230738


namespace geometric_sequence_third_term_l2307_230748

theorem geometric_sequence_third_term (a b c d : ℕ) (r : ℕ) 
  (h₁ : d * r = 81) 
  (h₂ : 81 * r = 243) 
  (h₃ : r = 3) : c = 27 :=
by
  -- Insert proof here
  sorry

end geometric_sequence_third_term_l2307_230748


namespace smallest_n_for_inequality_l2307_230719

theorem smallest_n_for_inequality (n : ℕ) : 5 + 3 * n > 300 ↔ n = 99 := by
  sorry

end smallest_n_for_inequality_l2307_230719


namespace min_k_inequality_l2307_230709

theorem min_k_inequality (α β : ℝ) (hα : 0 < α) (hα2 : α < 2 * Real.pi / 3)
  (hβ : 0 < β) (hβ2 : β < 2 * Real.pi / 3) :
  4 * Real.cos α ^ 2 + 2 * Real.cos α * Real.cos β + 4 * Real.cos β ^ 2
  - 3 * Real.cos α - 3 * Real.cos β - 6 < 0 :=
by
  sorry

end min_k_inequality_l2307_230709


namespace exponent_subtraction_l2307_230754

theorem exponent_subtraction (a : ℝ) (m n : ℝ) (hm : a^m = 3) (hn : a^n = 5) : a^(m-n) = 3 / 5 := 
  sorry

end exponent_subtraction_l2307_230754


namespace net_population_increase_per_day_l2307_230787

def birth_rate : Nat := 4
def death_rate : Nat := 2
def seconds_per_day : Nat := 24 * 60 * 60

theorem net_population_increase_per_day : 
  (birth_rate - death_rate) * (seconds_per_day / 2) = 86400 := by
  sorry

end net_population_increase_per_day_l2307_230787


namespace compare_neg_fractions_l2307_230794

theorem compare_neg_fractions : (-3 / 5) < (-1 / 3) := 
by {
  sorry
}

end compare_neg_fractions_l2307_230794


namespace prove_total_payment_l2307_230751

-- Define the conditions under which the problem is set
def monthly_subscription_cost : ℝ := 14
def split_ratio : ℝ := 0.5
def months_in_year : ℕ := 12

-- Define the target amount to prove
def total_payment_after_one_year : ℝ := 84

-- Theorem statement
theorem prove_total_payment
  (h1: monthly_subscription_cost = 14)
  (h2: split_ratio = 0.5)
  (h3: months_in_year = 12) :
  monthly_subscription_cost * split_ratio * months_in_year = total_payment_after_one_year := 
  by
  sorry

end prove_total_payment_l2307_230751


namespace prime_fraction_identity_l2307_230708

theorem prime_fraction_identity : ∀ (p q : ℕ),
  Prime p → Prime q → p = 2 → q = 2 →
  (pq + p^p + q^q) / (p + q) = 3 :=
by
  intros p q hp hq hp2 hq2
  sorry

end prime_fraction_identity_l2307_230708


namespace employees_without_increase_l2307_230728

-- Define the constants and conditions
def total_employees : ℕ := 480
def salary_increase_percentage : ℕ := 10
def travel_allowance_increase_percentage : ℕ := 20

-- Define the calculations derived from conditions
def employees_with_salary_increase : ℕ := (salary_increase_percentage * total_employees) / 100
def employees_with_travel_allowance_increase : ℕ := (travel_allowance_increase_percentage * total_employees) / 100

-- Total employees who got increases assuming no overlap
def employees_with_increases : ℕ := employees_with_salary_increase + employees_with_travel_allowance_increase

-- The proof statement
theorem employees_without_increase :
  total_employees - employees_with_increases = 336 := by
  sorry

end employees_without_increase_l2307_230728


namespace calculate_xy_yz_zx_l2307_230782

variable (x y z : ℝ)

theorem calculate_xy_yz_zx (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h1 : x^2 + x * y + y^2 = 75)
    (h2 : y^2 + y * z + z^2 = 49)
    (h3 : z^2 + z * x + x^2 = 124) : 
    x * y + y * z + z * x = 70 :=
sorry

end calculate_xy_yz_zx_l2307_230782


namespace average_of_shifted_sample_l2307_230726

theorem average_of_shifted_sample (x1 x2 x3 : ℝ) (hx_avg : (x1 + x2 + x3) / 3 = 40) (hx_var : ((x1 - 40) ^ 2 + (x2 - 40) ^ 2 + (x3 - 40) ^ 2) / 3 = 1) : 
  ((x1 + 40) + (x2 + 40) + (x3 + 40)) / 3 = 80 :=
sorry

end average_of_shifted_sample_l2307_230726


namespace speed_difference_l2307_230798

theorem speed_difference (h_cyclist : 88 / 8 = 11) (h_car : 48 / 8 = 6) :
  (11 - 6 = 5) :=
by
  sorry

end speed_difference_l2307_230798


namespace intersection_A_B_l2307_230712

open Set

def universal_set : Set ℤ := {x | 1 ≤ x ∧ x ≤ 5}
def A : Set ℤ := {1, 2, 3}
def complement_B : Set ℤ := {1, 2}
def B : Set ℤ := universal_set \ complement_B

theorem intersection_A_B : A ∩ B = {3} :=
by
  sorry

end intersection_A_B_l2307_230712


namespace carol_carrots_l2307_230721

def mother_picked := 16
def good_carrots := 38
def bad_carrots := 7
def total_carrots := good_carrots + bad_carrots
def carol_picked : Nat := total_carrots - mother_picked

theorem carol_carrots : carol_picked = 29 := by
  sorry

end carol_carrots_l2307_230721


namespace num_ways_choose_pair_of_diff_color_socks_l2307_230718

-- Define the numbers of socks of each color
def num_white := 5
def num_brown := 5
def num_blue := 3
def num_black := 3

-- Define the calculation for pairs of different colored socks
def num_pairs_white_brown := num_white * num_brown
def num_pairs_brown_blue := num_brown * num_blue
def num_pairs_white_blue := num_white * num_blue
def num_pairs_white_black := num_white * num_black
def num_pairs_brown_black := num_brown * num_black
def num_pairs_blue_black := num_blue * num_black

-- Define the total number of pairs
def total_pairs := num_pairs_white_brown + num_pairs_brown_blue + num_pairs_white_blue + num_pairs_white_black + num_pairs_brown_black + num_pairs_blue_black

-- The theorem to be proved
theorem num_ways_choose_pair_of_diff_color_socks : total_pairs = 94 := by
  -- Since we do not need to include the proof steps, we use sorry
  sorry

end num_ways_choose_pair_of_diff_color_socks_l2307_230718


namespace jose_profit_share_l2307_230785

theorem jose_profit_share (investment_tom : ℕ) (months_tom : ℕ) 
                         (investment_jose : ℕ) (months_jose : ℕ) 
                         (total_profit : ℕ) :
                         investment_tom = 30000 →
                         months_tom = 12 →
                         investment_jose = 45000 →
                         months_jose = 10 →
                         total_profit = 63000 →
                         (investment_jose * months_jose / 
                         (investment_tom * months_tom + investment_jose * months_jose)) * total_profit = 35000 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  norm_num
  sorry

end jose_profit_share_l2307_230785


namespace perimeter_eq_120_plus_2_sqrt_1298_l2307_230758

noncomputable def total_perimeter_of_two_quadrilaterals (AB BC CD : ℝ) (AC : ℝ := Real.sqrt (AB ^ 2 + BC ^ 2)) (AD : ℝ := Real.sqrt (AC ^ 2 + CD ^ 2)) : ℝ :=
2 * (AB + BC + CD + AD)

theorem perimeter_eq_120_plus_2_sqrt_1298 (hAB : AB = 15) (hBC : BC = 28) (hCD : CD = 17) :
  total_perimeter_of_two_quadrilaterals 15 28 17 = 120 + 2 * Real.sqrt 1298 :=
by
  sorry

end perimeter_eq_120_plus_2_sqrt_1298_l2307_230758


namespace sum_is_ten_l2307_230744

variable (x y : ℝ) (S : ℝ)

-- Conditions
def condition1 : Prop := x + y = S
def condition2 : Prop := x = 25 / y
def condition3 : Prop := x^2 + y^2 = 50

-- Theorem
theorem sum_is_ten (h1 : condition1 x y S) (h2 : condition2 x y) (h3 : condition3 x y) : S = 10 :=
sorry

end sum_is_ten_l2307_230744


namespace geometric_series_squares_sum_l2307_230778

theorem geometric_series_squares_sum (a : ℝ) (r : ℝ) (h : -1 < r ∧ r < 1) :
  (∑' n : ℕ, (a * r^n)^2) = a^2 / (1 - r^2) :=
by sorry

end geometric_series_squares_sum_l2307_230778


namespace average_of_three_l2307_230788

theorem average_of_three {a b c d e : ℚ}
    (h1 : (a + b + c + d + e) / 5 = 12)
    (h2 : (d + e) / 2 = 24) :
    (a + b + c) / 3 = 4 := by
  sorry

end average_of_three_l2307_230788


namespace taxi_trip_miles_l2307_230740

theorem taxi_trip_miles 
  (initial_fee : ℝ := 2.35)
  (additional_charge : ℝ := 0.35)
  (segment_length : ℝ := 2/5)
  (total_charge : ℝ := 5.50) :
  ∃ (miles : ℝ), total_charge = initial_fee + additional_charge * (miles / segment_length) ∧ miles = 3.6 :=
by
  sorry

end taxi_trip_miles_l2307_230740


namespace intersecting_lines_k_value_l2307_230752

theorem intersecting_lines_k_value :
  ∃ k : ℚ, (∀ x y : ℚ, y = 3 * x + 12 ∧ y = -5 * x - 7 → y = 2 * x + k) → k = 77 / 8 :=
sorry

end intersecting_lines_k_value_l2307_230752


namespace hyungjun_initial_ribbon_length_l2307_230707

noncomputable def initial_ribbon_length (R: ℝ) : Prop :=
  let used_for_first_box := R / 2 + 2000
  let remaining_after_first := R - used_for_first_box
  let used_for_second_box := (remaining_after_first / 2) + 2000
  remaining_after_first - used_for_second_box = 0

theorem hyungjun_initial_ribbon_length : ∃ R: ℝ, initial_ribbon_length R ∧ R = 12000 :=
  by
  exists 12000
  unfold initial_ribbon_length
  simp
  sorry

end hyungjun_initial_ribbon_length_l2307_230707


namespace solve_for_c_l2307_230727

theorem solve_for_c (a b c : ℝ) (B : ℝ) (ha : a = 4) (hb : b = 2*Real.sqrt 7) (hB : B = Real.pi / 3) : 
  (c^2 - 4*c - 12 = 0) → c = 6 :=
by 
  intro h
  -- Details of the proof would be here
  sorry

end solve_for_c_l2307_230727


namespace sum_quotient_reciprocal_eq_one_point_thirty_five_l2307_230759

theorem sum_quotient_reciprocal_eq_one_point_thirty_five (x y : ℝ)
  (h1 : x + y = 45) (h2 : x * y = 500) : x / y + 1 / x + 1 / y = 1.35 := by
  -- Proof details would go here
  sorry

end sum_quotient_reciprocal_eq_one_point_thirty_five_l2307_230759


namespace sum_of_their_ages_now_l2307_230770

variable (Nacho Divya : ℕ)

-- Conditions
def divya_current_age := 5
def nacho_in_5_years := 3 * (divya_current_age + 5)

-- Definition to determine current age of Nacho
def nacho_current_age := nacho_in_5_years - 5

-- Sum of current ages
def sum_of_ages := divya_current_age + nacho_current_age

-- Theorem to prove the sum of their ages now is 30
theorem sum_of_their_ages_now : sum_of_ages = 30 :=
by
  sorry

end sum_of_their_ages_now_l2307_230770


namespace final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r_l2307_230736

variable (k r s N : ℝ)
variable (h_pos_k : 0 < k)
variable (h_pos_r : 0 < r)
variable (h_pos_s : 0 < s)
variable (h_pos_N : 0 < N)
variable (h_r_lt_80 : r < 80)

theorem final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r :
  N * (1 + k / 100) * (1 - r / 100) + 10 * s > N ↔ k > 100 * r / (100 - r) :=
sorry

end final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r_l2307_230736


namespace carol_weight_l2307_230733

variable (a c : ℝ)

-- Conditions based on the problem statement
def combined_weight : Prop := a + c = 280
def weight_difference : Prop := c - a = c / 3

theorem carol_weight (h1 : combined_weight a c) (h2 : weight_difference a c) : c = 168 :=
by
  -- Proof goes here
  sorry

end carol_weight_l2307_230733


namespace trig_identity_simplification_l2307_230755

theorem trig_identity_simplification (θ : ℝ) (hθ : θ = 15 * Real.pi / 180) :
  (Real.sqrt 3 / 2 - Real.sqrt 3 * (Real.sin θ) ^ 2) = 3 / 4 := 
by sorry

end trig_identity_simplification_l2307_230755


namespace geometric_sequence_k_value_l2307_230750

theorem geometric_sequence_k_value (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ)
  (hS : ∀ n, S n = k + 3^n)
  (h_geom : ∀ n, a (n+1) = S (n+1) - S n)
  (h_geo_seq : ∀ n, a (n+2) / a (n+1) = a (n+1) / a n) :
  k = -1 := by
  sorry

end geometric_sequence_k_value_l2307_230750


namespace xiao_ming_water_usage_ge_8_l2307_230799

def min_monthly_water_usage (x : ℝ) : Prop :=
  ∀ (c : ℝ), c ≥ 15 →
    (c = if x ≤ 5 then x * 1.8 else (5 * 1.8 + (x - 5) * 2)) →
      x ≥ 8

theorem xiao_ming_water_usage_ge_8 : ∃ x : ℝ, min_monthly_water_usage x :=
  sorry

end xiao_ming_water_usage_ge_8_l2307_230799


namespace polynomial_root_fraction_l2307_230730

theorem polynomial_root_fraction (p q r s : ℝ) (h : p ≠ 0) 
    (h1 : p * 4^3 + q * 4^2 + r * 4 + s = 0)
    (h2 : p * (-3)^3 + q * (-3)^2 + r * (-3) + s = 0) : 
    (q + r) / p = -13 :=
by
  sorry

end polynomial_root_fraction_l2307_230730


namespace distinct_triplet_inequality_l2307_230724

theorem distinct_triplet_inequality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  abs (a / (b - c)) + abs (b / (c - a)) + abs (c / (a - b)) ≥ 2 := 
sorry

end distinct_triplet_inequality_l2307_230724


namespace salt_fraction_l2307_230764

variables {a x : ℝ}

-- First condition: the shortfall in salt the first time
def shortfall_first (a x : ℝ) : ℝ := a - x

-- Second condition: the shortfall in salt the second time
def shortfall_second (a x : ℝ) : ℝ := a - 2 * x

-- Third condition: relationship given by the problem
axiom condition : shortfall_first a x = 2 * shortfall_second a x

-- Prove fraction of necessary salt added the first time is 1/3
theorem salt_fraction (a x : ℝ) (h : shortfall_first a x = 2 * shortfall_second a x) : x = a / 3 :=
by
  sorry

end salt_fraction_l2307_230764


namespace vodka_shot_size_l2307_230743

theorem vodka_shot_size (x : ℝ) (h1 : 8 / 2 = 4) (h2 : 4 * x = 2 * 3) : x = 1.5 :=
by
  sorry

end vodka_shot_size_l2307_230743


namespace dorothy_and_jemma_sales_l2307_230741

theorem dorothy_and_jemma_sales :
  ∀ (frames_sold_by_jemma price_per_frame_jemma : ℕ)
  (price_per_frame_dorothy frames_sold_by_dorothy : ℚ)
  (total_sales_jemma total_sales_dorothy total_sales : ℚ),
  price_per_frame_jemma = 5 →
  frames_sold_by_jemma = 400 →
  price_per_frame_dorothy = price_per_frame_jemma / 2 →
  frames_sold_by_jemma = 2 * frames_sold_by_dorothy →
  total_sales_jemma = frames_sold_by_jemma * price_per_frame_jemma →
  total_sales_dorothy = frames_sold_by_dorothy * price_per_frame_dorothy →
  total_sales = total_sales_jemma + total_sales_dorothy →
  total_sales = 2500 := by
  sorry

end dorothy_and_jemma_sales_l2307_230741


namespace not_p_suff_not_q_l2307_230765

theorem not_p_suff_not_q (x : ℝ) :
  ¬(|x| ≥ 1) → ¬(x^2 + x - 6 ≥ 0) :=
sorry

end not_p_suff_not_q_l2307_230765


namespace find_constant_term_l2307_230711

theorem find_constant_term (x y C : ℤ) 
    (h1 : 5 * x + y = 19) 
    (h2 : 3 * x + 2 * y = 10) 
    (h3 : C = x + 3 * y) 
    : C = 1 := 
by 
  sorry

end find_constant_term_l2307_230711


namespace mixed_number_calculation_l2307_230745

theorem mixed_number_calculation :
  (481 + 1/6) + (265 + 1/12) + (904 + 1/20) - (184 + 29/30) - (160 + 41/42) - (703 + 55/56) = 603 + 3/8 := 
sorry

end mixed_number_calculation_l2307_230745


namespace symmetric_line_equation_l2307_230757

theorem symmetric_line_equation : 
  ∀ (P : ℝ × ℝ) (L : ℝ × ℝ × ℝ), 
  P = (1, 1) → 
  L = (2, 3, -6) → 
  (∃ (a b c : ℝ), a * 1 + b * 1 + c = 0 → a * x + b * y + c = 0 ↔ 2 * x + 3 * y - 4 = 0) 
:= 
sorry

end symmetric_line_equation_l2307_230757


namespace point_slope_intersection_lines_l2307_230766

theorem point_slope_intersection_lines : 
  ∀ s : ℝ, ∃ x y : ℝ, 2*x - 3*y = 8*s + 6 ∧ x + 2*y = 3*s - 1 ∧ y = -((2*x)/25 + 182/175) := 
sorry

end point_slope_intersection_lines_l2307_230766


namespace burgers_per_day_l2307_230786

def calories_per_burger : ℝ := 20
def total_calories_after_two_days : ℝ := 120

theorem burgers_per_day :
  total_calories_after_two_days / (2 * calories_per_burger) = 3 := 
by
  sorry

end burgers_per_day_l2307_230786


namespace min_value_expression_l2307_230792

theorem min_value_expression (x : ℝ) (h : x ≠ -7) : 
  ∃ y, y = 1 ∧ ∀ z, z = (2 * x ^ 2 + 98) / ((x + 7) ^ 2) → y ≤ z := 
sorry

end min_value_expression_l2307_230792


namespace factor_expression_l2307_230795

theorem factor_expression (x y a b : ℝ) : 
  ∃ f : ℝ, 3 * x * (a - b) - 9 * y * (b - a) = f * (x + 3 * y) ∧ f = 3 * (a - b) :=
by
  sorry

end factor_expression_l2307_230795


namespace correctPairsAreSkating_l2307_230762

def Friend := String
def Brother := String

structure SkatingPair where
  gentleman : Friend
  lady : Friend

-- Define the list of friends with their brothers
def friends : List Friend := ["Lyusya Egorova", "Olya Petrova", "Inna Krymova", "Anya Vorobyova"]
def brothers : List Brother := ["Andrey Egorov", "Serezha Petrov", "Dima Krymov", "Yura Vorobyov"]

-- Condition: The skating pairs such that gentlemen are taller than ladies and no one skates with their sibling
noncomputable def skatingPairs : List SkatingPair :=
  [ {gentleman := "Yura Vorobyov", lady := "Lyusya Egorova"},
    {gentleman := "Andrey Egorov", lady := "Olya Petrova"},
    {gentleman := "Serezha Petrov", lady := "Inna Krymova"},
    {gentleman := "Dima Krymov", lady := "Anya Vorobyova"} ]

-- Proving that the pairs are exactly as specified.
theorem correctPairsAreSkating :
  skatingPairs = 
    [ {gentleman := "Yura Vorobyov", lady := "Lyusya Egorova"},
      {gentleman := "Andrey Egorov", lady := "Olya Petrova"},
      {gentleman := "Serezha Petrov", lady := "Inna Krymova"},
      {gentleman := "Dima Krymov", lady := "Anya Vorobyova"} ] :=
by
  sorry

end correctPairsAreSkating_l2307_230762


namespace area_of_rectangle_l2307_230767

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l2307_230767


namespace range_of_a_l2307_230717

theorem range_of_a (a : ℝ) (x : ℝ) (h : x^2 + a * x + 1 < 0) : a < -2 ∨ a > 2 :=
sorry

end range_of_a_l2307_230717


namespace problem_solution_l2307_230774

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x - a

theorem problem_solution (x₀ x₁ a : ℝ) (h₁ : 3 * x₀^2 - 2 * x₀ + a = 0) (h₂ : f x₁ a = f x₀ a) (h₃ : x₁ ≠ x₀) : x₁ + 2 * x₀ = 1 :=
by
  sorry

end problem_solution_l2307_230774


namespace simplify_exponent_multiplication_l2307_230780

theorem simplify_exponent_multiplication :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_multiplication_l2307_230780


namespace original_numbers_placement_l2307_230763

-- Define each letter stands for a given number
def A : ℕ := 1
def B : ℕ := 3
def C : ℕ := 2
def D : ℕ := 5
def E : ℕ := 6
def F : ℕ := 4

-- Conditions provided
def white_triangle_condition (x y z : ℕ) : Prop :=
x + y = z

-- Main problem reformulated as theorem
theorem original_numbers_placement :
  (A = 1) ∧ (B = 3) ∧ (C = 2) ∧ (D = 5) ∧ (E = 6) ∧ (F = 4) :=
sorry

end original_numbers_placement_l2307_230763


namespace quadratic_inequality_solution_l2307_230701

theorem quadratic_inequality_solution {x : ℝ} :
  (x^2 + x - 6 ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l2307_230701


namespace compute_value_l2307_230729

theorem compute_value {a b : ℝ} 
  (h1 : ∀ x, (x + a) * (x + b) * (x + 12) = 0 → x ≠ -3 → x = -a ∨ x = -b ∨ x = -12)
  (h2 : ∀ x, (x + 2 * a) * (x + 3) * (x + 6) = 0 → x ≠ -b ∧ x ≠ -12 → x = -3) :
  100 * (3 / 2) + 6 = 156 :=
by
  sorry

end compute_value_l2307_230729


namespace gcd_polynomials_l2307_230716

theorem gcd_polynomials (b : ℤ) (h : b % 8213 = 0 ∧ b % 2 = 1) :
  Int.gcd (8 * b^2 + 63 * b + 144) (2 * b + 15) = 9 :=
sorry

end gcd_polynomials_l2307_230716


namespace single_shot_decrease_l2307_230735

theorem single_shot_decrease (S : ℝ) (r1 r2 r3 : ℝ) (h1 : r1 = 0.05) (h2 : r2 = 0.10) (h3 : r3 = 0.15) :
  (1 - (1 - r1) * (1 - r2) * (1 - r3)) * 100 = 27.325 := 
by
  sorry

end single_shot_decrease_l2307_230735


namespace sugar_percentage_l2307_230713

theorem sugar_percentage (x : ℝ) (h2 : 50 ≤ 100) (h1 : 1 / 4 * x + 12.5 = 20) : x = 10 :=
by
  sorry

end sugar_percentage_l2307_230713


namespace solve_problem_l2307_230791

noncomputable def problem_statement : Prop :=
  ∀ (a b c : ℕ),
    (a ≤ b) →
    (b ≤ c) →
    Nat.gcd (Nat.gcd a b) c = 1 →
    (a^2 * b) ∣ (a^3 + b^3 + c^3) →
    (b^2 * c) ∣ (a^3 + b^3 + c^3) →
    (c^2 * a) ∣ (a^3 + b^3 + c^3) →
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 2 ∧ c = 3)

-- Here we declare the main theorem but skip the proof.
theorem solve_problem : problem_statement :=
by sorry

end solve_problem_l2307_230791
