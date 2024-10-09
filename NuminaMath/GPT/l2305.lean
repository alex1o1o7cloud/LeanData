import Mathlib

namespace unit_price_quantity_inverse_proportion_map_distance_actual_distance_direct_proportion_l2305_230522

-- Definitions based on conditions
variable (unit_price quantity total_price : ℕ)
variable (map_distance actual_distance scale : ℕ)

-- Given conditions
def total_price_fixed := unit_price * quantity = total_price
def scale_fixed := map_distance * scale = actual_distance

-- Proof problem statements
theorem unit_price_quantity_inverse_proportion (h : total_price_fixed unit_price quantity total_price) :
  ∃ k : ℕ, unit_price = k / quantity := sorry

theorem map_distance_actual_distance_direct_proportion (h : scale_fixed map_distance actual_distance scale) :
  ∃ k : ℕ, map_distance * scale = k * actual_distance := sorry

end unit_price_quantity_inverse_proportion_map_distance_actual_distance_direct_proportion_l2305_230522


namespace combined_weight_of_boxes_l2305_230535

def first_box_weight := 2
def second_box_weight := 11
def last_box_weight := 5

theorem combined_weight_of_boxes :
  first_box_weight + second_box_weight + last_box_weight = 18 := by
  sorry

end combined_weight_of_boxes_l2305_230535


namespace triangle_inequality_difference_l2305_230514

theorem triangle_inequality_difference :
  (∀ (x : ℤ), (x + 7 > 9) ∧ (x + 9 > 7) ∧ (7 + 9 > x) → (3 ≤ x ∧ x ≤ 15) ∧ (15 - 3 = 12)) :=
by
  sorry

end triangle_inequality_difference_l2305_230514


namespace liam_finishes_on_wednesday_l2305_230592

theorem liam_finishes_on_wednesday :
  let start_day := 3  -- Wednesday, where 0 represents Sunday
  let total_books := 20
  let total_days := (total_books * (total_books + 1)) / 2
  (total_days % 7) = 0 :=
by sorry

end liam_finishes_on_wednesday_l2305_230592


namespace intersection_M_N_l2305_230539

noncomputable def set_M : Set ℝ := {x | x^2 - 3 * x - 4 ≤ 0}
noncomputable def set_N : Set ℝ := {x | Real.log x ≥ 0}

theorem intersection_M_N :
  {x | x ∈ set_M ∧ x ∈ set_N} = {x | 1 ≤ x ∧ x ≤ 4} :=
sorry

end intersection_M_N_l2305_230539


namespace add_two_inequality_l2305_230588

theorem add_two_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 :=
sorry

end add_two_inequality_l2305_230588


namespace man_l2305_230545

theorem man's_speed_downstream (v : ℕ) (h1 : v - 3 = 8) (s : ℕ := 3) : v + s = 14 :=
by
  sorry

end man_l2305_230545


namespace channels_taken_away_l2305_230552

theorem channels_taken_away (X : ℕ) : 
  (150 - X + 12 - 10 + 8 + 7 = 147) -> X = 20 :=
by
  sorry

end channels_taken_away_l2305_230552


namespace incorrect_statement_d_l2305_230530

noncomputable def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem incorrect_statement_d (n : ℤ) :
  (n < cbrt 9 ∧ cbrt 9 < n+1) → n ≠ 3 :=
by
  intro h
  have h2 : (2 : ℤ) < cbrt 9 := sorry
  have h3 : cbrt 9 < (3 : ℤ) := sorry
  exact sorry

end incorrect_statement_d_l2305_230530


namespace student_needs_33_percent_to_pass_l2305_230518

-- Define the conditions
def obtained_marks : ℕ := 125
def failed_by : ℕ := 40
def max_marks : ℕ := 500

-- The Lean statement to prove the required percentage
theorem student_needs_33_percent_to_pass : (obtained_marks + failed_by) * 100 / max_marks = 33 := by
  sorry

end student_needs_33_percent_to_pass_l2305_230518


namespace middle_school_mentoring_l2305_230512

theorem middle_school_mentoring (s n : ℕ) (h1 : s ≠ 0) (h2 : n ≠ 0) 
  (h3 : (n : ℚ) / 3 = (2 : ℚ) * (s : ℚ) / 5) : 
  (n / 3 + 2 * s / 5) / (n + s) = 4 / 11 := by
  sorry

end middle_school_mentoring_l2305_230512


namespace angle_BDC_is_55_l2305_230564

def right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] : Prop :=
  ∃ (angle_A angle_B angle_C : ℝ), angle_A + angle_B + angle_C = 180 ∧
  angle_A = 20 ∧ angle_C = 90

def bisector (B D : Type) [Inhabited B] [Inhabited D] (angle_ABC : ℝ) : Prop :=
  ∃ (angle_DBC : ℝ), angle_DBC = angle_ABC / 2

theorem angle_BDC_is_55 (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] :
  right_triangle A B C →
  bisector B D 70 →
  ∃ angle_BDC : ℝ, angle_BDC = 55 :=
by sorry

end angle_BDC_is_55_l2305_230564


namespace jessie_final_position_l2305_230591

theorem jessie_final_position :
  ∃ y : ℕ,
  (0 + 6 * 4 = 24) ∧
  (y = 24) :=
by
  sorry

end jessie_final_position_l2305_230591


namespace evaluate_expr_l2305_230554

theorem evaluate_expr :
  (150^2 - 12^2) / (90^2 - 21^2) * ((90 + 21) * (90 - 21)) / ((150 + 12) * (150 - 12)) = 2 :=
by sorry

end evaluate_expr_l2305_230554


namespace birch_trees_probability_l2305_230523

/--
A gardener plants four pine trees, five oak trees, and six birch trees in a row. He plants them in random order, each arrangement being equally likely.
Prove that no two birch trees are next to one another is \(\frac{2}{45}\).
--/
theorem birch_trees_probability: (∃ (m n : ℕ), (m = 2) ∧ (n = 45) ∧ (no_two_birch_trees_adjacent_probability = m / n)) := 
sorry

end birch_trees_probability_l2305_230523


namespace max_value_of_expression_l2305_230556

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l2305_230556


namespace mother_age_twice_xiaoming_in_18_years_l2305_230574

-- Definitions based on conditions
def xiaoming_age_now : ℕ := 6
def mother_age_now : ℕ := 30

theorem mother_age_twice_xiaoming_in_18_years : 
    ∀ (n : ℕ), xiaoming_age_now + n = 24 → mother_age_now + n = 2 * (xiaoming_age_now + n) → n = 18 :=
by
  intro n hn hm
  sorry

end mother_age_twice_xiaoming_in_18_years_l2305_230574


namespace ratio_of_areas_of_similar_triangles_l2305_230565

theorem ratio_of_areas_of_similar_triangles (a b a1 b1 S S1 : ℝ) (α k : ℝ) :
  S = (1/2) * a * b * (Real.sin α) →
  S1 = (1/2) * a1 * b1 * (Real.sin α) →
  a1 = k * a →
  b1 = k * b →
  S1 / S = k^2 := by
  intros h1 h2 h3 h4
  sorry

end ratio_of_areas_of_similar_triangles_l2305_230565


namespace fraction_evaluation_l2305_230589

theorem fraction_evaluation :
  let p := 8579
  let q := 6960
  p.gcd q = 1 ∧ (32 / 30 - 30 / 32 + 32 / 29) = p / q :=
by
  sorry

end fraction_evaluation_l2305_230589


namespace parallel_line_plane_l2305_230561

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry

-- Predicate for parallel lines
noncomputable def is_parallel_line (a b : line) : Prop := sorry

-- Predicate for parallel line and plane
noncomputable def is_parallel_plane (a : line) (α : plane) : Prop := sorry

-- Predicate for line contained within the plane
noncomputable def contained_in_plane (b : line) (α : plane) : Prop := sorry

theorem parallel_line_plane
  (a b : line) (α : plane)
  (h1 : is_parallel_line a b)
  (h2 : ¬ contained_in_plane a α)
  (h3 : contained_in_plane b α) :
  is_parallel_plane a α :=
sorry

end parallel_line_plane_l2305_230561


namespace perimeter_of_quadrilateral_l2305_230541

theorem perimeter_of_quadrilateral 
  (WXYZ_area : ℝ)
  (h_area : WXYZ_area = 2500)
  (WQ XQ YQ ZQ : ℝ)
  (h_WQ : WQ = 30)
  (h_XQ : XQ = 40)
  (h_YQ : YQ = 35)
  (h_ZQ : ZQ = 50) :
  ∃ (P : ℝ), P = 155 + 10 * Real.sqrt 34 + 5 * Real.sqrt 113 :=
by
  sorry

end perimeter_of_quadrilateral_l2305_230541


namespace solve_quadratic_l2305_230569

theorem solve_quadratic (y : ℝ) :
  y^2 - 3 * y - 10 = -(y + 2) * (y + 6) ↔ (y = -1/2 ∨ y = -2) :=
by
  sorry

end solve_quadratic_l2305_230569


namespace minimum_days_to_owe_double_l2305_230559

/-- Kim borrows $100$ dollars from Sam with a simple interest rate of $10\%$ per day.
    There's a one-time borrowing fee of $10$ dollars that is added to the debt immediately.
    We need to prove that the least integer number of days after which Kim will owe 
    Sam at least twice as much as she borrowed is 9 days.
-/
theorem minimum_days_to_owe_double :
  ∀ (x : ℕ), 100 + 10 + 10 * x ≥ 200 → x ≥ 9 :=
by
  intros x h
  sorry

end minimum_days_to_owe_double_l2305_230559


namespace iodine_initial_amount_l2305_230532

theorem iodine_initial_amount (half_life : ℕ) (days_elapsed : ℕ) (final_amount : ℕ) (initial_amount : ℕ) :
  half_life = 8 → days_elapsed = 24 → final_amount = 2 → initial_amount = final_amount * 2 ^ (days_elapsed / half_life) → initial_amount = 16 :=
by
  intros h_half_life h_days_elapsed h_final_amount h_initial_exp
  rw [h_half_life, h_days_elapsed, h_final_amount] at h_initial_exp
  norm_num at h_initial_exp
  exact h_initial_exp

end iodine_initial_amount_l2305_230532


namespace squad_sizes_l2305_230557

-- Definitions for conditions
def total_students (x y : ℕ) : Prop := x + y = 146
def equal_after_transfer (x y : ℕ) : Prop := x - 11 = y + 11

-- Theorem to prove the number of students in first and second-year squads
theorem squad_sizes (x y : ℕ) (h1 : total_students x y) (h2 : equal_after_transfer x y) : 
  x = 84 ∧ y = 62 :=
by
  sorry

end squad_sizes_l2305_230557


namespace length_BE_l2305_230585

-- Definitions and Conditions
def is_square (ABCD : Type) (side_length : ℝ) : Prop :=
  side_length = 2

def triangle_area (base : ℝ) (height : ℝ) : ℝ :=
  0.5 * base * height

def rectangle_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

-- Problem statement in Lean
theorem length_BE 
(ABCD : Type) (side_length : ℝ) 
(JKHG : Type) (BC : ℝ) (x : ℝ) 
(E : Type) (E_on_BC : E) 
(area_fact : rectangle_area BC x = 2 * triangle_area x BC) 
(h1 : is_square ABCD side_length) 
(h2 : BC = 2) : 
x = 1 :=
by {
  sorry
}

end length_BE_l2305_230585


namespace smallest_integral_k_l2305_230584

theorem smallest_integral_k (k : ℤ) :
  (297 - 108 * k < 0) ↔ (k ≥ 3) :=
sorry

end smallest_integral_k_l2305_230584


namespace total_produce_of_mangoes_is_400_l2305_230547

variable (A M O : ℕ)  -- Defines variables for total produce of apples, mangoes, and oranges respectively
variable (P : ℕ := 50)  -- Price per kg
variable (R : ℕ := 90000)  -- Total revenue

-- Definition of conditions
def apples_total_produce := 2 * M
def oranges_total_produce := M + 200
def total_weight_of_fruits := apples_total_produce + M + oranges_total_produce

-- Statement to prove
theorem total_produce_of_mangoes_is_400 :
  (total_weight_of_fruits = R / P) → (M = 400) :=
by
  sorry

end total_produce_of_mangoes_is_400_l2305_230547


namespace maximum_value_of_expression_l2305_230553

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression (x y z : ℝ ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 3) :
  problem_statement x y z ≤ 81 / 4 :=
sorry

end maximum_value_of_expression_l2305_230553


namespace value_of_expression_l2305_230597

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6 * b = 9 :=
  sorry

end value_of_expression_l2305_230597


namespace length_cut_XY_l2305_230549

theorem length_cut_XY (a x : ℝ) (h1 : 4 * a = 100) (h2 : a + a + 2 * x = 56) : x = 3 :=
by { sorry }

end length_cut_XY_l2305_230549


namespace children_on_playground_l2305_230519

theorem children_on_playground (boys_soccer girls_soccer boys_swings girls_swings boys_snacks girls_snacks : ℕ)
(h1 : boys_soccer = 27) (h2 : girls_soccer = 35)
(h3 : boys_swings = 15) (h4 : girls_swings = 20)
(h5 : boys_snacks = 10) (h6 : girls_snacks = 5) :
boys_soccer + girls_soccer + boys_swings + girls_swings + boys_snacks + girls_snacks = 112 := by
  sorry

end children_on_playground_l2305_230519


namespace power_function_nature_l2305_230544

def f (x : ℝ) : ℝ := x ^ (1/2)

theorem power_function_nature:
  (f 3 = Real.sqrt 3) ∧
  (¬ (∀ x, f (-x) = f x)) ∧
  (¬ (∀ x, f (-x) = -f x)) ∧
  (∀ x, 0 < x → 0 < f x) := 
by
  sorry

end power_function_nature_l2305_230544


namespace parts_per_hour_equality_l2305_230538

variable {x : ℝ}

theorem parts_per_hour_equality (h1 : x - 4 > 0) :
  (100 / x) = (80 / (x - 4)) :=
sorry

end parts_per_hour_equality_l2305_230538


namespace marble_weight_l2305_230540

-- Define the conditions
def condition1 (m k : ℝ) : Prop := 9 * m = 5 * k
def condition2 (k : ℝ) : Prop := 4 * k = 120

-- Define the main goal, i.e., proving m = 50/3 given the conditions
theorem marble_weight (m k : ℝ) 
  (h1 : condition1 m k) 
  (h2 : condition2 k) : 
  m = 50 / 3 := by 
  sorry

end marble_weight_l2305_230540


namespace range_of_m_l2305_230573

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (hxy : (1/x) + (4/y) = 1) :
  (x + y > m^2 + 8 * m) → (-9 < m ∧ m < 1) :=
by 
  sorry

end range_of_m_l2305_230573


namespace rate_of_increase_twice_l2305_230599

theorem rate_of_increase_twice {x : ℝ} (h : (1 + x)^2 = 2) : x = (Real.sqrt 2) - 1 :=
sorry

end rate_of_increase_twice_l2305_230599


namespace range_of_a_if_p_is_false_l2305_230508

theorem range_of_a_if_p_is_false :
  (∀ x : ℝ, x^2 + a * x + a ≥ 0) → (0 ≤ a ∧ a ≤ 4) := 
sorry

end range_of_a_if_p_is_false_l2305_230508


namespace logically_follows_l2305_230537

-- Define the predicates P and Q
variables {Student : Type} {P Q : Student → Prop}

-- The given condition
axiom Turner_statement : ∀ (x : Student), P x → Q x

-- The statement that necessarily follows
theorem logically_follows : (∀ (x : Student), ¬ Q x → ¬ P x) :=
sorry

end logically_follows_l2305_230537


namespace term_2_6_position_l2305_230543

theorem term_2_6_position : 
  ∃ (seq : ℕ → ℚ), 
    (seq 23 = 2 / 6) ∧ 
    (∀ n, ∃ k, (n = (k * (k + 1)) / 2 ∧ k > 0 ∧ k <= n)) :=
by sorry

end term_2_6_position_l2305_230543


namespace fabian_total_cost_l2305_230567

def mouse_cost : ℕ := 20

def keyboard_cost : ℕ := 2 * mouse_cost

def headphones_cost : ℕ := mouse_cost + 15

def usb_hub_cost : ℕ := 36 - mouse_cost

def total_cost : ℕ := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost

theorem fabian_total_cost : total_cost = 111 := 
by 
  unfold total_cost mouse_cost keyboard_cost headphones_cost usb_hub_cost
  sorry

end fabian_total_cost_l2305_230567


namespace simplify_expression_l2305_230570

theorem simplify_expression (x : ℝ) : 7 * x + 15 - 3 * x + 2 = 4 * x + 17 := 
by sorry

end simplify_expression_l2305_230570


namespace multiple_of_3804_l2305_230596

theorem multiple_of_3804 (n : ℕ) (hn : 0 < n) : 
  ∃ k : ℕ, (n^3 - n) * (5^(8*n+4) + 3^(4*n+2)) = k * 3804 :=
by
  sorry

end multiple_of_3804_l2305_230596


namespace seq_15_l2305_230560

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else if n = 2 then 2 else 2 * (n - 1) + 1 -- form inferred from solution

theorem seq_15 : seq 15 = 29 := by
  sorry

end seq_15_l2305_230560


namespace pyramid_rhombus_side_length_l2305_230506

theorem pyramid_rhombus_side_length
  (α β S: ℝ) (hα : 0 < α) (hβ : 0 < β) (hS : 0 < S) :
  ∃ a : ℝ, a = 2 * Real.sqrt (2 * S * Real.cos β / Real.sin α) :=
by
  sorry

end pyramid_rhombus_side_length_l2305_230506


namespace jake_spent_more_l2305_230581

def cost_of_balloons (helium_count : ℕ) (foil_count : ℕ) (helium_price : ℝ) (foil_price : ℝ) : ℝ :=
  helium_count * helium_price + foil_count * foil_price

theorem jake_spent_more 
  (allan_helium : ℕ) (allan_foil : ℕ) (jake_helium : ℕ) (jake_foil : ℕ)
  (helium_price : ℝ) (foil_price : ℝ)
  (h_allan_helium : allan_helium = 2) (h_allan_foil : allan_foil = 3) 
  (h_jake_helium : jake_helium = 4) (h_jake_foil : jake_foil = 2)
  (h_helium_price : helium_price = 1.5) (h_foil_price : foil_price = 2.5) :
  cost_of_balloons jake_helium jake_foil helium_price foil_price - 
  cost_of_balloons allan_helium allan_foil helium_price foil_price = 0.5 := 
by
  sorry

end jake_spent_more_l2305_230581


namespace speed_of_stream_l2305_230571

-- Conditions
variables (v : ℝ) -- speed of the stream in kmph
variables (boat_speed_still_water : ℝ := 10) -- man's speed in still water in kmph
variables (distance : ℝ := 90) -- distance traveled down the stream in km
variables (time : ℝ := 5) -- time taken to travel the distance down the stream in hours

-- Proof statement
theorem speed_of_stream : v = 8 :=
  by
    -- effective speed down the stream = boat_speed_still_water + v
    -- given that distance = speed * time
    -- 90 = (10 + v) * 5
    -- solving for v
    sorry

end speed_of_stream_l2305_230571


namespace cos_identity_l2305_230516

theorem cos_identity (θ : ℝ) (h : Real.cos (π / 6 + θ) = (Real.sqrt 3) / 3) : 
  Real.cos (5 * π / 6 - θ) = - (Real.sqrt 3 / 3) :=
by
  sorry

end cos_identity_l2305_230516


namespace additional_people_proof_l2305_230563

variable (initialPeople additionalPeople mowingHours trimmingRate totalNewPeople totalMowingPeople requiredPersonHours totalPersonHours: ℕ)

noncomputable def mowingLawn (initialPeople mowingHours : ℕ) : ℕ :=
  initialPeople * mowingHours

noncomputable def mowingRate (requiredPersonHours : ℕ) (mowingHours : ℕ) : ℕ :=
  (requiredPersonHours / mowingHours)

noncomputable def trimmingEdges (totalMowingPeople trimmingRate : ℕ) : ℕ :=
  (totalMowingPeople / trimmingRate)

noncomputable def totalPeople (mowingPeople trimmingPeople : ℕ) : ℕ :=
  (mowingPeople + trimmingPeople)

noncomputable def additionalPeopleNeeded (totalPeople initialPeople : ℕ) : ℕ :=
  (totalPeople - initialPeople)

theorem additional_people_proof :
  initialPeople = 8 →
  mowingHours = 3 →
  totalPersonHours = mowingLawn initialPeople mowingHours →
  totalMowingPeople = mowingRate totalPersonHours 2 →
  trimmingRate = 3 →
  requiredPersonHours = totalPersonHours →
  totalNewPeople = totalPeople totalMowingPeople (trimmingEdges totalMowingPeople trimmingRate) →
  additionalPeople = additionalPeopleNeeded totalNewPeople initialPeople →
  additionalPeople = 8 :=
by
  sorry

end additional_people_proof_l2305_230563


namespace product_of_roots_l2305_230582

noncomputable def quadratic_equation (x : ℝ) : Prop :=
  (x + 4) * (x - 5) = 22

theorem product_of_roots :
  ∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → (x1 * x2 = -42) := 
by
  sorry

end product_of_roots_l2305_230582


namespace prime_iff_good_fractions_l2305_230598

def isGoodFraction (n : ℕ) (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ (a + b = n)

def canBeExpressedUsingGoodFractions (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (expressedFraction : ℕ → ℕ → Prop), expressedFraction a b ∧
  ∀ x y, expressedFraction x y → isGoodFraction n x y

theorem prime_iff_good_fractions {n : ℕ} (hn : n > 1) :
  Prime n ↔
    ∀ a b : ℕ, b < n → (a > 0 ∧ b > 0) → canBeExpressedUsingGoodFractions n a b :=
sorry

end prime_iff_good_fractions_l2305_230598


namespace total_books_l2305_230536

theorem total_books (joan_books tom_books sarah_books alex_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : sarah_books = 25)
  (h4 : alex_books = 45) : 
  joan_books + tom_books + sarah_books + alex_books = 118 := 
by 
  sorry

end total_books_l2305_230536


namespace ab_value_l2305_230583

theorem ab_value (a b : ℚ) 
  (h1 : (a + b) ^ 2 + |b + 5| = b + 5) 
  (h2 : 2 * a - b + 1 = 0) : 
  a * b = -1 / 9 :=
by
  sorry

end ab_value_l2305_230583


namespace find_s_l2305_230526

theorem find_s (n : ℤ) (hn : n ≠ 0) (s : ℝ)
  (hs : s = (20 / (2^(2*n+4) + 2^(2*n+2)))^(1 / n)) :
  s = 1 / 4 :=
by
  sorry

end find_s_l2305_230526


namespace compute_3X4_l2305_230504

def operation_X (a b : ℤ) : ℤ := b + 12 * a - a^2

theorem compute_3X4 : operation_X 3 4 = 31 := 
by
  sorry

end compute_3X4_l2305_230504


namespace problem_126_times_3_pow_6_l2305_230568

theorem problem_126_times_3_pow_6 (p : ℝ) (h : 126 * 3^8 = p) : 
  126 * 3^6 = (1 / 9) * p := 
by {
  -- Placeholder for the proof
  sorry
}

end problem_126_times_3_pow_6_l2305_230568


namespace radius_of_circle_l2305_230576

theorem radius_of_circle:
  (∃ (r: ℝ), 
    (∀ (x: ℝ), (x^2 + r - x) = 0 → 1 - 4 * r = 0)
  ) → r = 1 / 4 := 
sorry

end radius_of_circle_l2305_230576


namespace plane_equation_l2305_230531

theorem plane_equation
  (A B C D : ℤ)
  (hA : A > 0)
  (h_gcd : Int.gcd A B = 1 ∧ Int.gcd A C = 1 ∧ Int.gcd A D = 1)
  (h_point : (A * 4 + B * (-4) + C * 5 + D = 0)) :
  A = 4 ∧ B = -4 ∧ C = 5 ∧ D = -57 :=
  sorry

end plane_equation_l2305_230531


namespace two_digit_number_solution_l2305_230505

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l2305_230505


namespace jackie_more_apples_oranges_l2305_230517

-- Definitions of initial conditions
def adams_apples : ℕ := 25
def adams_oranges : ℕ := 34
def jackies_apples : ℕ := 43
def jackies_oranges : ℕ := 29

-- The proof statement
theorem jackie_more_apples_oranges :
  (jackies_apples - adams_apples) + (jackies_oranges - adams_oranges) = 13 :=
by
  sorry

end jackie_more_apples_oranges_l2305_230517


namespace jeans_price_increase_l2305_230550

theorem jeans_price_increase
  (C R P : ℝ)
  (h1 : P = 1.15 * R)
  (h2 : P = 1.6100000000000001 * C) :
  R = 1.4 * C :=
by
  sorry

end jeans_price_increase_l2305_230550


namespace smallest_solution_l2305_230566

theorem smallest_solution (x : ℝ) : 
  (∃ x, (3 * x / (x - 3)) + ((3 * x^2 - 27) / x) = 15 ∧ ∀ y, (3 * y / (y - 3)) + ((3 * y^2 - 27) / y) = 15 → y ≥ x) → 
  x = -1 := 
by
  sorry

end smallest_solution_l2305_230566


namespace not_all_positive_l2305_230507

theorem not_all_positive (a b c : ℝ) (h1 : a + b + c = 4) (h2 : a^2 + b^2 + c^2 = 12) (h3 : a * b * c = 1) : a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0 :=
sorry

end not_all_positive_l2305_230507


namespace arithmetic_sequence_sum_l2305_230586

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d
noncomputable def S_n (a1 d : ℕ) (n : ℕ) : ℕ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum (a1 d : ℕ) 
  (h1 : a1 + d = 6) 
  (h2 : (a1 + 2 * d)^2 = a1 * (a1 + 6 * d)) 
  (h3 : d ≠ 0) : 
  S_n a1 d 8 = 88 := 
by 
  sorry

end arithmetic_sequence_sum_l2305_230586


namespace determine_jug_capacity_l2305_230500

variable (jug_capacity : Nat)
variable (small_jug : Nat)

theorem determine_jug_capacity (h1 : jug_capacity = 5) (h2 : small_jug = 3 ∨ small_jug = 4):
  (∃ overflow_remains : Nat, 
    (overflow_remains = jug_capacity ∧ small_jug = 4) ∨ 
    (¬(overflow_remains = jug_capacity) ∧ small_jug = 3)) :=
by
  sorry

end determine_jug_capacity_l2305_230500


namespace hired_waiters_l2305_230521

theorem hired_waiters (W H : Nat) (hcooks : Nat := 9) 
                      (initial_ratio : 3 * W = 11 * hcooks)
                      (new_ratio : 9 = 5 * (W + H)) 
                      (original_waiters : W = 33) 
                      : H = 12 :=
by
  sorry

end hired_waiters_l2305_230521


namespace education_fund_growth_l2305_230534

theorem education_fund_growth (x : ℝ) :
  2500 * (1 + x)^2 = 3600 :=
sorry

end education_fund_growth_l2305_230534


namespace correlation_coefficients_l2305_230510

-- Definition of the variables and constants
def relative_risks_starting_age : List (ℕ × ℝ) := [(16, 15.10), (18, 12.81), (20, 9.72), (22, 3.21)]
def relative_risks_cigarettes_per_day : List (ℕ × ℝ) := [(10, 7.5), (20, 9.5), (30, 16.6)]

def r1 : ℝ := -- The correlation coefficient between starting age and relative risk
  sorry

def r2 : ℝ := -- The correlation coefficient between number of cigarettes per day and relative risk
  sorry

theorem correlation_coefficients :
  r1 < 0 ∧ 0 < r2 :=
by {
  -- Proof is skipped with sorry
  sorry
}

end correlation_coefficients_l2305_230510


namespace solve_equation_l2305_230515

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l2305_230515


namespace max_area_of_triangle_l2305_230594

noncomputable def max_area_triangle (a A : ℝ) : ℝ :=
  let bcsinA := sorry
  1 / 2 * bcsinA

theorem max_area_of_triangle (a A : ℝ) (hab : a = 4) (hAa : A = Real.pi / 3) :
  max_area_triangle a A = 4 * Real.sqrt 3 :=
by
  sorry

end max_area_of_triangle_l2305_230594


namespace increasing_interval_l2305_230503

def my_function (x : ℝ) : ℝ := -(x - 3) * |x|

theorem increasing_interval : ∀ x y : ℝ, 0 ≤ x → x ≤ y → my_function x ≤ my_function y :=
by
  sorry

end increasing_interval_l2305_230503


namespace positive_difference_of_b_values_l2305_230525

noncomputable def g (n : ℤ) : ℤ :=
if n ≤ 0 then n^2 + 3 * n + 2 else 3 * n - 15

theorem positive_difference_of_b_values : 
  abs (-5 - 9) = 14 :=
by {
  sorry
}

end positive_difference_of_b_values_l2305_230525


namespace blue_pill_cost_correct_l2305_230520

-- Defining the conditions
def num_days : Nat := 21
def total_cost : Nat := 672
def red_pill_cost (blue_pill_cost : Nat) : Nat := blue_pill_cost - 2
def daily_cost (blue_pill_cost : Nat) : Nat := blue_pill_cost + red_pill_cost blue_pill_cost

-- The statement to prove
theorem blue_pill_cost_correct : ∃ (y : Nat), daily_cost y * num_days = total_cost ∧ y = 17 :=
by
  sorry

end blue_pill_cost_correct_l2305_230520


namespace son_daughter_eggs_per_morning_l2305_230579

-- Define the given conditions in Lean 4
def trays_per_week : Nat := 2
def eggs_per_tray : Nat := 24
def eggs_per_night_rhea_husband : Nat := 4
def nights_per_week : Nat := 7
def uneaten_eggs_per_week : Nat := 6

-- Define the total eggs bought per week
def total_eggs_per_week : Nat := trays_per_week * eggs_per_tray

-- Define the eggs eaten per week by Rhea and her husband
def eggs_eaten_per_week_rhea_husband : Nat := eggs_per_night_rhea_husband * nights_per_week

-- Prove the number of eggs eaten by son and daughter every morning
theorem son_daughter_eggs_per_morning :
  (total_eggs_per_week - eggs_eaten_per_week_rhea_husband - uneaten_eggs_per_week) = 14 :=
sorry

end son_daughter_eggs_per_morning_l2305_230579


namespace statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_l2305_230527

theorem statement_A_correct :
  (∃ x0 : ℝ, x0^2 + 2 * x0 + 2 < 0) ↔ (¬ ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0) :=
sorry

theorem statement_B_incorrect :
  ¬ (∀ x y : ℝ, x > y → |x| > |y|) :=
sorry

theorem statement_C_incorrect :
  ¬ ∀ x : ℤ, x^2 > 0 :=
sorry

theorem statement_D_correct :
  (∀ m : ℝ, (∃ x1 x2 : ℝ, x1 + x2 = 2 ∧ x1 * x2 = m ∧ x1 * x2 > 0) ↔ m < 0) :=
sorry

end statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_l2305_230527


namespace speed_ratio_l2305_230511

variable (v_A v_B : ℝ)

def equidistant_3min : Prop := 3 * v_A = abs (-800 + 3 * v_B)
def equidistant_8min : Prop := 8 * v_A = abs (-800 + 8 * v_B)
def speed_ratio_correct : Prop := v_A / v_B = 1 / 2

theorem speed_ratio (h1 : equidistant_3min v_A v_B) (h2 : equidistant_8min v_A v_B) : speed_ratio_correct v_A v_B :=
by
  sorry

end speed_ratio_l2305_230511


namespace x1_x2_in_M_l2305_230578

-- Definitions of the set M and the condition x ∈ M
def M : Set ℕ := { x | ∃ a b : ℤ, x = a^2 + b^2 }

-- Statement of the problem
theorem x1_x2_in_M (x1 x2 : ℕ) (h1 : x1 ∈ M) (h2 : x2 ∈ M) : (x1 * x2) ∈ M :=
sorry

end x1_x2_in_M_l2305_230578


namespace passing_marks_l2305_230555

theorem passing_marks
  (T P : ℝ)
  (h1 : 0.20 * T = P - 40)
  (h2 : 0.30 * T = P + 20) :
  P = 160 :=
by
  sorry

end passing_marks_l2305_230555


namespace perfect_squares_unique_l2305_230580

theorem perfect_squares_unique (n : ℕ) (h1 : ∃ k : ℕ, 20 * n = k^2) (h2 : ∃ p : ℕ, 5 * n + 275 = p^2) :
  n = 125 :=
by
  sorry

end perfect_squares_unique_l2305_230580


namespace q_domain_range_l2305_230524

open Set

-- Given the function h with the specified domain and range
variable (h : ℝ → ℝ) (h_domain : ∀ x, -1 ≤ x ∧ x ≤ 3 → h x ∈ Icc 0 2)

def q (x : ℝ) : ℝ := 2 - h (x - 2)

theorem q_domain_range :
  (∀ x, 1 ≤ x ∧ x ≤ 5 → (q h x) ∈ Icc 0 2) ∧
  (∀ y, q h y ∈ Icc 0 2 ↔ y ∈ Icc 1 5) :=
by
  sorry

end q_domain_range_l2305_230524


namespace find_term_in_sequence_l2305_230548

theorem find_term_in_sequence (n : ℕ) (k : ℕ) (term_2020: ℚ) : 
  (3^7 = 2187) → 
  (2020 : ℕ) / (2187 : ℕ) = term_2020 → 
  (term_2020 = 2020 / 2187) →
  (∃ (k : ℕ), k = 2020 ∧ (2 ≤ k ∧ k < 2187 ∧ k % 3 ≠ 0)) → 
  (2020 / 2187 = (1347 / 2187 : ℚ)) :=
by {
  sorry
}

end find_term_in_sequence_l2305_230548


namespace find_term_number_l2305_230546

noncomputable def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem find_term_number
  (a₁ : ℤ)
  (d : ℤ)
  (n : ℕ)
  (h₀ : a₁ = 1)
  (h₁ : d = 3)
  (h₂ : arithmetic_sequence a₁ d n = 2011) :
  n = 671 :=
  sorry

end find_term_number_l2305_230546


namespace max_pencils_l2305_230577

theorem max_pencils 
  (p : ℕ → ℝ)
  (h_price1 : ∀ n : ℕ, n ≤ 10 → p n = 0.75 * n)
  (h_price2 : ∀ n : ℕ, n > 10 → p n = 0.75 * 10 + 0.65 * (n - 10))
  (budget : ℝ) (h_budget : budget = 10) :
  ∃ n : ℕ, p n ≤ budget ∧ (∀ m : ℕ, p m ≤ budget → m ≤ 13) :=
by {
  sorry
}

end max_pencils_l2305_230577


namespace solve_abs_quadratic_l2305_230572

theorem solve_abs_quadratic :
  ∀ x : ℝ, abs (x^2 - 4 * x + 4) = 3 - x ↔ (x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) :=
by
  sorry

end solve_abs_quadratic_l2305_230572


namespace egg_weight_probability_l2305_230595

theorem egg_weight_probability : 
  let P_lt_30 := 0.3
  let P_30_40 := 0.5
  P_lt_30 + P_30_40 ≤ 1 → (1 - (P_lt_30 + P_30_40) = 0.2) := by
  intro h
  sorry

end egg_weight_probability_l2305_230595


namespace infinite_series_sum_zero_l2305_230551

theorem infinite_series_sum_zero : ∑' n : ℕ, (3 * n + 4) / ((n + 1) * (n + 2) * (n + 3)) = 0 :=
by
  sorry

end infinite_series_sum_zero_l2305_230551


namespace factorial_div_l2305_230590

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_div : (factorial 4) / (factorial (4 - 3)) = 24 := by
  sorry

end factorial_div_l2305_230590


namespace no_triangle_with_heights_1_2_3_l2305_230587

open Real

theorem no_triangle_with_heights_1_2_3 :
  ¬(∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
     ∃ (k : ℝ), k > 0 ∧ 
       a * k = 1 ∧ b * (k / 2) = 2 ∧ c * (k / 3) = 3 ∧
       (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by 
  sorry

end no_triangle_with_heights_1_2_3_l2305_230587


namespace ratio_of_girls_to_boys_l2305_230501

theorem ratio_of_girls_to_boys (total_students girls boys : ℕ) 
  (h1 : total_students = 26) 
  (h2 : girls = boys + 6) 
  (h3 : girls + boys = total_students) : 
  (girls : ℚ) / boys = 8 / 5 :=
by
  sorry

end ratio_of_girls_to_boys_l2305_230501


namespace circle_area_ratio_l2305_230513

theorem circle_area_ratio (R_C R_D : ℝ)
  (h₁ : (60 / 360 * 2 * Real.pi * R_C) = (40 / 360 * 2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 9 / 4 :=
by 
  sorry

end circle_area_ratio_l2305_230513


namespace intersection_product_l2305_230509

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 - 4 * x + y^2 - 6 * y + 9 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 - 8 * x + y^2 - 6 * y + 25 = 0

-- Define the theorem to prove the product of the coordinates of the intersection points
theorem intersection_product : ∀ x y : ℝ, circle1 x y → circle2 x y → x * y = 12 :=
by
  intro x y h1 h2
  -- Insert proof here
  sorry

end intersection_product_l2305_230509


namespace find_price_of_turban_l2305_230575

-- Define the main variables and conditions
def price_of_turban (T : ℝ) : Prop :=
  ((3 / 4) * 90 + T = 60 + T) → T = 30

-- State the theorem with the given conditions and aim to find T
theorem find_price_of_turban (T : ℝ) (h1 : 90 + T = 120) :  price_of_turban T :=
by
  intros
  sorry


end find_price_of_turban_l2305_230575


namespace sin_A_mul_sin_B_find_c_l2305_230528

-- Definitions for the triangle and the given conditions
variable (A B C : ℝ) -- Angles of the triangle
variable (a b c : ℝ) -- Opposite sides of the triangle

-- Given conditions
axiom h1 : c^2 = 4 * a * b * (Real.sin C)^2

-- The first proof problem statement
theorem sin_A_mul_sin_B (ha : A + B + C = π) (h2 : Real.sin C ≠ 0) :
  Real.sin A * Real.sin B = 1/4 :=
by
  sorry

-- The second proof problem statement with additional given conditions
theorem find_c (ha : A = π / 6) (ha2 : a = 3) (hb2 : b = 3) : 
  c = 3 * Real.sqrt 3 :=
by
  sorry

end sin_A_mul_sin_B_find_c_l2305_230528


namespace xiao_ming_selects_cooking_probability_l2305_230558

theorem xiao_ming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let probability (event: String) := if event ∈ courses then 1 / (courses.length : ℝ) else 0
  probability "cooking" = 1 / 4 :=
by
  sorry

end xiao_ming_selects_cooking_probability_l2305_230558


namespace sarah_bottle_caps_l2305_230542

theorem sarah_bottle_caps (initial_caps : ℕ) (additional_caps : ℕ) (total_caps : ℕ) : initial_caps = 26 → additional_caps = 3 → total_caps = initial_caps + additional_caps → total_caps = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sarah_bottle_caps_l2305_230542


namespace restaurant_total_cost_l2305_230593

theorem restaurant_total_cost :
  let vegetarian_cost := 5
  let chicken_cost := 7
  let steak_cost := 10
  let kids_cost := 3
  let tax_rate := 0.10
  let tip_rate := 0.15
  let num_vegetarians := 3
  let num_chicken_lovers := 4
  let num_steak_enthusiasts := 2
  let num_kids_hot_dog := 3
  let subtotal := (num_vegetarians * vegetarian_cost) + (num_chicken_lovers * chicken_cost) + (num_steak_enthusiasts * steak_cost) + (num_kids_hot_dog * kids_cost)
  let tax := subtotal * tax_rate
  let tip := subtotal * tip_rate
  let total_cost := subtotal + tax + tip
  total_cost = 90 :=
by sorry

end restaurant_total_cost_l2305_230593


namespace power_inequality_l2305_230529

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) := 
by 
  sorry

end power_inequality_l2305_230529


namespace solve_trig_eq_l2305_230562

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  (x = (π / 12) + 2 * k * π ∨
   x = (7 * π / 12) + 2 * k * π ∨
   x = (7 * π / 6) + 2 * k * π ∨
   x = -(5 * π / 6) + 2 * k * π) →
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 :=
sorry

end solve_trig_eq_l2305_230562


namespace arithmetic_sum_l2305_230502

variable {a : ℕ → ℝ}

def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sum :
  is_arithmetic_seq a →
  a 5 + a 6 + a 7 = 15 →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by
  intros
  sorry

end arithmetic_sum_l2305_230502


namespace find_annual_interest_rate_l2305_230533

noncomputable def compound_interest_problem : Prop :=
  ∃ (r : ℝ),
    let P := 8000
    let CI := 3109
    let t := 2.3333
    let A := 11109
    let n := 1
    A = P * (1 + r/n)^(n*t) ∧ r = 0.1505

theorem find_annual_interest_rate : compound_interest_problem :=
by sorry

end find_annual_interest_rate_l2305_230533
