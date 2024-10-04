import Mathlib

namespace part1_part2_i_part2_ii_l127_127718

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x * Real.log x - 1

theorem part1 (a : ℝ) (x : ℝ) : f x a + x^2 * f (1 / x) a = 0 :=
by sorry

theorem part2_i (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0) : 2 < a :=
by sorry

theorem part2_ii (a : ℝ) (x1 x2 x3 : ℝ) (h : x1 < x2 ∧ x2 < x3 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0) : x1 + x3 > 2 * a - 2 :=
by sorry

end part1_part2_i_part2_ii_l127_127718


namespace solution_set_of_cx_sq_minus_bx_plus_a_l127_127581

theorem solution_set_of_cx_sq_minus_bx_plus_a (a b c : ℝ) (h1 : a < 0)
(h2 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ 2 < x ∧ x < 3) :
  ∀ x : ℝ, cx^2 - bx + a > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by
  sorry

end solution_set_of_cx_sq_minus_bx_plus_a_l127_127581


namespace ashley_champagne_bottles_l127_127211

theorem ashley_champagne_bottles (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) 
  (h1 : guests = 120) (h2 : glasses_per_guest = 2) (h3 : servings_per_bottle = 6) : 
  (guests * glasses_per_guest) / servings_per_bottle = 40 :=
by
  -- The proof will go here
  sorry

end ashley_champagne_bottles_l127_127211


namespace simplify_and_evaluate_expression_l127_127767

theorem simplify_and_evaluate_expression (a b : ℝ) (h₁ : a = 2 + Real.sqrt 3) (h₂ : b = 2 - Real.sqrt 3) :
  (a^2 - b^2) / a / (a - (2 * a * b - b^2) / a) = 2 * Real.sqrt 3 / 3 :=
by
  -- Proof to be provided
  sorry

end simplify_and_evaluate_expression_l127_127767


namespace binom_20_10_eq_184756_l127_127544

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end binom_20_10_eq_184756_l127_127544


namespace smallest_four_digit_divisible_by_primes_l127_127085

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end smallest_four_digit_divisible_by_primes_l127_127085


namespace tax_deduction_cents_l127_127811

def bob_hourly_wage : ℝ := 25
def tax_rate : ℝ := 0.025

theorem tax_deduction_cents :
  (bob_hourly_wage * 100 * tax_rate) = 62.5 :=
by
  -- This is the statement that needs to be proven.
  sorry

end tax_deduction_cents_l127_127811


namespace minimum_value_of_a_l127_127547

-- Define the given condition
axiom a_pos : ℝ → Prop
axiom positive : ∀ (x : ℝ), 0 < x

-- Definition of the equation
def equation (x y a : ℝ) : Prop :=
  (2 * x - y / Real.exp 1) * Real.log (y / x) = x / (a * Real.exp 1)

-- The mathematical statement we need to prove
theorem minimum_value_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) (h_eq : equation x y a) : 
  a ≥ 1 / Real.exp 1 :=
sorry

end minimum_value_of_a_l127_127547


namespace total_number_of_people_l127_127346

-- Conditions
def number_of_parents : ℕ := 105
def number_of_pupils : ℕ := 698

-- Theorem stating the total number of people is 803 given the conditions
theorem total_number_of_people : 
  number_of_parents + number_of_pupils = 803 :=
by
  sorry

end total_number_of_people_l127_127346


namespace bottles_needed_l127_127206

-- Define specific values provided in conditions
def servings_per_guest : ℕ := 2
def number_of_guests : ℕ := 120
def servings_per_bottle : ℕ := 6

-- Define total servings needed
def total_servings : ℕ := servings_per_guest * number_of_guests

-- Define the number of bottles needed (as a proof statement)
theorem bottles_needed : total_servings / servings_per_bottle = 40 := by
  /-
    The proof will go here. For now we place a sorry to mark the place where
    a proof would be required. The statement should check the equivalence of 
    number of bottles needed being 40 given the total servings divided by 
    servings per bottle.
  -/
  sorry

end bottles_needed_l127_127206


namespace price_of_A_correct_l127_127494

noncomputable def A_price : ℝ := 25

theorem price_of_A_correct (H1 : 6000 / A_price - 4800 / (1.2 * A_price) = 80) 
                           (H2 : ∀ B_price : ℝ, B_price = 1.2 * A_price) : A_price = 25 := 
by
  sorry

end price_of_A_correct_l127_127494


namespace kim_saplings_left_l127_127597

def sprouted_pits (total_pits num_sprouted_pits: ℕ) (percent_sprouted: ℝ) : Prop :=
  percent_sprouted * total_pits = num_sprouted_pits

def sold_saplings (total_saplings saplings_sold saplings_left: ℕ) : Prop :=
  total_saplings - saplings_sold = saplings_left

theorem kim_saplings_left
  (total_pits : ℕ) (num_sprouted_pits : ℕ) (percent_sprouted : ℝ)
  (saplings_sold : ℕ) (saplings_left : ℕ) :
  total_pits = 80 →
  percent_sprouted = 0.25 →
  saplings_sold = 6 →
  sprouted_pits total_pits num_sprouted_pits percent_sprouted →
  sold_saplings num_sprouted_pits saplings_sold saplings_left →
  saplings_left = 14 :=
by
  intros
  sorry

end kim_saplings_left_l127_127597


namespace find_base_b_l127_127133

theorem find_base_b : ∃ b : ℕ, b > 4 ∧ (b + 2)^2 = b^2 + 4 * b + 4 ∧ b = 5 := 
sorry

end find_base_b_l127_127133


namespace smallest_four_digit_divisible_five_smallest_primes_l127_127079

theorem smallest_four_digit_divisible_five_smallest_primes :
  let lcm_5_min_primes := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 (Nat.lcm 7 11))
  ∀ m : ℕ, (m ≥ 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ m)) → m = 2310 :=
by
  let lcm_5_min_primes := 2310
  intro m h
  have h1 : m = 2310 := sorry
  exact h1

end smallest_four_digit_divisible_five_smallest_primes_l127_127079


namespace determine_function_l127_127692

theorem determine_function (f : ℕ → ℕ)
  (h : ∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 → f (a + b) = f a + f b + f c + f d) :
  ∀ n : ℕ, f n = n^2 * f 1 := 
sorry

end determine_function_l127_127692


namespace find_f_quarter_and_solve_f_neg_x_l127_127551

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2^x else real.log 2 x

-- The Lean theorem statement without providing proof:

theorem find_f_quarter_and_solve_f_neg_x : 
  f (1 / 4) = -2 ∧ ((f (-1)) = (1 / 2) ∧ (f (-(-sqrt 2))) = (1 / 2)) :=
by 
  sorry

end find_f_quarter_and_solve_f_neg_x_l127_127551


namespace number_of_girls_l127_127744

variable (G : ℕ) -- Number of girls in the school
axiom boys_count : G + 807 = 841 -- Given condition

theorem number_of_girls : G = 34 :=
by
  sorry

end number_of_girls_l127_127744


namespace min_f_l127_127845

noncomputable def f (x y z : ℝ) : ℝ :=
  (3 * x^2 - x) / (1 + x^2) + 
  (3 * y^2 - y) / (1 + y^2) + 
  (3 * z^2 - z) / (1 + z^2)

theorem min_f (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  f x y z = 0 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
sorry

end min_f_l127_127845


namespace elena_pens_l127_127070

theorem elena_pens (X Y : ℝ) 
  (h1 : X + Y = 12) 
  (h2 : 4 * X + 2.80 * Y = 40) :
  X = 5 :=
by
  sorry

end elena_pens_l127_127070


namespace average_lifespan_is_28_l127_127658

-- Define the given data
def batteryLifespans : List ℕ := [30, 35, 25, 25, 30, 34, 26, 25, 29, 21]

-- Define a function to calculate the average of a list of natural numbers
def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

-- State the theorem to be proved
theorem average_lifespan_is_28 :
  average batteryLifespans = 28 := by
  sorry

end average_lifespan_is_28_l127_127658


namespace base_any_number_l127_127730

theorem base_any_number (base : ℝ) (x y : ℝ) (h1 : 3^x * base^y = 19683) (h2 : x - y = 9) (h3 : x = 9) : true :=
by
  sorry

end base_any_number_l127_127730


namespace num_circles_rectangle_l127_127601

structure Rectangle (α : Type*) [Field α] :=
  (A B C D : α × α)
  (AB_parallel_CD : B.1 = A.1 ∧ D.1 = C.1)
  (AD_parallel_BC : D.2 = A.2 ∧ C.2 = B.2)

def num_circles_with_diameter_vertices (R : Rectangle ℝ) : ℕ :=
  sorry

theorem num_circles_rectangle (R : Rectangle ℝ) : num_circles_with_diameter_vertices R = 5 :=
  sorry

end num_circles_rectangle_l127_127601


namespace sum_of_squares_of_roots_l127_127833

theorem sum_of_squares_of_roots :
  (∃ r1 r2 : ℝ, (r1 + r2 = 10 ∧ r1 * r2 = 16) ∧ (r1^2 + r2^2 = 68)) :=
by
  sorry

end sum_of_squares_of_roots_l127_127833


namespace g_neither_even_nor_odd_l127_127591

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 2) + 1/3

theorem g_neither_even_nor_odd : ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) :=
by
  -- insert proof here
  sorry

end g_neither_even_nor_odd_l127_127591


namespace remainder_of_xyz_l127_127416

theorem remainder_of_xyz {x y z : ℕ} (hx: x < 9) (hy: y < 9) (hz: z < 9)
  (h1: (x + 3*y + 2*z) % 9 = 0)
  (h2: (2*x + 2*y + z) % 9 = 7)
  (h3: (x + 2*y + 3*z) % 9 = 5) :
  (x * y * z) % 9 = 5 :=
sorry

end remainder_of_xyz_l127_127416


namespace find_sum_of_squares_l127_127949

theorem find_sum_of_squares (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + x + y = 119) (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := 
by
  sorry

end find_sum_of_squares_l127_127949


namespace total_sand_volume_l127_127353

noncomputable def cone_diameter : ℝ := 10
noncomputable def cone_radius : ℝ := cone_diameter / 2
noncomputable def cone_height : ℝ := 0.75 * cone_diameter
noncomputable def cylinder_height : ℝ := 0.5 * cone_diameter
noncomputable def total_volume : ℝ := (1 / 3 * Real.pi * cone_radius^2 * cone_height) + (Real.pi * cone_radius^2 * cylinder_height)

theorem total_sand_volume : total_volume = 187.5 * Real.pi := 
by
  sorry

end total_sand_volume_l127_127353


namespace trig_identity_l127_127636

theorem trig_identity : (Real.cos (15 * Real.pi / 180))^2 - (Real.sin (15 * Real.pi / 180))^2 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l127_127636


namespace max_area_of_rectangle_l127_127860

theorem max_area_of_rectangle (L : ℝ) (hL : L = 16) :
  ∃ (A : ℝ), (∀ (x : ℝ), 0 < x ∧ x < 8 → A = x * (8 - x)) ∧ A = 16 :=
by
  sorry

end max_area_of_rectangle_l127_127860


namespace time_distribution_l127_127043

noncomputable def total_hours_at_work (hours_task1 day : ℕ) (hours_task2 day : ℕ) (work_days : ℕ) (reduce_per_week : ℕ) : ℕ :=
  (hours_task1 + hours_task2) * work_days

theorem time_distribution (h1 : 5 = 5) (h2 : 3 = 3) (days : 5 = 5) (reduction : 5 = 5) :
  total_hours_at_work 5 3 5 5 = 40 :=
by
  sorry

end time_distribution_l127_127043


namespace units_digit_7_pow_75_plus_6_l127_127794

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_75_plus_6 : units_digit (7 ^ 75 + 6) = 9 := 
by
  sorry

end units_digit_7_pow_75_plus_6_l127_127794


namespace find_y_coordinate_l127_127000

-- Define points A, B, C, and D
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-2, 2)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the property that a point P satisfies PA + PD = PB + PC = 10
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PD := Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let PC := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  PA + PD = 10 ∧ PB + PC = 10

-- Lean statement to prove the y-coordinate of P that satisfies the condition
theorem find_y_coordinate :
  ∃ (P : ℝ × ℝ), satisfies_condition P ∧ ∃ (a b c d : ℕ), a = 0 ∧ b = 1 ∧ c = 21 ∧ d = 3 ∧ P.2 = (14 + Real.sqrt 21) / 3 ∧ a + b + c + d = 25 :=
by
  sorry

end find_y_coordinate_l127_127000


namespace dice_probability_l127_127331

open ProbabilityTheory

noncomputable def prob_two_ones_and_one_six : ℝ :=
  let n := 12
  let k := 2
  let p := 1 / 6
  let q := 5 / 6
  let combinations := Nat.choose n k
  let prob_two_ones := combinations * p^k * q^(n - k)
  let prob_at_least_one_six := 1 - q^10
  prob_two_ones * prob_at_least_one_six

theorem dice_probability :
  prob_two_ones_and_one_six = 0.049 :=
sorry

end dice_probability_l127_127331


namespace prove_u_div_p_l127_127273

theorem prove_u_div_p (p r s u : ℚ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) : 
  u / p = 15 / 8 := 
by 
  sorry

end prove_u_div_p_l127_127273


namespace sum_of_elements_in_T_l127_127992

   /-- T is the set of all positive integers that have five digits in base 2 -/
   def T : Set ℕ := {n | (16 ≤ n ∧ n ≤ 31)}

   /-- The sum of all elements in the set T, expressed in base 2, is 111111000_2 -/
   theorem sum_of_elements_in_T :
     (∑ n in T, n) = 0b111111000 :=
   by
     sorry
   
end sum_of_elements_in_T_l127_127992


namespace units_digit_2008_pow_2008_l127_127142

theorem units_digit_2008_pow_2008 : (2008 ^ 2008) % 10 = 6 := 
by
  -- The units digits of powers of 8 repeat in a cycle: 8, 4, 2, 6
  -- 2008 mod 4 = 0 which implies it falls on the 4th position in the pattern cycle
  sorry

end units_digit_2008_pow_2008_l127_127142


namespace change_in_surface_area_zero_l127_127667

-- Original rectangular solid dimensions
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

-- Smaller prism dimensions
structure SmallerPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Conditions
def originalSolid : RectangularSolid := { length := 4, width := 3, height := 2 }
def removedPrism : SmallerPrism := { length := 1, width := 1, height := 2 }

-- Surface area calculation function
def surface_area (solid : RectangularSolid) : ℝ := 
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

-- Calculate the change in surface area
theorem change_in_surface_area_zero :
  let original_surface_area := surface_area originalSolid
  let removed_surface_area := (removedPrism.length * removedPrism.height)
  let new_exposed_area := (removedPrism.length * removedPrism.height)
  (original_surface_area - removed_surface_area + new_exposed_area) = original_surface_area :=
by
  sorry

end change_in_surface_area_zero_l127_127667


namespace find_f_at_4_l127_127621

noncomputable def f : ℝ → ℝ := sorry -- We assume such a function exists

theorem find_f_at_4:
  (∀ x : ℝ, f (4^x) + x * f (4^(-x)) = 3) → f (4) = 0 := by
  intro h
  -- Proof would go here, but is omitted as per instructions
  sorry

end find_f_at_4_l127_127621


namespace half_difference_donation_l127_127306

def margoDonation : ℝ := 4300
def julieDonation : ℝ := 4700

theorem half_difference_donation : (julieDonation - margoDonation) / 2 = 200 := by
  sorry

end half_difference_donation_l127_127306


namespace train_passes_bridge_in_20_seconds_l127_127809

def train_length : ℕ := 360
def bridge_length : ℕ := 140
def train_speed_kmh : ℕ := 90

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance : ℕ := train_length + bridge_length
noncomputable def travel_time : ℝ := total_distance / train_speed_ms

theorem train_passes_bridge_in_20_seconds :
  travel_time = 20 := by
  sorry

end train_passes_bridge_in_20_seconds_l127_127809


namespace number_of_cds_on_shelf_l127_127820

-- Definitions and hypotheses
def cds_per_rack : ℕ := 8
def racks_per_shelf : ℕ := 4

-- Theorem statement
theorem number_of_cds_on_shelf :
  cds_per_rack * racks_per_shelf = 32 :=
by sorry

end number_of_cds_on_shelf_l127_127820


namespace point_inside_circle_l127_127405

theorem point_inside_circle (O A : Type) (r OA : ℝ) (h1 : r = 6) (h2 : OA = 5) :
  OA < r :=
by
  sorry

end point_inside_circle_l127_127405


namespace one_hundred_fiftieth_digit_l127_127790

theorem one_hundred_fiftieth_digit : (0.135 : Real) * 10^150 = 1 :=
by
  sorry

end one_hundred_fiftieth_digit_l127_127790


namespace heather_total_distance_l127_127556

-- Definitions for distances walked
def distance_car_to_entrance : ℝ := 0.33
def distance_entrance_to_rides : ℝ := 0.33
def distance_rides_to_car : ℝ := 0.08

-- Statement of the problem to be proven
theorem heather_total_distance :
  distance_car_to_entrance + distance_entrance_to_rides + distance_rides_to_car = 0.74 :=
by
  sorry

end heather_total_distance_l127_127556


namespace sqrt_x_minus_1_meaningful_example_l127_127857

theorem sqrt_x_minus_1_meaningful_example :
  ∃ x : ℝ, x - 1 ≥ 0 ∧ x = 2 :=
by
  use 2
  split
  · linarith
  · refl

end sqrt_x_minus_1_meaningful_example_l127_127857


namespace simplify_expression1_simplify_expression2_l127_127888

variable {x y : ℝ} -- Declare x and y as real numbers

theorem simplify_expression1 :
  3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 :=
sorry

theorem simplify_expression2 :
  3 * x^2 * y - (2 * x * y - 2 * (x * y - (3/2) * x^2 * y) + x^2 * y^2) = - x^2 * y^2 :=
sorry

end simplify_expression1_simplify_expression2_l127_127888


namespace circle_area_isosceles_triangle_l127_127193

theorem circle_area_isosceles_triangle : 
  ∀ (A B C : Type) (AB AC : Type) (a b c : ℝ),
  a = 5 →
  b = 5 →
  c = 4 →
  isosceles_triangle A B C a b c →
  circle_passes_through_vertices A B C →
  ∃ (r : ℝ), 
    area_of_circle_passing_through_vertices A B C = (15625 * π) / 1764 :=
by intros A B C AB AC a b c ha hb hc ht hcirc
   sorry

end circle_area_isosceles_triangle_l127_127193


namespace remainder_of_division_l127_127023

theorem remainder_of_division (L S R : ℕ) (hL : L = 1620) (h_diff : L - S = 1365) (h_div : L = 6 * S + R) : R = 90 :=
by {
  -- Since we are not providing the proof, we use sorry
  sorry
}

end remainder_of_division_l127_127023


namespace find_multiplier_l127_127973

theorem find_multiplier (x : ℝ) : 3 - 3 * x < 14 ↔ x = -3 :=
by {
  sorry
}

end find_multiplier_l127_127973


namespace treasures_first_level_is_4_l127_127039

-- Definitions based on conditions
def points_per_treasure : ℕ := 5
def treasures_second_level : ℕ := 3
def score_second_level : ℕ := treasures_second_level * points_per_treasure
def total_score : ℕ := 35
def points_first_level : ℕ := total_score - score_second_level

-- Main statement to prove
theorem treasures_first_level_is_4 : points_first_level / points_per_treasure = 4 := 
by
  -- We are skipping the proof here and using sorry.
  sorry

end treasures_first_level_is_4_l127_127039


namespace sum_of_nonnegative_reals_l127_127129

theorem sum_of_nonnegative_reals (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 52) (h2 : a * b + b * c + c * a = 24) (h3 : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :
  a + b + c = 10 :=
sorry

end sum_of_nonnegative_reals_l127_127129


namespace find_first_term_l127_127398

variable {a : ℕ → ℕ}

-- Given conditions
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) + a n = 4 * n

-- Question to prove
theorem find_first_term : a 0 = 1 :=
sorry

end find_first_term_l127_127398


namespace proportion_of_second_prize_winners_l127_127670

-- conditions
variables (A B C : ℝ) -- A, B, and C represent the proportions of first, second, and third prize winners respectively.
variables (h1 : A + B = 3 / 4)
variables (h2 : B + C = 2 / 3)

-- statement
theorem proportion_of_second_prize_winners : B = 5 / 12 :=
by
  sorry

end proportion_of_second_prize_winners_l127_127670


namespace vector_parallel_addition_l127_127968

theorem vector_parallel_addition 
  (x : ℝ)
  (a : ℝ × ℝ := (2, 1))
  (b : ℝ × ℝ := (x, -2)) 
  (h_parallel : 2 / x = 1 / -2) :
  a + b = (-2, -1) := 
by
  -- While the proof is omitted, the statement is complete and correct.
  sorry

end vector_parallel_addition_l127_127968


namespace max_value_pq_qr_rs_sp_l127_127624

variable (p q r s : ℕ)

theorem max_value_pq_qr_rs_sp :
  (p = 1 ∨ p = 3 ∨ p = 5 ∨ p = 7) →
  (q = 1 ∨ q = 3 ∨ q = 5 ∨ q = 7) →
  (r = 1 ∨ r = 3 ∨ r = 5 ∨ r = 7) →
  (s = 1 ∨ s = 3 ∨ s = 5 ∨ s = 7) →
  (p ≠ q) →
  (p ≠ r) →
  (p ≠ s) →
  (q ≠ r) →
  (q ≠ s) →
  (r ≠ s) →
  pq + qr + rs + sp ≤ 64 :=
sorry

end max_value_pq_qr_rs_sp_l127_127624


namespace largest_n_det_A_nonzero_l127_127750

open Matrix

noncomputable section

-- Define the function that creates each element of the matrix A
def a_ij (i j : ℕ) : ℤ :=
  (i^j + j^i) % 3

-- Define the matrix A using the given conditions
def matrix_A (n : ℕ) : Matrix (Fin n) (Fin n) ℤ :=
  of_fun (λ i j, a_ij i.val j.val)

-- State that the largest n for which the determinant of A is not zero is 5
theorem largest_n_det_A_nonzero : ∀ n, det (matrix_A n) ≠ 0 ↔ n ≤ 5 := sorry

end largest_n_det_A_nonzero_l127_127750


namespace probability_third_white_no_replacement_probability_red_no_more_than_4_in_6_draws_with_replacement_l127_127169

noncomputable section

-- Problem 1: Probability of drawing a white ball on the third draw without replacement is 1/3.
theorem probability_third_white_no_replacement :
  let red := 4
  let white := 2
  let totalBalls := red + white
  let totalWaysToDraw3 := Nat.choose totalBalls 3
  let favorableWays := Nat.choose (totalBalls - 1) 2 * Nat.choose white 1
  let probability := favorableWays / totalWaysToDraw3
  probability = 1 / 3 :=
by
  sorry

-- Problem 2: Probability of drawing red balls no more than 4 times in 6 draws with replacement is 441/729.
theorem probability_red_no_more_than_4_in_6_draws_with_replacement :
  let red := 4
  let white := 2
  let totalBalls := red + white
  let p_red := red / totalBalls
  let p_X5 := Nat.choose 6 5 * p_red^5 * (1 - p_red)
  let p_X6 := Nat.choose 6 6 * p_red^6
  let probability := 1 - p_X5 - p_X6
  probability = 441 / 729 :=
by
  sorry

end probability_third_white_no_replacement_probability_red_no_more_than_4_in_6_draws_with_replacement_l127_127169


namespace find_x_plus_inv_x_l127_127257

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end find_x_plus_inv_x_l127_127257


namespace select_pairs_eq_l127_127720

open Set

-- Definitions for sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Statement of the theorem
theorem select_pairs_eq :
  {p | p.1 ∈ A ∧ p.2 ∈ B} = {(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)} :=
by sorry

end select_pairs_eq_l127_127720


namespace factorize_x4_minus_64_l127_127388

theorem factorize_x4_minus_64 (x : ℝ) : (x^4 - 64) = (x^2 - 8) * (x^2 + 8) :=
by sorry

end factorize_x4_minus_64_l127_127388


namespace total_red_and_green_peaches_l127_127349

def red_peaches : ℕ := 6
def green_peaches : ℕ := 16

theorem total_red_and_green_peaches :
  red_peaches + green_peaches = 22 :=
  by 
    sorry

end total_red_and_green_peaches_l127_127349


namespace sum_of_edges_l127_127339

-- Define the number of edges for a triangle and a rectangle
def edges_triangle : Nat := 3
def edges_rectangle : Nat := 4

-- The theorem states that the sum of the edges of a triangle and a rectangle is 7
theorem sum_of_edges : edges_triangle + edges_rectangle = 7 := 
by
  -- proof omitted
  sorry

end sum_of_edges_l127_127339


namespace articles_count_l127_127121

noncomputable def cost_price_per_article : ℝ := 1
noncomputable def selling_price_per_article (x : ℝ) : ℝ := x / 16
noncomputable def profit : ℝ := 0.50

theorem articles_count (x : ℝ) (h1 : cost_price_per_article * x = selling_price_per_article x * 16)
                       (h2 : selling_price_per_article 16 = cost_price_per_article * (1 + profit)) :
  x = 24 :=
by
  sorry

end articles_count_l127_127121


namespace max_diff_real_roots_l127_127793

-- Definitions of the quadratic equations
def eq1 (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eq2 (a b c x : ℝ) : Prop := b * x^2 + c * x + a = 0
def eq3 (a b c x : ℝ) : Prop := c * x^2 + a * x + b = 0

-- The proof statement
theorem max_diff_real_roots (a b c : ℝ) (hnz : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  ∃ x y : ℝ, eq1 a b c x ∧ eq1 a b c y ∧ eq2 a b c x ∧ eq2 a b c y ∧ eq3 a b c x ∧ eq3 a b c y ∧ 
  abs (x - y) = 0 := sorry

end max_diff_real_roots_l127_127793


namespace lcm_15_48_eq_240_l127_127635

def is_least_common_multiple (n a b : Nat) : Prop :=
  n % a = 0 ∧ n % b = 0 ∧ ∀ m, (m % a = 0 ∧ m % b = 0) → n ≤ m

theorem lcm_15_48_eq_240 : is_least_common_multiple 240 15 48 :=
by
  sorry

end lcm_15_48_eq_240_l127_127635


namespace persistence_of_2_persistence_iff_2_l127_127815

def is_persistent (T : ℝ) : Prop :=
  ∀ (a b c d : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
                    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ d ≠ 1) →
    (a + b + c + d = T) →
    (1 / a + 1 / b + 1 / c + 1 / d = T) →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) + 1 / (1 - d) = T)

theorem persistence_of_2 : is_persistent 2 :=
by
  -- The proof is omitted as per instructions
  sorry

theorem persistence_iff_2 (T : ℝ) : is_persistent T ↔ T = 2 :=
by
  -- The proof is omitted as per instructions
  sorry

end persistence_of_2_persistence_iff_2_l127_127815


namespace midpoint_trajectory_l127_127960

theorem midpoint_trajectory (x y : ℝ) :
  (∃ B C : ℝ × ℝ, B ≠ C ∧ (B.1^2 + B.2^2 = 25) ∧ (C.1^2 + C.2^2 = 25) ∧ 
                   (x, y) = ((B.1 + C.1)/2, (B.2 + C.2)/2) ∧ 
                   (B.1 - C.1)^2 + (B.2 - C.2)^2 = 36) →
  x^2 + y^2 = 16 :=
sorry

end midpoint_trajectory_l127_127960


namespace num_solutions_system_eqns_l127_127147

theorem num_solutions_system_eqns :
  ∃ (c : ℕ), 
    (∀ (a1 a2 a3 a4 a5 a6 : ℕ), 
       a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 = 26 ∧ 
       a1 + a2 + a3 + a4 + a5 + a6 = 5 → 
       (a1, a2, a3, a4, a5, a6) ∈ (solutions : Finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ))) ∧
    solutions.card = 5 := sorry

end num_solutions_system_eqns_l127_127147


namespace correct_answer_l127_127866

variables (x y : ℝ)

def cost_equations (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 120) ∧ (2 * x - y = 20)

theorem correct_answer : cost_equations x y :=
sorry

end correct_answer_l127_127866


namespace problem_statement_l127_127251

variable (x : ℝ) (x₀ : ℝ)

def p : Prop := ∀ x > 0, x + 4 / x ≥ 4

def q : Prop := ∃ x₀ ∈ Set.Ioi (0 : ℝ), 2 * x₀ = 1 / 2

theorem problem_statement : p ∧ ¬q :=
by
  sorry

end problem_statement_l127_127251


namespace beetle_total_distance_l127_127172

theorem beetle_total_distance 
  (r_outer : ℝ) (r_middle : ℝ) (r_inner : ℝ)
  (r_outer_eq : r_outer = 25)
  (r_middle_eq : r_middle = 15)
  (r_inner_eq : r_inner = 5)
  : (1/3 * 2 * Real.pi * r_middle + (r_outer - r_middle) + 1/2 * 2 * Real.pi * r_inner + 2 * r_outer + (r_middle - r_inner)) = (15 * Real.pi + 70) :=
by
  rw [r_outer_eq, r_middle_eq, r_inner_eq]
  have := Real.pi
  sorry

end beetle_total_distance_l127_127172


namespace distance_Tim_covers_l127_127799

theorem distance_Tim_covers (initial_distance : ℕ) (tim_speed elan_speed : ℕ) (double_speed_time : ℕ)
  (h_initial_distance : initial_distance = 30)
  (h_tim_speed : tim_speed = 10)
  (h_elan_speed : elan_speed = 5)
  (h_double_speed_time : double_speed_time = 1) :
  ∃ t d : ℕ, d = 20 ∧ t ∈ {t | t = d / tim_speed + (initial_distance - d) / (tim_speed * 2)} :=
sorry

end distance_Tim_covers_l127_127799


namespace product_eq_one_l127_127140

theorem product_eq_one (a b c : ℝ) (h1 : a^2 + 2 = b^4) (h2 : b^2 + 2 = c^4) (h3 : c^2 + 2 = a^4) : 
  (a^2 - 1) * (b^2 - 1) * (c^2 - 1) = 1 :=
sorry

end product_eq_one_l127_127140


namespace isosceles_triangle_EF_length_l127_127745

theorem isosceles_triangle_EF_length (DE DF EF DK EK KF : ℝ)
  (h1 : DE = 5) (h2 : DF = 5) (h3 : DK^2 + EK^2 = DE^2) (h4 : DK^2 + KF^2 = EF^2)
  (h5 : EK + KF = EF) (h6 : EK = 4 * KF) :
  EF = Real.sqrt 10 :=
by sorry

end isosceles_triangle_EF_length_l127_127745


namespace solve_problem_l127_127580

theorem solve_problem (a : ℝ) (x : ℝ) (h1 : 3 * x + |a - 2| = -3) (h2 : 3 * x + 4 = 0) :
  (a = 3 ∨ a = 1) → ((a - 2) ^ 2010 - 2 * a + 1 = -4 ∨ (a - 2) ^ 2010 - 2 * a + 1 = 0) :=
by {
  sorry
}

end solve_problem_l127_127580


namespace maximum_volume_prism_l127_127135

-- Define the conditions
variables {l w h : ℝ}
axiom area_sum_eq : 2 * h * l + l * w = 30

-- Define the volume of the prism
def volume (l w h : ℝ) : ℝ := l * w * h

-- Statement to be proved
theorem maximum_volume_prism : 
  (∃ l w h : ℝ, 2 * h * l + l * w = 30 ∧ 
  ∀ u v t : ℝ, 2 * t * u + u * v = 30 → l * w * h ≥ u * v * t) → volume l w h = 112.5 :=
by
  sorry

end maximum_volume_prism_l127_127135


namespace cost_price_of_article_l127_127062

theorem cost_price_of_article (M : ℝ) (SP : ℝ) (C : ℝ) 
  (hM : M = 65)
  (hSP : SP = 0.95 * M)
  (hProfit : SP = 1.30 * C) : 
  C = 47.50 :=
by 
  sorry

end cost_price_of_article_l127_127062


namespace ratio_shaded_area_to_circle_area_l127_127016

noncomputable def segmentAB : ℝ := 10
noncomputable def segmentAC : ℝ := 6
noncomputable def segmentCB : ℝ := 4

noncomputable def radius_big_semicircle : ℝ := segmentAB / 2
noncomputable def radius_mid_semicircle : ℝ := segmentAC / 2
noncomputable def radius_small_semicircle : ℝ := segmentCB / 2

noncomputable def area_big_semicircle : ℝ := (1 / 2) * real.pi * radius_big_semicircle ^ 2
noncomputable def area_mid_semicircle : ℝ := (1 / 2) * real.pi * radius_mid_semicircle ^ 2
noncomputable def area_small_semicircle : ℝ := (1 / 2) * real.pi * radius_small_semicircle ^ 2

noncomputable def total_shaded_area : ℝ := 
  area_big_semicircle - area_mid_semicircle - area_small_semicircle

noncomputable def area_circle_diameterCB : ℝ := real.pi * radius_small_semicircle ^ 2

theorem ratio_shaded_area_to_circle_area : 
  (total_shaded_area / area_circle_diameterCB) = 1.5 :=
by
  sorry

end ratio_shaded_area_to_circle_area_l127_127016


namespace probability_1_lt_X_lt_2_l127_127530

noncomputable theory

-- Assume X is a random variable following normal distribution N(1, 4)
def X : ProbabilityTheory.ProbMeasure ℝ := 
  ProbabilityTheory.ProbMeasure.normal 1 (real.sqrt 4)  -- N(mean=1, std=sqrt(variance))

-- Given P(X < 2) = 0.72
axiom P_X_less_than_2 : ProbabilityTheory.ProbMeasure.cdf X 2 = 0.72

-- The theorem to be proven
theorem probability_1_lt_X_lt_2 : 
  ProbabilityTheory.ProbMeasure.prob (set.Ioo 1 2) = 0.22 :=
sorry

end probability_1_lt_X_lt_2_l127_127530


namespace fifth_coordinate_is_14_l127_127485

theorem fifth_coordinate_is_14
  (a : Fin 16 → ℝ)
  (h_1 : a 0 = 2)
  (h_16 : a 15 = 47)
  (h_avg : ∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) :
  a 4 = 14 :=
by
  sorry

end fifth_coordinate_is_14_l127_127485


namespace square_area_l127_127607

theorem square_area (x : ℝ) (G H : ℝ) (hyp_1 : 0 ≤ G) (hyp_2 : G ≤ x) (hyp_3 : 0 ≤ H) (hyp_4 : H ≤ x) (AG : ℝ) (GH : ℝ) (HD : ℝ)
  (hyp_5 : AG = 20) (hyp_6 : GH = 20) (hyp_7 : HD = 20) (hyp_8 : x = 20 * Real.sqrt 2) :
  x^2 = 800 :=
by
  sorry

end square_area_l127_127607


namespace total_tiles_number_l127_127926

-- Define the conditions based on the problem statement
def square_floor_tiles (s : ℕ) : ℕ := s * s

def black_tiles_count (s : ℕ) : ℕ := 3 * s - 3

-- The main theorem statement: given the number of black tiles as 201,
-- prove that the total number of tiles is 4624
theorem total_tiles_number (s : ℕ) (h₁ : black_tiles_count s = 201) : 
  square_floor_tiles s = 4624 :=
by
  -- This is where the proof would go
  sorry

end total_tiles_number_l127_127926


namespace correct_statement_l127_127340

def degree (term : String) : ℕ :=
  if term = "1/2πx^2" then 2
  else if term = "-4x^2y" then 3
  else 0

def coefficient (term : String) : ℤ :=
  if term = "-4x^2y" then -4
  else if term = "3(x+y)" then 3
  else 0

def is_monomial (term : String) : Bool :=
  if term = "8" then true
  else false

theorem correct_statement : 
  (degree "1/2πx^2" ≠ 3) ∧ 
  (coefficient "-4x^2y" ≠ 4) ∧ 
  (is_monomial "8" = true) ∧ 
  (coefficient "3(x+y)" ≠ 3) := 
by
  sorry

end correct_statement_l127_127340


namespace arithmetic_sequence_sum_l127_127841

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (a_1 : ℚ) (d : ℚ) (m : ℕ) 
    (ha1 : a_1 = 2) 
    (ha2 : a 2 + a 8 = 24)
    (ham : 2 * a m = 24) 
    (h_sum : ∀ n, S n = (n * (2 * a_1 + (n - 1) * d)) / 2) 
    (h_an : ∀ n, a n = a_1 + (n - 1) * d) : 
    S (2 * m) = 265 / 2 :=
by
    sorry

end arithmetic_sequence_sum_l127_127841


namespace weight_box_plate_cups_l127_127775

theorem weight_box_plate_cups (b p c : ℝ) 
  (h₁ : b + 20 * p + 30 * c = 4.8)
  (h₂ : b + 40 * p + 50 * c = 8.4) : 
  b + 10 * p + 20 * c = 3 :=
sorry

end weight_box_plate_cups_l127_127775


namespace geometric_sequence_sum_twenty_terms_l127_127961

noncomputable def geom_seq_sum : ℕ → ℕ → ℕ := λ a r =>
  if r = 1 then a * (1 + 20 - 1) else a * ((1 - r^20) / (1 - r))

theorem geometric_sequence_sum_twenty_terms (a₁ q : ℕ) (h1 : a₁ * (q + 2) = 4) (h2 : (a₃:ℕ) * (q ^ 4) = (a₁ : ℕ) * (q ^ 4)) :
  geom_seq_sum a₁ q = 2^20 - 1 :=
sorry

end geometric_sequence_sum_twenty_terms_l127_127961


namespace valid_y_values_for_triangle_l127_127505

-- Define the triangle inequality conditions for sides 8, 11, and y^2
theorem valid_y_values_for_triangle (y : ℕ) (h_pos : y > 0) :
  (8 + 11 > y^2) ∧ (8 + y^2 > 11) ∧ (11 + y^2 > 8) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by
  sorry

end valid_y_values_for_triangle_l127_127505


namespace total_wash_time_l127_127448

theorem total_wash_time (clothes_time : ℕ) (towels_time : ℕ) (sheets_time : ℕ) (total_time : ℕ) 
  (h1 : clothes_time = 30) 
  (h2 : towels_time = 2 * clothes_time) 
  (h3 : sheets_time = towels_time - 15) 
  (h4 : total_time = clothes_time + towels_time + sheets_time) : 
  total_time = 135 := 
by 
  sorry

end total_wash_time_l127_127448


namespace sum_mod_16_l127_127685

theorem sum_mod_16 :
  (70 + 71 + 72 + 73 + 74 + 75 + 76 + 77) % 16 = 0 := 
by
  sorry

end sum_mod_16_l127_127685


namespace distinct_cube_constructions_l127_127220

theorem distinct_cube_constructions :
  let group_rotations := 7
  let identity_fixed := binomial 8 5
  let edge_rotations_fixed := 0
  (identity_fixed + 6 * edge_rotations_fixed) / group_rotations = 8 :=
by 
  let group_rotations := 7
  let identity_fixed := binomial 8 5
  let edge_rotations_fixed := 0
  have identity_contrib : identity_fixed = 56 := rfl
  have edge_contribs : 6 * edge_rotations_fixed = 0 := rfl
  show (identity_fixed + 6 * edge_rotations_fixed) / group_rotations = 8
  calc (identity_fixed + 6 * edge_rotations_fixed) / group_rotations
    = (56 + 0) / 7 : by rw [identity_contrib, edge_contribs]
    ... = 8 : rfl

end distinct_cube_constructions_l127_127220


namespace paul_money_left_l127_127884

-- Conditions
def cost_of_bread : ℕ := 2
def cost_of_butter : ℕ := 3
def cost_of_juice : ℕ := 2 * cost_of_bread
def total_money : ℕ := 15

-- Definition of total cost
def total_cost := cost_of_bread + cost_of_butter + cost_of_juice

-- Statement of the theorem
theorem paul_money_left : total_money - total_cost = 6 := by
  -- Sorry, implementation skipped
  sorry

end paul_money_left_l127_127884


namespace percentage_error_in_area_l127_127798

theorem percentage_error_in_area (s : ℝ) (h : s ≠ 0) :
  let s' := 1.02 * s
  let A := s^2
  let A' := s'^2
  ((A' - A) / A) * 100 = 4.04 := by
  sorry

end percentage_error_in_area_l127_127798


namespace find_a_l127_127723

def setA (a : ℝ) : Set ℝ := {2, 4, a^3 - 2 * a^2 - a + 7}
def setB (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem find_a (a : ℝ) : 
  (setA a ∩ setB a = {2, 5}) → (a = -1 ∨ a = 2) :=
sorry

end find_a_l127_127723


namespace seat_arrangement_l127_127327

theorem seat_arrangement (seats : ℕ) (people : ℕ) (min_empty_between : ℕ) : 
  seats = 9 ∧ people = 3 ∧ min_empty_between = 2 → 
  ∃ ways : ℕ, ways = 60 :=
by
  intro h
  sorry

end seat_arrangement_l127_127327


namespace isosceles_triangle_vertex_angle_l127_127587

theorem isosceles_triangle_vertex_angle (θ : ℝ) (h₀ : θ = 80) (h₁ : ∃ (x y z : ℝ), (x = y ∨ y = z ∨ z = x) ∧ x + y + z = 180) : θ = 80 ∨ θ = 20 := 
sorry

end isosceles_triangle_vertex_angle_l127_127587


namespace problem1_problem2_problem3_l127_127861

-- Problem 1
def s_type_sequence (a : ℕ → ℕ) : Prop := 
∀ n ≥ 1, a (n+1) - a n > 3

theorem problem1 (a : ℕ → ℕ) (h₀ : a 1 = 4) (h₁ : a 2 = 8) 
  (h₂ : ∀ n ≥ 2, a n + a (n - 1) = 8 * n - 4) : s_type_sequence a := 
sorry

-- Problem 2
theorem problem2 (a : ℕ → ℕ) (h₀ : ∀ n m, a (n * m) = (a n) ^ m)
  (b : ℕ → ℕ) (h₁ : ∀ n, b n = (3 * a n) / 4)
  (h₂ : s_type_sequence a)
  (h₃ : ¬ s_type_sequence b) : 
  (∀ n, a n = 2^(n+1)) ∨ (∀ n, a n = 2 * 3^(n-1)) ∨ (∀ n, a n = 5^ (n-1)) :=
sorry

-- Problem 3
theorem problem3 (c : ℕ → ℕ) 
  (h₀ : c 2 = 9)
  (h₁ : ∀ n ≥ 2, (1 / n - 1 / (n + 1)) * (2 + 1 / c n) ≤ 1 / c (n - 1) + 1 / c n 
               ∧ 1 / c (n - 1) + 1 / c n ≤ (1 / n - 1 / (n + 1)) * (2 + 1 / c (n-1))) :
  ∃ f : ℕ → ℕ, (s_type_sequence c) ∧ (∀ n, c n = (n + 1)^2) := 
sorry

end problem1_problem2_problem3_l127_127861


namespace reflected_light_ray_equation_l127_127662

-- Definitions for the points and line
structure Point := (x : ℝ) (y : ℝ)

-- Given points M and N
def M : Point := ⟨2, 6⟩
def N : Point := ⟨-3, 4⟩

-- Given line l
def l (p : Point) : Prop := p.x - p.y + 3 = 0

-- The target equation of the reflected light ray
def target_equation (p : Point) : Prop := p.x - 6 * p.y + 27 = 0

-- Statement to prove
theorem reflected_light_ray_equation :
  (∃ K : Point, (M.x = 2 ∧ M.y = 6) ∧ l (⟨K.x + (K.x - M.x), K.y + (K.y - M.y)⟩)
     ∧ (N.x = -3 ∧ N.y = 4)) →
  (∀ P : Point, target_equation P ↔ (P.x - 6 * P.y + 27 = 0)) := by
sorry

end reflected_light_ray_equation_l127_127662


namespace option_C_holds_l127_127858

theorem option_C_holds (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a - b / a > b - a / b := 
  sorry

end option_C_holds_l127_127858


namespace triangle_area_l127_127285

variable (a b c k : ℝ)
variable (h1 : a = 2 * k)
variable (h2 : b = 3 * k)
variable (h3 : c = k * Real.sqrt 13)

theorem triangle_area (h_right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 * a * b) = 3 * k^2 := 
by 
  sorry

end triangle_area_l127_127285


namespace number_of_pairs_l127_127242

theorem number_of_pairs (H : ∀ x y : ℕ , 0 < x → 0 < y → x < y → 2 * x * y / (x + y) = 4 ^ 15) :
  ∃ n : ℕ, n = 29 :=
by
  sorry

end number_of_pairs_l127_127242


namespace mrs_sheridan_initial_cats_l127_127151

theorem mrs_sheridan_initial_cats (bought_cats total_cats : ℝ) (h_bought : bought_cats = 43.0) (h_total : total_cats = 54) : total_cats - bought_cats = 11 :=
by
  rw [h_bought, h_total]
  norm_num

end mrs_sheridan_initial_cats_l127_127151


namespace value_of_expression_l127_127279

theorem value_of_expression
  (a b c : ℝ)
  (h1 : |a - b| = 1)
  (h2 : |b - c| = 1)
  (h3 : |c - a| = 2)
  (h4 : a * b * c = 60) :
  (a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c) = 1 / 10 :=
sorry

end value_of_expression_l127_127279


namespace correct_statement_D_l127_127042

def is_correct_option (n : ℕ) := n = 4

theorem correct_statement_D : is_correct_option 4 :=
  sorry

end correct_statement_D_l127_127042


namespace x2022_equals_1_l127_127244

noncomputable def sequence (n : ℕ) : ℤ :=
if n = 1 then 1 else
if n = 2 then 1 else
if n = 3 then -1 else
sequence (n-1) * sequence (n-3)

theorem x2022_equals_1 : sequence 2022 = 1 :=
sorry

end x2022_equals_1_l127_127244


namespace min_value_cos_sin_l127_127832

noncomputable def min_value_expression : ℝ :=
  -1 / 2

theorem min_value_cos_sin (θ : ℝ) (hθ1 : 0 ≤ θ) (hθ2 : θ ≤ 3 * Real.pi / 2) :
  ∃ (y : ℝ), y = Real.cos (θ / 3) * (1 - Real.sin θ) ∧ y = min_value_expression :=
sorry

end min_value_cos_sin_l127_127832


namespace range_of_a_iff_l127_127267

def cubic_inequality (x : ℝ) : Prop := x^3 + 3 * x^2 - x - 3 > 0

def quadratic_inequality (x a : ℝ) : Prop := x^2 - 2 * a * x - 1 ≤ 0

def integer_solution_condition (x : ℤ) (a : ℝ) : Prop := 
  x^3 + 3 * x^2 - x - 3 > 0 ∧ x^2 - 2 * a * x - 1 ≤ 0

def range_of_a (a : ℝ) : Prop := (3 / 4 : ℝ) ≤ a ∧ a < (4 / 3 : ℝ)

theorem range_of_a_iff : 
  (∃ x : ℤ, integer_solution_condition x a) ↔ range_of_a a := 
sorry

end range_of_a_iff_l127_127267


namespace inequality_proof_l127_127454

noncomputable def problem_statement (x y z : ℝ) (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z) 
  (condition : x * y * z + x * y + y * z + z * x = x + y + z + 1) : Prop :=
  (1 / 3) * (Real.sqrt ((1 + x^2) / (1 + x)) + Real.sqrt ((1 + y^2) / (1 + y)) + Real.sqrt ((1 + z^2) / (1 + z))) ≤ 
  ((x + y + z) / 3) ^ (5 / 8)

-- The statement below is what needs to be proven.
theorem inequality_proof (x y z : ℝ) (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z) 
  (condition : x * y * z + x * y + y * z + z * x = x + y + z + 1) : problem_statement x y z positive_x positive_y positive_z condition :=
sorry

end inequality_proof_l127_127454


namespace total_supervisors_l127_127027

def buses : ℕ := 7
def supervisors_per_bus : ℕ := 3

theorem total_supervisors : buses * supervisors_per_bus = 21 := 
by
  have h : buses * supervisors_per_bus = 21 := by sorry
  exact h

end total_supervisors_l127_127027


namespace jared_current_age_condition_l127_127174

variable (t j: ℕ)

-- Conditions
def tom_current_age := 25
def tom_future_age_condition := t + 5 = 30
def jared_past_age_condition := j - 2 = 2 * (t - 2)

-- Question
theorem jared_current_age_condition : 
  (t + 5 = 30) ∧ (j - 2 = 2 * (t - 2)) → j = 48 :=
by
  sorry

end jared_current_age_condition_l127_127174


namespace longest_diagonal_of_rhombus_l127_127356

noncomputable def length_of_longest_diagonal (area : ℝ) (ratio : ℝ) :=
  (let x := (area * 8 / (ratio + 1)^2).sqrt in 4 * x)

theorem longest_diagonal_of_rhombus :
  length_of_longest_diagonal 144  (4 / 3) = 8 * Real.sqrt 6 :=
by
  sorry

end longest_diagonal_of_rhombus_l127_127356


namespace candle_flower_groupings_l127_127648

theorem candle_flower_groupings : 
  (nat.choose 4 2) * (nat.choose 9 8) = 54 :=
by sorry

end candle_flower_groupings_l127_127648


namespace sum_five_digit_binary_numbers_l127_127990

def T : set ℕ := { n | n >= 16 ∧ n <= 31 }

theorem sum_five_digit_binary_numbers :
  (∑ x in (finset.filter (∈ T) (finset.range 32)), x) = 0b111111000 :=
sorry

end sum_five_digit_binary_numbers_l127_127990


namespace TV_cost_l127_127150

theorem TV_cost (savings_furniture_fraction : ℚ)
                (original_savings : ℝ)
                (spent_on_furniture : ℝ)
                (spent_on_TV : ℝ)
                (hfurniture : savings_furniture_fraction = 2/4)
                (hsavings : original_savings = 600)
                (hspent_furniture : spent_on_furniture = original_savings * savings_furniture_fraction) :
                spent_on_TV = 300 := 
sorry

end TV_cost_l127_127150


namespace tangent_line_circle_intersection_l127_127107

open Real

noncomputable def is_tangent (θ : ℝ) : Prop :=
  abs (4 * tan θ) / sqrt ((tan θ) ^ 2 + 1) = 2

theorem tangent_line_circle_intersection (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < π) :
  is_tangent θ ↔ θ = π / 6 ∨ θ = 5 * π / 6 :=
sorry

end tangent_line_circle_intersection_l127_127107


namespace largest_prime_factor_of_12321_l127_127234

theorem largest_prime_factor_of_12321 : ∃ p : ℕ, prime p ∧ p = 43 ∧ (∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p) :=
by
  sorry

end largest_prime_factor_of_12321_l127_127234


namespace factorize_expression_l127_127228

variable (a x y : ℝ)

theorem factorize_expression : a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l127_127228


namespace mushrooms_picked_on_second_day_l127_127696

theorem mushrooms_picked_on_second_day :
  ∃ (n2 : ℕ), (∃ (n1 n3 : ℕ), n3 = 2 * n2 ∧ n1 + n2 + n3 = 65) ∧ n2 = 21 :=
by
  sorry

end mushrooms_picked_on_second_day_l127_127696


namespace harmonic_mean_pairs_count_l127_127240

theorem harmonic_mean_pairs_count :
  ∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, p.1 < p.2 ∧ 2 * p.1 * p.2 = 4^15 * (p.1 + p.2)) ∧ s.card = 29 :=
sorry

end harmonic_mean_pairs_count_l127_127240


namespace Vasya_fraction_impossible_l127_127651

theorem Vasya_fraction_impossible
  (a b n : ℕ) (h_ab : a < b) (h_na : n < a) (h_nb : n < b)
  (h1 : (a + n) / (b + n) > 3 * a / (2 * b))
  (h2 : (a - n) / (b - n) > a / (2 * b)) : false :=
by
  sorry

end Vasya_fraction_impossible_l127_127651


namespace smallest_four_digit_number_divisible_by_smallest_primes_l127_127088

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end smallest_four_digit_number_divisible_by_smallest_primes_l127_127088


namespace largest_prime_factor_12321_l127_127236

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, p.prime ∧ p ∣ 12321 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 12321 → q ≤ p :=
sorry

end largest_prime_factor_12321_l127_127236


namespace sandy_correct_sums_l127_127646

theorem sandy_correct_sums
  (c i : ℕ)
  (h1 : c + i = 30)
  (h2 : 3 * c - 2 * i = 45) :
  c = 21 :=
by
  sorry

end sandy_correct_sums_l127_127646


namespace find_ratio_l127_127408

-- Given that the tangent of angle θ (inclination angle) is -2
def tan_theta (θ : Real) : Prop := Real.tan θ = -2

theorem find_ratio (θ : Real) (h : tan_theta θ) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1 / 3 := by
  sorry

end find_ratio_l127_127408


namespace range_of_a_for_no_extreme_points_l127_127578

theorem range_of_a_for_no_extreme_points :
  ∀ (a : ℝ), (∀ x : ℝ, x * (x - 2 * a) * x + 1 ≠ 0) ↔ -1 ≤ a ∧ a ≤ 1 := sorry

end range_of_a_for_no_extreme_points_l127_127578


namespace juan_marbles_l127_127066

-- Conditions
def connie_marbles : ℕ := 39
def extra_marbles_juan : ℕ := 25

-- Theorem statement: Total marbles Juan has
theorem juan_marbles : connie_marbles + extra_marbles_juan = 64 :=
by
  sorry

end juan_marbles_l127_127066


namespace integral_solution_l127_127215

noncomputable def integral_problem : ℝ :=
  ∫ x in -1..1, 2 * real.sqrt(1 - x^2) - real.sin x

theorem integral_solution : integral_problem = real.pi := by
  sorry

end integral_solution_l127_127215


namespace angela_age_in_fifteen_years_l127_127682

-- Condition 1: Angela is currently 3 times as old as Beth
def angela_age_three_times_beth (A B : ℕ) := A = 3 * B

-- Condition 2: Angela is half as old as Derek
def angela_half_derek (A D : ℕ) := A = D / 2

-- Condition 3: Twenty years ago, the sum of their ages was equal to Derek's current age
def sum_ages_twenty_years_ago (A B D : ℕ) := (A - 20) + (B - 20) + (D - 20) = D

-- Condition 4: In seven years, the difference in the square root of Angela's age and one-third of Beth's age is a quarter of Derek's age
def age_diff_seven_years (A B D : ℕ) := Real.sqrt (A + 7) - (B + 7) / 3 = D / 4

-- Define the main theorem to be proven
theorem angela_age_in_fifteen_years (A B D : ℕ) 
  (h1 : angela_age_three_times_beth A B)
  (h2 : angela_half_derek A D) 
  (h3 : sum_ages_twenty_years_ago A B D) 
  (h4 : age_diff_seven_years A B D) :
  A + 15 = 60 := 
  sorry

end angela_age_in_fifteen_years_l127_127682


namespace triangle_is_right_triangle_l127_127139

theorem triangle_is_right_triangle 
  (A B C : ℝ)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = 180)
  (h3 : A / B = 2 / 3)
  (h4 : A / C = 2 / 5) : 
  A = 36 ∧ B = 54 ∧ C = 90 := 
sorry

end triangle_is_right_triangle_l127_127139


namespace total_balloons_l127_127984

-- Define the number of balloons each person has
def joan_balloons : ℕ := 40
def melanie_balloons : ℕ := 41

-- State the theorem about the total number of balloons
theorem total_balloons : joan_balloons + melanie_balloons = 81 :=
by
  sorry

end total_balloons_l127_127984


namespace quadratic_function_is_explicit_form_l127_127108

-- Conditions
variable {f : ℝ → ℝ}
variable (H1 : f (-1) = 0)
variable (H2 : ∀ x : ℝ, x ≤ f x ∧ f x ≤ (x^2 + 1) / 2)

-- The quadratic function we aim to prove
def quadratic_function_form_proof (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = (1/4) * x^2 + (1/2) * x + (1/4)

-- Main theorem statement
theorem quadratic_function_is_explicit_form : quadratic_function_form_proof f :=
by
  -- Placeholder for the proof
  sorry

end quadratic_function_is_explicit_form_l127_127108


namespace coordinate_fifth_point_l127_127482

theorem coordinate_fifth_point : 
  ∃ (a : Fin 16 → ℝ), 
    a 0 = 2 ∧ 
    a 15 = 47 ∧ 
    (∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) ∧ 
    a 4 = 14 := 
sorry

end coordinate_fifth_point_l127_127482


namespace identity_proof_l127_127611

theorem identity_proof (n : ℝ) (h1 : n^2 ≥ 4) (h2 : n ≠ 0) :
    (n^3 - 3*n + (n^2 - 1)*Real.sqrt (n^2 - 4) - 2) / 
    (n^3 - 3*n + (n^2 - 1)*Real.sqrt (n^2 - 4) + 2)
    = ((n + 1) * Real.sqrt (n - 2)) / ((n - 1) * Real.sqrt (n + 2)) := by
  sorry

end identity_proof_l127_127611


namespace range_of_a_l127_127407

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = x^2 - 2 * a * x + a^2 - 1) 
(h_sol : ∀ x, f (f x) ≥ 0) : a ≤ -2 :=
sorry

end range_of_a_l127_127407


namespace new_estimated_y_value_l127_127715

theorem new_estimated_y_value
  (initial_slope : ℝ) (initial_intercept : ℝ) (avg_x_initial : ℝ)
  (datapoints_removed_low_x : ℝ) (datapoints_removed_high_x : ℝ)
  (datapoints_removed_low_y : ℝ) (datapoints_removed_high_y : ℝ)
  (new_slope : ℝ) 
  (x_value : ℝ)
  (estimated_y_new : ℝ) :
  initial_slope = 1.5 →
  initial_intercept = 1 →
  avg_x_initial = 2 →
  datapoints_removed_low_x = 2.6 →
  datapoints_removed_high_x = 1.4 →
  datapoints_removed_low_y = 2.8 →
  datapoints_removed_high_y = 5.2 →
  new_slope = 1.4 →
  x_value = 6 →
  estimated_y_new = new_slope * x_value + (4 - new_slope * avg_x_initial) →
  estimated_y_new = 9.6 := by
  sorry

end new_estimated_y_value_l127_127715


namespace best_pit_numbers_l127_127488

-- Definitions based on given conditions
def total_distance_walked (n : ℕ) (x : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, (abs (i + 1 - x) * 10 : ℝ))

-- Prove the two best pit numbers to minimize the total distance is 10 and 11
theorem best_pit_numbers (n : ℕ) (h1 : n = 20) :
  ∃ x y, x = 10 ∧ y = 11 ∧ total_distance_walked n x = total_distance_walked n y :=
by
  -- The proof part is not required, so we add sorry.
  sorry

end best_pit_numbers_l127_127488


namespace gcd_1722_966_l127_127634

theorem gcd_1722_966 : Nat.gcd 1722 966 = 42 :=
  sorry

end gcd_1722_966_l127_127634


namespace inverse_of_A_cubed_l127_127274

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -2,  3],
    ![  0,  1]]

theorem inverse_of_A_cubed :
  (A_inv ^ 3) = ![![ -8,  9],
                    ![  0,  1]] :=
by sorry

end inverse_of_A_cubed_l127_127274


namespace coordinate_fifth_point_l127_127483

theorem coordinate_fifth_point : 
  ∃ (a : Fin 16 → ℝ), 
    a 0 = 2 ∧ 
    a 15 = 47 ∧ 
    (∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) ∧ 
    a 4 = 14 := 
sorry

end coordinate_fifth_point_l127_127483


namespace product_of_two_numbers_l127_127464

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : a * b = 875 :=
sorry

end product_of_two_numbers_l127_127464


namespace census_suitable_survey_l127_127369

theorem census_suitable_survey (A B C D : Prop) : 
  D := 
sorry

end census_suitable_survey_l127_127369


namespace Jack_Income_Ratio_l127_127823

noncomputable def Ernie_current_income (x : ℕ) : ℕ :=
  (4 / 5) * x

noncomputable def Jack_current_income (combined_income Ernie_current_income : ℕ) : ℕ :=
  combined_income - Ernie_current_income

theorem Jack_Income_Ratio (Ernie_previous_income combined_income : ℕ) (h₁ : Ernie_previous_income = 6000) (h₂ : combined_income = 16800) :
  let Ernie_current := Ernie_current_income Ernie_previous_income
  let Jack_current := Jack_current_income combined_income Ernie_current
  (Jack_current / Ernie_previous_income) = 2 := by
  sorry

end Jack_Income_Ratio_l127_127823


namespace chloe_profit_l127_127376

def cost_per_dozen : ℕ := 50
def sell_per_half_dozen : ℕ := 30
def total_dozens_sold : ℕ := 50

def total_cost (n: ℕ) : ℕ := n * cost_per_dozen
def total_revenue (n: ℕ) : ℕ := n * (sell_per_half_dozen * 2)
def profit (cost revenue : ℕ) : ℕ := revenue - cost

theorem chloe_profit : 
  profit (total_cost total_dozens_sold) (total_revenue total_dozens_sold) = 500 := 
by
  sorry

end chloe_profit_l127_127376


namespace ratio_p_q_l127_127155

theorem ratio_p_q 
  (total_amount : ℕ) 
  (amount_r : ℕ) 
  (ratio_q_r : ℕ × ℕ) 
  (total_amount_eq : total_amount = 1210) 
  (amount_r_eq : amount_r = 400) 
  (ratio_q_r_eq : ratio_q_r = (9, 10)) :
  ∃ (amount_p amount_q : ℕ), 
    total_amount = amount_p + amount_q + amount_r ∧ 
    (amount_q : ℕ) = 9 * (amount_r / 10) ∧ 
    (amount_p : ℕ) / (amount_q : ℕ) = 5 / 4 := 
by sorry

end ratio_p_q_l127_127155


namespace apples_count_l127_127881

def total_apples (mike_apples nancy_apples keith_apples : Nat) : Nat :=
  mike_apples + nancy_apples + keith_apples

theorem apples_count :
  total_apples 7 3 6 = 16 :=
by
  rfl

end apples_count_l127_127881


namespace least_value_y_l127_127912

theorem least_value_y
  (h : ∀ y : ℝ, 5 * y ^ 2 + 7 * y + 3 = 6 → -3 ≤ y) : 
  ∃ y : ℝ, 5 * y ^ 2 + 7 * y + 3 = 6 ∧ y = -3 :=
by
  sorry

end least_value_y_l127_127912


namespace champagne_bottles_needed_l127_127208

-- Define the initial conditions of the problem
def num_guests : ℕ := 120
def glasses_per_guest : ℕ := 2
def servings_per_bottle : ℕ := 6

-- The statement we need to prove
theorem champagne_bottles_needed : 
  (num_guests * glasses_per_guest) / servings_per_bottle = 40 := 
by
  sorry

end champagne_bottles_needed_l127_127208


namespace fruit_display_total_l127_127472

-- Define the number of bananas
def bananas : ℕ := 5

-- Define the number of oranges, twice the number of bananas
def oranges : ℕ := 2 * bananas

-- Define the number of apples, twice the number of oranges
def apples : ℕ := 2 * oranges

-- Define the total number of fruits
def total_fruits : ℕ := bananas + oranges + apples

-- Theorem statement: Prove that the total number of fruits is 35
theorem fruit_display_total : total_fruits = 35 := 
by
  -- proof will be skipped, but all necessary conditions and definitions are included to ensure the statement is valid
  sorry

end fruit_display_total_l127_127472


namespace tommys_profit_l127_127173

-- Definitions of the conditions
def crateA_cost : ℕ := 220
def crateB_cost : ℕ := 375
def crateC_cost : ℕ := 180

def crateA_count : ℕ := 2
def crateB_count : ℕ := 3
def crateC_count : ℕ := 1

def crateA_capacity : ℕ := 20
def crateB_capacity : ℕ := 25
def crateC_capacity : ℕ := 30

def crateA_rotten : ℕ := 4
def crateB_rotten : ℕ := 5
def crateC_rotten : ℕ := 3

def crateA_price_per_kg : ℕ := 5
def crateB_price_per_kg : ℕ := 6
def crateC_price_per_kg : ℕ := 7

-- Calculations based on the conditions
def total_cost : ℕ := crateA_cost + crateB_cost + crateC_cost

def sellable_weightA : ℕ := crateA_count * crateA_capacity - crateA_rotten
def sellable_weightB : ℕ := crateB_count * crateB_capacity - crateB_rotten
def sellable_weightC : ℕ := crateC_count * crateC_capacity - crateC_rotten

def revenueA : ℕ := sellable_weightA * crateA_price_per_kg
def revenueB : ℕ := sellable_weightB * crateB_price_per_kg
def revenueC : ℕ := sellable_weightC * crateC_price_per_kg

def total_revenue : ℕ := revenueA + revenueB + revenueC

def profit : ℕ := total_revenue - total_cost

-- The theorem we want to verify
theorem tommys_profit : profit = 14 := by
  sorry

end tommys_profit_l127_127173


namespace largest_set_size_l127_127997

open Nat

noncomputable def largest_set_size_condition : Prop :=
  ∃ (A : Finset ℕ), A ⊆ Finset.range (2020) ∧
  (∀ x y ∈ A, x ≠ y → ¬ prime (|x - y|)) ∧
  A.card = 505

theorem largest_set_size : largest_set_size_condition := sorry

end largest_set_size_l127_127997


namespace parallel_lines_l127_127610

theorem parallel_lines (k1 k2 l1 l2 : ℝ) :
  (∀ x, (k1 ≠ k2) -> (k1 * x + l1 ≠ k2 * x + l2)) ↔ 
  (k1 = k2 ∧ l1 ≠ l2) := 
by sorry

end parallel_lines_l127_127610


namespace arithmetic_sequence_sum_l127_127870

variable {a_n : ℕ → ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∀ (m n k : ℕ), m < n → (n - m) = k → a_n n = a_n m + k * (a_n 1 - a_n 0)

theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a_n →
  a_n 2 = 5 →
  a_n 6 = 33 →
  a_n 3 + a_n 5 = 38 :=
by
  intros h_seq h_a2 h_a6
  sorry

end arithmetic_sequence_sum_l127_127870


namespace pear_distribution_problem_l127_127732

-- Defining the given conditions as hypotheses
variables (G P : ℕ)

-- The first condition: P = G + 1
def condition1 : Prop := P = G + 1

-- The second condition: P = 2G - 2
def condition2 : Prop := P = 2 * G - 2

-- The main theorem to prove
theorem pear_distribution_problem (h1 : condition1 G P) (h2 : condition2 G P) :
  G = 3 ∧ P = 4 :=
by
  sorry

end pear_distribution_problem_l127_127732


namespace arithmetic_sequence_common_difference_l127_127894

theorem arithmetic_sequence_common_difference (d : ℚ) (a₁ : ℚ) (h : a₁ = -10)
  (h₁ : ∀ n ≥ 10, a₁ + (n - 1) * d > 0) :
  10 / 9 < d ∧ d ≤ 5 / 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l127_127894


namespace team_total_score_l127_127381

theorem team_total_score (Connor_score Amy_score Jason_score : ℕ)
  (h1 : Connor_score = 2)
  (h2 : Amy_score = Connor_score + 4)
  (h3 : Jason_score = 2 * Amy_score) :
  Connor_score + Amy_score + Jason_score = 20 :=
by
  sorry

end team_total_score_l127_127381


namespace cos_double_angle_l127_127576

variable (θ : Real)

theorem cos_double_angle (h : ∑' n, (Real.cos θ)^(2 * n) = 7) : Real.cos (2 * θ) = 5 / 7 := 
  by sorry

end cos_double_angle_l127_127576


namespace same_asymptotes_hyperbolas_l127_127111

theorem same_asymptotes_hyperbolas (M : ℝ) :
  (∀ x y : ℝ, ((x^2 / 9) - (y^2 / 16) = 1) ↔ ((y^2 / 32) - (x^2 / M) = 1)) →
  M = 18 :=
by
  sorry

end same_asymptotes_hyperbolas_l127_127111


namespace form_five_squares_l127_127310

-- The conditions of the problem as premises
variables (initial_configuration : Set (ℕ × ℕ))               -- Initial positions of 12 matchsticks
          (final_configuration : Set (ℕ × ℕ))                 -- Final positions of matchsticks to form 5 squares
          (fixed_matchsticks : Set (ℕ × ℕ))                    -- Positions of 6 fixed matchsticks
          (movable_matchsticks : Set (ℕ × ℕ))                 -- Positions of 6 movable matchsticks

-- Condition to avoid duplication or free ends
variables (no_duplication : Prop)
          (no_free_ends : Prop)

-- Proof statement
theorem form_five_squares : ∃ rearranged_configuration, 
  rearranged_configuration = final_configuration ∧
  initial_configuration = fixed_matchsticks ∪ movable_matchsticks ∧
  no_duplication ∧
  no_free_ends :=
sorry -- Proof omitted.

end form_five_squares_l127_127310


namespace circle_area_isosceles_triangle_l127_127192

theorem circle_area_isosceles_triangle (a b c : ℝ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 4) :
  let r := ((25 * Real.sqrt 21) / 42)
  in ∃ (O : Point) (r : ℝ), Circle O r ∧ r^2 * Real.pi = (13125 / 1764) * Real.pi := by
  sorry

end circle_area_isosceles_triangle_l127_127192


namespace find_150th_digit_l127_127791

theorem find_150th_digit (n : ℕ) (hn : n = 150) : 
  (decimal_of_fraction (5/37) n = 5) := 
sorry

end find_150th_digit_l127_127791


namespace geom_seq_inc_condition_l127_127301

theorem geom_seq_inc_condition (a₁ a₂ q : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ = a₁ * q) :
  (a₁^2 < a₂^2) ↔ 
  (∀ n m : ℕ, n < m → (a₁ * q^n) < (a₁ * q^m) ∨ ((a₁ * q^n) = (a₁ * q^m) ∧ q = 1)) :=
by
  sorry

end geom_seq_inc_condition_l127_127301


namespace all_numbers_even_l127_127842

theorem all_numbers_even
  (A B C D E : ℤ)
  (h1 : (A + B + C) % 2 = 0)
  (h2 : (A + B + D) % 2 = 0)
  (h3 : (A + B + E) % 2 = 0)
  (h4 : (A + C + D) % 2 = 0)
  (h5 : (A + C + E) % 2 = 0)
  (h6 : (A + D + E) % 2 = 0)
  (h7 : (B + C + D) % 2 = 0)
  (h8 : (B + C + E) % 2 = 0)
  (h9 : (B + D + E) % 2 = 0)
  (h10 : (C + D + E) % 2 = 0) :
  (A % 2 = 0) ∧ (B % 2 = 0) ∧ (C % 2 = 0) ∧ (D % 2 = 0) ∧ (E % 2 = 0) :=
sorry

end all_numbers_even_l127_127842


namespace Matt_received_more_pencils_than_Lauren_l127_127459

-- Definitions based on conditions
def total_pencils := 2 * 12
def pencils_to_Lauren := 6
def pencils_after_Lauren := total_pencils - pencils_to_Lauren
def pencils_left := 9
def pencils_to_Matt := pencils_after_Lauren - pencils_left

-- Formulate the problem statement
theorem Matt_received_more_pencils_than_Lauren (total_pencils := 24) (pencils_to_Lauren := 6) (pencils_after_Lauren := 18) (pencils_left := 9) (correct_answer := 3) :
  pencils_to_Matt - pencils_to_Lauren = correct_answer := 
by 
  sorry

end Matt_received_more_pencils_than_Lauren_l127_127459


namespace simplify_expression_l127_127887

theorem simplify_expression : (8 * (15 / 9) * (-45 / 40) = -(1 / 15)) :=
by
  sorry

end simplify_expression_l127_127887


namespace koala_fiber_intake_l127_127300

theorem koala_fiber_intake (x : ℝ) (h : 0.40 * x = 16) : x = 40 :=
by
  sorry

end koala_fiber_intake_l127_127300


namespace probability_sum_fifteen_l127_127733

noncomputable def prob_sum_fifteen_three_dice : ℚ :=
  let outcomes := (finset.fin_range 6).product (finset.fin_range 6).product (finset.fin_range 6)
  let valid_outcomes := outcomes.filter (λ x, x.1.1 + x.1.2 + x.2 = 15)
  valid_outcomes.card.to_rat / outcomes.card.to_rat

theorem probability_sum_fifteen : prob_sum_fifteen_three_dice = 7 / 72 := 
  sorry

end probability_sum_fifteen_l127_127733


namespace second_part_shorter_l127_127506

def length_wire : ℕ := 180
def length_part1 : ℕ := 106
def length_part2 : ℕ := length_wire - length_part1
def length_difference : ℕ := length_part1 - length_part2

theorem second_part_shorter :
  length_difference = 32 :=
by
  sorry

end second_part_shorter_l127_127506


namespace find_length_of_AB_l127_127284

open Real

theorem find_length_of_AB (A B C : ℝ) 
    (h1 : tan A = 3 / 4) 
    (h2 : B = 6) 
    (h3 : C = π / 2) : sqrt (B^2 + ((3/4) * B)^2) = 7.5 :=
by
  sorry

end find_length_of_AB_l127_127284


namespace chocolates_received_per_boy_l127_127672

theorem chocolates_received_per_boy (total_chocolates : ℕ) (total_people : ℕ)
(boys : ℕ) (girls : ℕ) (chocolates_per_girl : ℕ)
(h_total_chocolates : total_chocolates = 3000)
(h_total_people : total_people = 120)
(h_boys : boys = 60)
(h_girls : girls = 60)
(h_chocolates_per_girl : chocolates_per_girl = 3) :
  (total_chocolates - (girls * chocolates_per_girl)) / boys = 47 :=
by
  sorry

end chocolates_received_per_boy_l127_127672


namespace necessary_but_not_sufficient_for_lt_l127_127393

variable {a b : ℝ}

theorem necessary_but_not_sufficient_for_lt (h : a < b + 1) : a < b := 
sorry

end necessary_but_not_sufficient_for_lt_l127_127393


namespace necessary_but_not_sufficient_condition_l127_127665

theorem necessary_but_not_sufficient_condition (x : ℝ) : (x > 5) → (x > 4) :=
by 
  intro h
  linarith

end necessary_but_not_sufficient_condition_l127_127665


namespace geometric_sum_first_six_terms_l127_127938

theorem geometric_sum_first_six_terms : 
  let a := (1 : ℚ) / 2
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 4095 / 6144 :=
by
  -- Definitions and properties of geometric series
  sorry

end geometric_sum_first_six_terms_l127_127938


namespace triangle_BC_length_l127_127367

theorem triangle_BC_length
  (y_eq_2x2 : ∀ (x : ℝ), ∃ (y : ℝ), y = 2 * x ^ 2)
  (area_ABC : ∃ (A B C : ℝ × ℝ), 
    A = (0, 0) ∧ (∃ (a : ℝ), B = (a, 2 * a ^ 2) ∧ C = (-a, 2 * a ^ 2) ∧ 2 * a ^ 3 = 128))
  : ∃ (a : ℝ), 2 * a = 8 := 
sorry

end triangle_BC_length_l127_127367


namespace find_150th_digit_l127_127783

theorem find_150th_digit : 
  let frac := 5/37 in
  (150th_digit_after_decimal frac = 3) :=
by
  sorry

end find_150th_digit_l127_127783


namespace rectangle_area_change_l127_127623

theorem rectangle_area_change
  (L B : ℝ)
  (hL : L > 0)
  (hB : B > 0)
  (new_L : ℝ := 1.25 * L)
  (new_B : ℝ := 0.85 * B):
  (new_L * new_B = 1.0625 * (L * B)) :=
by
  sorry

end rectangle_area_change_l127_127623


namespace necessary_but_not_sufficient_condition_l127_127654

noncomputable def condition (m : ℝ) : Prop := 1 < m ∧ m < 3

def represents_ellipse (m : ℝ) (x y : ℝ) : Prop :=
  (x ^ 2) / (m - 1) + (y ^ 2) / (3 - m) = 1

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (∃ x y, represents_ellipse m x y) → condition m :=
sorry

end necessary_but_not_sufficient_condition_l127_127654


namespace third_butcher_delivered_8_packages_l127_127880

variables (x y z t1 t2 t3 : ℕ)

-- Given Conditions
axiom h1 : x = 10
axiom h2 : y = 7
axiom h3 : 4 * x + 4 * y + 4 * z = 100
axiom t1_time : t1 = 8
axiom t2_time : t2 = 10
axiom t3_time : t3 = 18

-- Proof Problem
theorem third_butcher_delivered_8_packages :
  z = 8 :=
by
  -- proof to be filled
  sorry

end third_butcher_delivered_8_packages_l127_127880


namespace unique_sequence_l127_127602

theorem unique_sequence (n : ℕ) (h : 1 < n)
  (x : Fin (n-1) → ℕ)
  (h_pos : ∀ i, 0 < x i)
  (h_incr : ∀ i j, i < j → x i < x j)
  (h_symm : ∀ i : Fin (n-1), x i + x ⟨n - 2 - i.val, sorry⟩ = 2 * n)
  (h_sum : ∀ i j : Fin (n-1), x i + x j < 2 * n → ∃ k : Fin (n-1), x i + x j = x k) :
  ∀ i : Fin (n-1), x i = 2 * (i + 1) :=
by
  sorry

end unique_sequence_l127_127602


namespace unique_solution_for_a_eq_1_l127_127391

def equation (a x : ℝ) : Prop :=
  5^(x^2 - 6 * a * x + 9 * a^2) = a * x^2 - 6 * a^2 * x + 9 * a^3 + a^2 - 6 * a + 6

theorem unique_solution_for_a_eq_1 :
  (∃! x : ℝ, equation 1 x) ∧ 
  (∀ a : ℝ, (∃! x : ℝ, equation a x) → a = 1) :=
sorry

end unique_solution_for_a_eq_1_l127_127391


namespace trapezoid_length_KLMN_l127_127295

variables {K L M N P Q : Type}
variables (trapezoid KLMN : K L M N)
variable (KM : ℝ) (KP MQ LM MP : ℝ)
variables (perp1 : KP > 0) (perp2 : MQ > 0)
variables (equal1 : KM = 1) (equal2 : KP = MQ) (equal3 : LM = MP)

theorem trapezoid_length_KLMN
(equality_KM: KM = 1)
(equality_KP_MQ: KP = MQ)
(equality_LM_MP: LM = MP)
: LM = sqrt 2 := 
by sorry

end trapezoid_length_KLMN_l127_127295


namespace minimum_students_exceeds_1000_l127_127923

theorem minimum_students_exceeds_1000 (n : ℕ) :
  (∃ k : ℕ, k > 1000 ∧ k % 10 = 0 ∧ k % 14 = 0 ∧ k % 18 = 0 ∧ n = k) ↔ n = 1260 :=
sorry

end minimum_students_exceeds_1000_l127_127923


namespace binom_20_10_l127_127542

-- Definitions for the provided conditions
def binom_18_8 := 43758
def binom_18_9 := 48620
def binom_18_10 := 43758

-- The theorem we need to prove
theorem binom_20_10 : ∀
  (binom_18_8 = 43758)
  (binom_18_9 = 48620)
  (binom_18_10 = 43758),
  binomial 20 10 = 184756 :=
by
  sorry

end binom_20_10_l127_127542


namespace david_marks_in_english_l127_127221

variable (E : ℕ)
variable (marks_in_math : ℕ := 98)
variable (marks_in_physics : ℕ := 99)
variable (marks_in_chemistry : ℕ := 100)
variable (marks_in_biology : ℕ := 98)
variable (average_marks : ℚ := 98.2)
variable (num_subjects : ℕ := 5)

theorem david_marks_in_english 
  (H1 : average_marks = (E + marks_in_math + marks_in_physics + marks_in_chemistry + marks_in_biology) / num_subjects) :
  E = 96 :=
sorry

end david_marks_in_english_l127_127221


namespace find_number_satisfies_l127_127575

noncomputable def find_number (m : ℤ) (n : ℤ) : Prop :=
  (m % n = 2) ∧ (3 * m % n = 1)

theorem find_number_satisfies (m : ℤ) : ∃ n : ℤ, find_number m n ∧ n = 5 :=
by
  sorry

end find_number_satisfies_l127_127575


namespace perimeter_of_rectangle_l127_127163

theorem perimeter_of_rectangle (area : ℝ) (num_squares : ℕ) (square_side : ℝ) (width : ℝ) (height : ℝ) 
  (h1 : area = 216) (h2 : num_squares = 6) (h3 : area / num_squares = square_side^2)
  (h4 : width = 3 * square_side) (h5 : height = 2 * square_side) : 
  2 * (width + height) = 60 :=
by
  sorry

end perimeter_of_rectangle_l127_127163


namespace only_zero_and_one_square_equal_themselves_l127_127917

theorem only_zero_and_one_square_equal_themselves (x: ℝ) : (x^2 = x) ↔ (x = 0 ∨ x = 1) :=
by sorry

end only_zero_and_one_square_equal_themselves_l127_127917


namespace find_N_l127_127978

-- Definitions based on conditions from the problem
def remainder := 6
def dividend := 86
def divisor (Q : ℕ) := 5 * Q
def number_added_to_thrice_remainder (N : ℕ) := 3 * remainder + N
def quotient (Q : ℕ) := Q

-- The condition that relates dividend, divisor, quotient, and remainder
noncomputable def division_equation (Q : ℕ) := dividend = divisor Q * Q + remainder

-- Now, prove the condition
theorem find_N : ∃ N Q : ℕ, division_equation Q ∧ divisor Q = number_added_to_thrice_remainder N ∧ N = 2 :=
by
  sorry

end find_N_l127_127978


namespace sum_of_squares_of_roots_l127_127834

theorem sum_of_squares_of_roots : 
  ∀ (r1 r2 : ℝ), (r1 + r2 = 10) ∧ (r1 * r2 = 16) → (r1^2 + r2^2 = 68) :=
by
  intros r1 r2 h
  cases h with h1 h2
  sorry

end sum_of_squares_of_roots_l127_127834


namespace race_distance_l127_127743

theorem race_distance (D : ℝ)
  (A_time : D / 36 * 45 = D + 20) : 
  D = 80 :=
by
  sorry

end race_distance_l127_127743


namespace arithmetic_sequence_sum_l127_127164

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 :=
by
  sorry

end arithmetic_sequence_sum_l127_127164


namespace binom_20_10_l127_127541

-- Definitions for the provided conditions
def binom_18_8 := 43758
def binom_18_9 := 48620
def binom_18_10 := 43758

-- The theorem we need to prove
theorem binom_20_10 : ∀
  (binom_18_8 = 43758)
  (binom_18_9 = 48620)
  (binom_18_10 = 43758),
  binomial 20 10 = 184756 :=
by
  sorry

end binom_20_10_l127_127541


namespace price_of_other_frisbees_l127_127360

-- Lean 4 Statement
theorem price_of_other_frisbees (P : ℝ) (x : ℕ) (h1 : x ≥ 40) (h2 : P * x + 4 * (60 - x) = 200) :
  P = 3 := 
  sorry

end price_of_other_frisbees_l127_127360


namespace binary_division_remainder_l127_127222

theorem binary_division_remainder : 
  let b := 0b101101011010
  let n := 8
  b % n = 2 
:= by 
  sorry

end binary_division_remainder_l127_127222


namespace fifth_coordinate_is_14_l127_127484

theorem fifth_coordinate_is_14
  (a : Fin 16 → ℝ)
  (h_1 : a 0 = 2)
  (h_16 : a 15 = 47)
  (h_avg : ∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) :
  a 4 = 14 :=
by
  sorry

end fifth_coordinate_is_14_l127_127484


namespace tax_percentage_first_40000_l127_127132

theorem tax_percentage_first_40000 (P : ℝ) :
  (0 < P) → 
  (P / 100) * 40000 + 0.20 * 10000 = 8000 →
  P = 15 :=
by
  intros hP h
  sorry

end tax_percentage_first_40000_l127_127132


namespace surface_area_of_sphere_l127_127248

/-- Given a right prism with all vertices on a sphere, a height of 4, and a volume of 64,
    the surface area of this sphere is 48π -/
theorem surface_area_of_sphere (h : ℝ) (V : ℝ) (S : ℝ) :
  h = 4 → V = 64 → S = 48 * Real.pi := by
  sorry

end surface_area_of_sphere_l127_127248


namespace probability_at_least_one_heart_l127_127058

theorem probability_at_least_one_heart (total_cards hearts : ℕ) 
  (top_card_positions : Π n : ℕ, n = 3) 
  (non_hearts_cards : Π n : ℕ, n = total_cards - hearts) 
  (h_total_cards : total_cards = 52) (h_hearts : hearts = 13) 
  : (1 - ((39 * 38 * 37 : ℚ) / (52 * 51 * 50))) = (325 / 425) := 
by {
  sorry
}

end probability_at_least_one_heart_l127_127058


namespace find_a3_a4_a5_l127_127165

variable (a : ℕ → ℝ)

-- Recurrence relation for the sequence (condition for n ≥ 2)
axiom rec_relation (n : ℕ) (h : n ≥ 2) : 2 * a n = a (n - 1) + a (n + 1)

-- Additional conditions
axiom cond1 : a 1 + a 3 + a 5 = 9
axiom cond2 : a 3 + a 5 + a 7 = 15

-- Statement to prove
theorem find_a3_a4_a5 : a 3 + a 4 + a 5 = 12 :=
  sorry

end find_a3_a4_a5_l127_127165


namespace abs_expression_eq_five_l127_127814

theorem abs_expression_eq_five : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry -- proof omitted

end abs_expression_eq_five_l127_127814


namespace prob_1_less_X_less_2_l127_127531

noncomputable def NormalDist (mean variance : ℝ) : Type := sorry -- Placeholder for normal distribution type

variable (X : NormalDist 1 4)

axiom prob_X_less_than_2 : P(X < 2) = 0.72

theorem prob_1_less_X_less_2 : P(1 < X < 2) = 0.22 :=
by
  have h1 : P(X ≥ 2) = 1 - P(X < 2) := sorry
  have h2 : P(X > 1) = 0.5 := sorry
  have h3 : P(1 < X < 2) = P(X > 1) - P(X ≥ 2) := sorry
  show P(1 < X < 2) = 0.22
sorry

end prob_1_less_X_less_2_l127_127531


namespace find_y_value_l127_127663

theorem find_y_value :
  ∀ (y : ℝ), (dist (1, 3) (7, y) = 13) ∧ (y > 0) → y = 3 + Real.sqrt 133 :=
by
  sorry

end find_y_value_l127_127663


namespace max_g_at_8_l127_127996

noncomputable def g : ℝ → ℝ :=
  sorry -- We define g here abstractly, with nonnegative coefficients

axiom g_nonneg_coeffs : ∀ x, 0 ≤ g x
axiom g_at_4 : g 4 = 16
axiom g_at_16 : g 16 = 256

theorem max_g_at_8 : g 8 ≤ 64 :=
by sorry

end max_g_at_8_l127_127996


namespace probability_at_least_half_girls_l127_127436

theorem probability_at_least_half_girls (n : ℕ) (hn : n = 6) :
  (probability (λ (s : vector bool n), s.foldr (λ b acc, if b then acc + 1 else acc) 0 ≥ n/2))
  = 21 / 32 := by
  sorry

end probability_at_least_half_girls_l127_127436


namespace jane_crayons_l127_127434

theorem jane_crayons :
  let start := 87
  let eaten := 7
  start - eaten = 80 :=
by
  sorry

end jane_crayons_l127_127434


namespace intersect_condition_l127_127849

theorem intersect_condition (m : ℕ) (h : m ≠ 0) : 
  (∃ x y : ℝ, (3 * x - 2 * y = 0) ∧ ((x - m)^2 + y^2 = 1)) → m = 1 :=
by 
  sorry

end intersect_condition_l127_127849


namespace B_took_18_more_boxes_than_D_l127_127059

noncomputable def A_boxes : ℕ := sorry
noncomputable def B_boxes : ℕ := A_boxes + 4
noncomputable def C_boxes : ℕ := sorry
noncomputable def D_boxes : ℕ := C_boxes + 8
noncomputable def A_owes_C : ℕ := 112
noncomputable def B_owes_D : ℕ := 72

theorem B_took_18_more_boxes_than_D : (B_boxes - D_boxes) = 18 :=
sorry

end B_took_18_more_boxes_than_D_l127_127059


namespace ducks_percentage_non_heron_birds_l127_127438

theorem ducks_percentage_non_heron_birds
  (total_birds : ℕ)
  (geese_percent pelicans_percent herons_percent ducks_percent : ℝ)
  (H_geese : geese_percent = 20 / 100)
  (H_pelicans: pelicans_percent = 40 / 100)
  (H_herons : herons_percent = 15 / 100)
  (H_ducks : ducks_percent = 25 / 100)
  (hnz : total_birds ≠ 0) :
  (ducks_percent / (1 - herons_percent)) * 100 = 30 :=
by
  sorry

end ducks_percentage_non_heron_birds_l127_127438


namespace arithmetic_sum_problem_l127_127532

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sum_problem
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_terms S a)
  (h_S10 : S 10 = 4) :
  a 3 + a 8 = 4 / 5 := 
sorry

end arithmetic_sum_problem_l127_127532


namespace find_third_side_length_l127_127760

noncomputable def triangle_third_side_length (a b c : ℝ) (B C : ℝ) 
  (h1 : B = 3 * C) (h2 : b = 12) (h3 : c = 20) : Prop :=
a = 16

theorem find_third_side_length (a b c : ℝ) (B C : ℝ)
  (h1 : B = 3 * C) (h2 : b = 12) (h3 : c = 20) :
  triangle_third_side_length a b c B C h1 h2 h3 :=
sorry

end find_third_side_length_l127_127760


namespace omitted_decimal_sum_is_integer_l127_127362

def numbers : List ℝ := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

theorem omitted_decimal_sum_is_integer :
  1.05 + 1.15 + 1.25 + 1.4 + (15 : ℝ) + 1.6 + 1.75 + 1.85 + 1.95 = 27 :=
by sorry

end omitted_decimal_sum_is_integer_l127_127362


namespace football_players_count_l127_127584

def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def softball_players : ℕ := 13
def total_players : ℕ := 59

theorem football_players_count :
  total_players - (cricket_players + hockey_players + softball_players) = 18 :=
by 
  sorry

end football_players_count_l127_127584


namespace alyssa_earnings_l127_127681

theorem alyssa_earnings
    (weekly_allowance: ℤ)
    (spent_on_movies_fraction: ℤ)
    (amount_ended_with: ℤ)
    (h1: weekly_allowance = 8)
    (h2: spent_on_movies_fraction = 1 / 2)
    (h3: amount_ended_with = 12)
    : ∃ money_earned_from_car_wash: ℤ, money_earned_from_car_wash = 8 :=
by
  sorry

end alyssa_earnings_l127_127681


namespace correct_operation_B_l127_127182

theorem correct_operation_B (a b : ℝ) : 2 * a * b * b^2 = 2 * a * b^3 :=
sorry

end correct_operation_B_l127_127182


namespace probability_at_least_half_girls_l127_127435

-- Conditions
def six_children : ℕ := 6
def prob_girl : ℝ := 0.5

-- Statement to prove
theorem probability_at_least_half_girls :
  (∑ k in finset.range (six_children + 1), if 3 ≤ k then ↑(nat.binomial six_children k) * (prob_girl ^ k) * ((1 - prob_girl) ^ (six_children - k)) else 0) = 21 / 32 :=
by sorry

end probability_at_least_half_girls_l127_127435


namespace min_both_attendees_l127_127583

-- Defining the parameters and conditions
variable (n : ℕ) -- total number of attendees
variable (glasses name_tags both : ℕ) -- attendees wearing glasses, name tags, and both

-- Conditions provided in the problem
def wearing_glasses_condition (n : ℕ) (glasses : ℕ) : Prop := glasses = n / 3
def wearing_name_tags_condition (n : ℕ) (name_tags : ℕ) : Prop := name_tags = n / 2
def total_attendees_condition (n : ℕ) : Prop := n = 6

-- Theorem to prove the minimum attendees wearing both glasses and name tags is 1
theorem min_both_attendees (n glasses name_tags both : ℕ) (h1 : wearing_glasses_condition n glasses) 
  (h2 : wearing_name_tags_condition n name_tags) (h3 : total_attendees_condition n) : 
  both = 1 :=
sorry

end min_both_attendees_l127_127583


namespace driver_speed_l127_127051

theorem driver_speed (v t : ℝ) (h1 : t > 0) (h2 : v > 0) (h3 : v * t = (v + 37.5) * (3 / 8) * t) : v = 22.5 :=
by
  sorry

end driver_speed_l127_127051


namespace range_g_minus_2x_l127_127161

variable (g : ℝ → ℝ)
variable (x : ℝ)

axiom g_values : ∀ x, x ∈ Set.Icc (-4 : ℝ) 4 → 
  (g x = x ∨ g x = x - 1 ∨ g x = x - 2 ∨ g x = x - 3 ∨ g x = x - 4)

axiom g_le_2x : ∀ x, x ∈ Set.Icc (-4 : ℝ) 4 → g x ≤ 2 * x

theorem range_g_minus_2x : 
  Set.range (fun x => g x - 2 * x) = Set.Icc (-5 : ℝ) 0 :=
sorry

end range_g_minus_2x_l127_127161


namespace disjoint_subsets_mod_l127_127002

open Finset

def S : Finset ℕ := (range 12).map (λ x, x + 1)

noncomputable def number_of_disjoint_subsets (S : Finset ℕ) : ℕ :=
  (3^12 - 2 * (2^12) + 1) / 2

theorem disjoint_subsets_mod :
  (number_of_disjoint_subsets S) % 1000 = 625 := by
  sorry

end disjoint_subsets_mod_l127_127002


namespace number_of_white_balls_l127_127287

theorem number_of_white_balls (x : ℕ) : (3 : ℕ) + x = 12 → x = 9 :=
by
  intros h
  sorry

end number_of_white_balls_l127_127287


namespace faster_train_length_l127_127631

theorem faster_train_length
  (speed_faster : ℝ)
  (speed_slower : ℝ)
  (time_to_cross : ℝ)
  (relative_speed_limit: ℝ)
  (h1 : speed_faster = 108 * 1000 / 3600)
  (h2: speed_slower = 36 * 1000 / 3600)
  (h3: time_to_cross = 17)
  (h4: relative_speed_limit = 2) :
  (speed_faster - speed_slower) * time_to_cross = 340 := 
sorry

end faster_train_length_l127_127631


namespace solution_set_of_floor_equation_l127_127828

theorem solution_set_of_floor_equation (x : ℝ) : 
  (⌊⌊2 * x⌋ - 1/2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
by sorry

end solution_set_of_floor_equation_l127_127828


namespace tan_sum_of_angles_eq_neg_sqrt_three_l127_127397

theorem tan_sum_of_angles_eq_neg_sqrt_three 
  (A B C : ℝ)
  (h1 : B - A = C - B)
  (h2 : A + B + C = Real.pi) :
  Real.tan (A + C) = -Real.sqrt 3 :=
sorry

end tan_sum_of_angles_eq_neg_sqrt_three_l127_127397


namespace tan_pi_minus_alpha_l127_127402

theorem tan_pi_minus_alpha 
  (α : ℝ) 
  (h1 : Real.sin α = 1 / 3) 
  (h2 : π / 2 < α) 
  (h3 : α < π) :
  Real.tan (π - α) = Real.sqrt 2 / 4 :=
by
  sorry

end tan_pi_minus_alpha_l127_127402


namespace smallest_number_of_cubes_l127_127501

noncomputable def container_cubes (length_ft : ℕ) (height_ft : ℕ) (width_ft : ℕ) (prime_inch : ℕ) : ℕ :=
  let length_inch := length_ft * 12
  let height_inch := height_ft * 12
  let width_inch := width_ft * 12
  (length_inch / prime_inch) * (height_inch / prime_inch) * (width_inch / prime_inch)

theorem smallest_number_of_cubes :
  container_cubes 60 24 30 3 = 2764800 :=
by
  sorry

end smallest_number_of_cubes_l127_127501


namespace find_angle_C_l127_127282

noncomputable def ABC_triangle (A B C a b c : ℝ) : Prop :=
b = c * Real.cos A + Real.sqrt 3 * a * Real.sin C

theorem find_angle_C (A B C a b c : ℝ) (h : ABC_triangle A B C a b c) :
  C = π / 6 :=
sorry

end find_angle_C_l127_127282


namespace area_of_intersection_l127_127290

-- Define the region M
def in_region_M (x y : ℝ) : Prop :=
  y ≥ 0 ∧ y ≤ x ∧ y ≤ 2 - x

-- Define the region N as it changes with t
def in_region_N (t x : ℝ) : Prop :=
  t ≤ x ∧ x ≤ t + 1 ∧ 0 ≤ t ∧ t ≤ 1

-- Define the function f(t) which represents the common area of M and N
noncomputable def f (t : ℝ) : ℝ :=
  -t^2 + t + 0.5

-- Prove that f(t) is correct given the above conditions
theorem area_of_intersection (t : ℝ) :
  (∀ x y : ℝ, in_region_M x y → in_region_N t x → y ≤ f t) →
  0 ≤ t ∧ t ≤ 1 →
  f t = -t^2 + t + 0.5 :=
by
  sorry

end area_of_intersection_l127_127290


namespace binomial_20_10_l127_127539

open Nat

theorem binomial_20_10 :
  (binomial 18 8 = 43758) →
  (binomial 18 9 = 48620) →
  (binomial 18 10 = 43758) →
  binomial 20 10 = 184756 :=
by
  intros h1 h2 h3
  sorry

end binomial_20_10_l127_127539


namespace distinct_four_digit_integers_with_product_18_l127_127564

theorem distinct_four_digit_integers_with_product_18 :
  ∃ n : ℕ, n = 24 ∧ ∀ (d1 d2 d3 d4 : ℕ), (d1 * d2 * d3 * d4 = 18 ∧ 1000 ≤ 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ 1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10000) →
    set.finite { x | ∃ (d1 d2 d3 d4 : ℕ), x = 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ d1 * d2 * d3 * d4 = 18 ∧ ∀ i ∈ [d1, d2, d3, d4], 1 ≤ i ∧ i ≤ 9 } :=
begin
  sorry
end

end distinct_four_digit_integers_with_product_18_l127_127564


namespace sqrt_of_4_equals_2_l127_127766

theorem sqrt_of_4_equals_2 : Real.sqrt 4 = 2 :=
by sorry

end sqrt_of_4_equals_2_l127_127766


namespace arithmetic_sequence_equality_l127_127289

theorem arithmetic_sequence_equality {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (a20 : a ≠ c) (a2012 : b ≠ c) 
(h₄ : ∀ (i : ℕ), ∃ d : ℝ, a_n = a + i * d) : 
  1992 * a * c - 1811 * b * c - 181 * a * b = 0 := 
by {
  sorry
}

end arithmetic_sequence_equality_l127_127289


namespace sequence_x_2022_l127_127243

theorem sequence_x_2022 :
  ∃ (x : ℕ → ℤ), x 1 = 1 ∧ x 2 = 1 ∧ x 3 = -1 ∧
  (∀ n, 4 ≤ n → x n = x (n-1) * x (n-3)) ∧ x 2022 = 1 := by
  sorry

end sequence_x_2022_l127_127243


namespace cups_of_sugar_already_put_in_l127_127008

-- Defining the given conditions
variable (f s x : ℕ)

-- The total flour and sugar required
def total_flour_required := 9
def total_sugar_required := 6

-- Mary needs to add 7 more cups of flour than cups of sugar
def remaining_flour_to_sugar_difference := 7

-- Proof goal: to find how many cups of sugar Mary has already put in
theorem cups_of_sugar_already_put_in (total_flour_remaining : ℕ := 9 - 7)
    (remaining_sugar : ℕ := 9 - 7) 
    (already_added_sugar : ℕ := 6 - 2) : already_added_sugar = 4 :=
by sorry

end cups_of_sugar_already_put_in_l127_127008


namespace profit_percentage_l127_127925

-- Define the selling price and the cost price
def SP : ℝ := 100
def CP : ℝ := 86.95652173913044

-- State the theorem for profit percentage
theorem profit_percentage :
  ((SP - CP) / CP) * 100 = 15 :=
by
  sorry

end profit_percentage_l127_127925


namespace charge_per_kilo_l127_127152

variable (x : ℝ)

theorem charge_per_kilo (h : 5 * x + 10 * x + 20 * x = 70) : x = 2 := by
  -- Proof goes here
  sorry

end charge_per_kilo_l127_127152


namespace total_money_l127_127498

theorem total_money (n : ℕ) (h1 : n * 3 = 36) :
  let one_rupee := n * 1
  let five_rupee := n * 5
  let ten_rupee := n * 10
  (one_rupee + five_rupee + ten_rupee) = 192 :=
by
  -- Note: The detailed calculations would go here in the proof
  -- Since we don't need to provide the proof, we add sorry to indicate the omitted part
  sorry

end total_money_l127_127498


namespace find_fraction_l127_127159

noncomputable def fraction_of_eighths (N : ℝ) (a b : ℝ) : Prop :=
  (3/8) * N * (a/b) = 24

noncomputable def two_fifty_percent (N : ℝ) : Prop :=
  2.5 * N = 199.99999999999997

theorem find_fraction {N a b : ℝ} (h1 : fraction_of_eighths N a b) (h2 : two_fifty_percent N) :
  a/b = 4/5 :=
sorry

end find_fraction_l127_127159


namespace no_tiling_10x10_1x4_l127_127593

-- Define the problem using the given conditions
def checkerboard_tiling (n k : ℕ) : Prop :=
  ∃ t : ℕ, t * k = n * n ∧ n % k = 0

-- Prove that it is impossible to tile a 10x10 board with 1x4 tiles
theorem no_tiling_10x10_1x4 : ¬ checkerboard_tiling 10 4 :=
sorry

end no_tiling_10x10_1x4_l127_127593


namespace ROI_difference_is_correct_l127_127071

noncomputable def compound_interest (P : ℝ) (rates : List ℝ) : ℝ :=
rates.foldl (λ acc rate => acc * (1 + rate)) P

noncomputable def Emma_investment := compound_interest 300 [0.15, 0.12, 0.18]

noncomputable def Briana_investment := compound_interest 500 [0.10, 0.08, 0.14]

noncomputable def ROI_difference := Briana_investment - Emma_investment

theorem ROI_difference_is_correct : ROI_difference = 220.808 := 
sorry

end ROI_difference_is_correct_l127_127071


namespace cara_sitting_pairs_l127_127373

theorem cara_sitting_pairs : ∀ (n : ℕ), n = 7 → ∃ (pairs : ℕ), pairs = 6 :=
by
  intros n hn
  have h : n - 1 = 6 := sorry
  exact ⟨n - 1, h⟩

end cara_sitting_pairs_l127_127373


namespace smallest_N_divisible_by_p_l127_127005

theorem smallest_N_divisible_by_p (p : ℕ) (hp : Nat.Prime p)
    (N1 : ℕ) (N2 : ℕ) :
  (∃ N1 N2, 
    (N1 % p = 0 ∧ ∀ n, 1 ≤ n ∧ n < p → N1 % n = 1) ∧
    (N2 % p = 0 ∧ ∀ n, 1 ≤ n ∧ n < p → N2 % n = n - 1)
  ) :=
sorry

end smallest_N_divisible_by_p_l127_127005


namespace zadam_solution_l127_127835

noncomputable def ZadamProbability 
  (p : ℕ → ℕ → ℚ)
  (satisfies_sum : ∀ (m : Fin 7 → ℕ), (∑ i, (i.succ * m i).toNat) = 35 → m 4 = 1) :=
  ∑ m in {m : Fin 7 → ℕ | (∑ i, (i.succ * m i).toNat) = 35}, p 4 1 = 4/5

theorem zadam_solution :
  ZadamProbability
  (λ i k, (2 ^ i - 1)/2 ^ (i * k))
  (fun m => satisfies_sum m) := sorry

end zadam_solution_l127_127835


namespace heather_final_blocks_l127_127270

def heather_initial_blocks : ℝ := 86.0
def jose_shared_blocks : ℝ := 41.0

theorem heather_final_blocks : heather_initial_blocks + jose_shared_blocks = 127.0 :=
by
  sorry

end heather_final_blocks_l127_127270


namespace range_of_x_l127_127395

noncomputable def A (x : ℝ) : ℤ := Int.ceil x

theorem range_of_x (x : ℝ) (h₁ : x > 0) (h₂ : A (2 * x * A x) = 5) : x ∈ Set.Ioc 1 (5 / 4 : ℝ) :=
sorry

end range_of_x_l127_127395


namespace shoes_sold_first_week_eq_100k_l127_127657

-- Define variables for purchase price and total revenue
def purchase_price : ℝ := 180
def total_revenue : ℝ := 216

-- Define markups
def first_week_markup : ℝ := 1.25
def remaining_markup : ℝ := 1.16

-- Define the conditions
theorem shoes_sold_first_week_eq_100k (x y : ℝ) 
  (h1 : x + y = purchase_price) 
  (h2 : first_week_markup * x + remaining_markup * y = total_revenue) :
  first_week_markup * x = 100  := 
sorry

end shoes_sold_first_week_eq_100k_l127_127657


namespace good_games_count_l127_127653

theorem good_games_count :
  ∀ (g1 g2 b : ℕ), g1 = 50 → g2 = 27 → b = 74 → g1 + g2 - b = 3 := by
  intros g1 g2 b hg1 hg2 hb
  sorry

end good_games_count_l127_127653


namespace quadrilateral_area_correct_l127_127690

-- Definitions of given conditions
structure Quadrilateral :=
(W X Y Z : Type)
(WX XY YZ YW : ℝ)
(angle_WXY : ℝ)
(area : ℝ)

-- Quadrilateral satisfies given conditions
def quadrilateral_WXYZ : Quadrilateral :=
{ W := ℝ,
  X := ℝ,
  Y := ℝ,
  Z := ℝ,
  WX := 9,
  XY := 5,
  YZ := 12,
  YW := 15,
  angle_WXY := 90,
  area := 76.5 }

-- The theorem stating the area of quadrilateral WXYZ is 76.5
theorem quadrilateral_area_correct : quadrilateral_WXYZ.area = 76.5 :=
sorry

end quadrilateral_area_correct_l127_127690


namespace total_fruits_on_display_l127_127470

-- Declare the variables and conditions as hypotheses
variables (apples oranges bananas : ℕ)
variables (h1 : apples = 2 * oranges)
variables (h2 : oranges = 2 * bananas)
variables (h3 : bananas = 5)

-- Define what we want to prove
theorem total_fruits_on_display : apples + oranges + bananas = 35 :=
by sorry

end total_fruits_on_display_l127_127470


namespace binom_20_10_l127_127535

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end binom_20_10_l127_127535


namespace digit_150_after_decimal_of_5_over_37_is_3_l127_127787

theorem digit_150_after_decimal_of_5_over_37_is_3 :
  let dec := (5 / 37 : ℚ).decimalExpansion in
  dec.nthDigit 150 = 3 :=
by
  sorry

end digit_150_after_decimal_of_5_over_37_is_3_l127_127787


namespace bottles_needed_l127_127205

-- Define specific values provided in conditions
def servings_per_guest : ℕ := 2
def number_of_guests : ℕ := 120
def servings_per_bottle : ℕ := 6

-- Define total servings needed
def total_servings : ℕ := servings_per_guest * number_of_guests

-- Define the number of bottles needed (as a proof statement)
theorem bottles_needed : total_servings / servings_per_bottle = 40 := by
  /-
    The proof will go here. For now we place a sorry to mark the place where
    a proof would be required. The statement should check the equivalence of 
    number of bottles needed being 40 given the total servings divided by 
    servings per bottle.
  -/
  sorry

end bottles_needed_l127_127205


namespace trigonometric_comparison_l127_127953

theorem trigonometric_comparison :
  let a := Real.sin (2 * Real.pi / 7)
  let b := Real.cos (12 * Real.pi / 7)
  let c := Real.tan (9 * Real.pi / 7)
  a = Real.sin (2 * Real.pi / 7) →
  b = Real.cos (12 * Real.pi / 7) →
  c = Real.tan (9 * Real.pi / 7) →
  (c > a ∧ a > b) :=
by
  sorry

end trigonometric_comparison_l127_127953


namespace financing_term_years_l127_127555

def monthly_payment : Int := 150
def total_financed_amount : Int := 9000

theorem financing_term_years : 
  (total_financed_amount / monthly_payment) / 12 = 5 := 
by
  sorry

end financing_term_years_l127_127555


namespace sum_of_fractions_l127_127947

theorem sum_of_fractions :
  (1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)) = (2 / 9) :=
by
  sorry

end sum_of_fractions_l127_127947


namespace quadratic_real_roots_condition_l127_127127

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (m-1) * x₁^2 - 4 * x₁ + 1 = 0 ∧ (m-1) * x₂^2 - 4 * x₂ + 1 = 0) ↔ (m < 5 ∧ m ≠ 1) :=
by
  sorry

end quadratic_real_roots_condition_l127_127127


namespace no_solutions_eq_l127_127916

theorem no_solutions_eq (x y : ℝ) : (x + y)^2 ≠ x^2 + y^2 + 1 :=
by sorry

end no_solutions_eq_l127_127916


namespace find_opposite_pair_l127_127200

def is_opposite (x y : ℤ) : Prop := x = -y

theorem find_opposite_pair :
  ¬is_opposite 4 4 ∧ ¬is_opposite 2 2 ∧ ¬is_opposite (-8) (-8) ∧ is_opposite 4 (-4) := 
by
  sorry

end find_opposite_pair_l127_127200


namespace count_four_digit_integers_with_product_18_l127_127561

def valid_digits (n : ℕ) : Prop := 
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def digit_product_18 (a b c d : ℕ) : Prop := 
  a * b * c * d = 18

def four_digit_integer (a b c d : ℕ) : Prop := 
  valid_digits a ∧ valid_digits b ∧ valid_digits c ∧ valid_digits d

theorem count_four_digit_integers_with_product_18 : 
  (∑ a b c d in {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    ite (four_digit_integer a b c d ∧ digit_product_18 a b c d) 1 0) = 48 := 
sorry

end count_four_digit_integers_with_product_18_l127_127561


namespace joshuas_share_l127_127873

theorem joshuas_share (total amount : ℝ) (joshua_share : ℝ) (justin_share: ℝ) 
  (h1: total amount = 40) 
  (h2: joshua_share = 3 * justin_share) 
  (h3: total amount = joshua_share + justin_share) 
: joshua_share = 30 := 
by  sorry

end joshuas_share_l127_127873


namespace total_passengers_transportation_l127_127807

theorem total_passengers_transportation : 
  let passengers_one_way := 100
  let passengers_return := 60
  let first_trip_total := passengers_one_way + passengers_return
  let additional_trips := 3
  let additional_trips_total := additional_trips * first_trip_total
  let total_passengers := first_trip_total + additional_trips_total
  total_passengers = 640 := 
by
  sorry

end total_passengers_transportation_l127_127807


namespace find_a_l127_127137

theorem find_a (a : ℝ) (h_pos : 0 < a) 
  (prob : (2 / a) = (1 / 3)) : a = 6 :=
by sorry

end find_a_l127_127137


namespace library_wall_length_l127_127933

theorem library_wall_length 
  (D B : ℕ) 
  (h1: D = B) 
  (desk_length bookshelf_length leftover_space : ℝ) 
  (h2: desk_length = 2) 
  (h3: bookshelf_length = 1.5) 
  (h4: leftover_space = 1) : 
  3.5 * D + leftover_space = 8 :=
by { sorry }

end library_wall_length_l127_127933


namespace geometric_sequence_seventh_term_l127_127354

theorem geometric_sequence_seventh_term (a r: ℤ) (h1 : a = 3) (h2 : a * r ^ 5 = 729) : a * r ^ 6 = 2187 :=
by sorry

end geometric_sequence_seventh_term_l127_127354


namespace smallest_four_digit_divisible_by_2_3_5_7_11_l127_127080

theorem smallest_four_digit_divisible_by_2_3_5_7_11 :
  ∃ n : ℕ, n = 2310 ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ (n % 11 = 0) ∧ 1000 ≤ n ∧ n < 10000 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_2_3_5_7_11_l127_127080


namespace problem_part1_problem_part2_l127_127109

variable (a : ℝ)

def quadratic_solution_set_1 := {x : ℝ | x^2 + 2*x + a = 0}
def quadratic_solution_set_2 := {x : ℝ | a*x^2 + 2*x + 2 = 0}

theorem problem_part1 :
  (quadratic_solution_set_1 a = ∅ ∨ quadratic_solution_set_2 a = ∅) ∧ ¬ (quadratic_solution_set_1 a = ∅ ∧ quadratic_solution_set_2 a = ∅) →
  (1/2 < a ∧ a ≤ 1) :=
sorry

theorem problem_part2 :
  quadratic_solution_set_1 a ∪ quadratic_solution_set_2 a ≠ ∅ →
  a ≤ 1 :=
sorry

end problem_part1_problem_part2_l127_127109


namespace thomas_friends_fraction_l127_127806

noncomputable def fraction_of_bars_taken (x : ℝ) (initial_bars : ℝ) (returned_bars : ℝ) 
  (piper_bars : ℝ) (remaining_bars : ℝ) : ℝ :=
  x / initial_bars

theorem thomas_friends_fraction 
  (initial_bars : ℝ)
  (total_taken_by_all : ℝ)
  (returned_bars : ℝ)
  (piper_bars : ℝ)
  (remaining_bars : ℝ)
  (h_initial : initial_bars = 200)
  (h_remaining : remaining_bars = 110)
  (h_taken : 200 - 110 = 90)
  (h_total_taken_by_all : total_taken_by_all = 90)
  (h_returned : returned_bars = 5)
  (h_x_calculation : 2 * (total_taken_by_all + returned_bars - initial_bars) + initial_bars = total_taken_by_all + returned_bars)
  : fraction_of_bars_taken ((total_taken_by_all + returned_bars - initial_bars) + 2 * initial_bars) initial_bars returned_bars piper_bars remaining_bars = 21 / 80 :=
  sorry

end thomas_friends_fraction_l127_127806


namespace largest_prime_factor_of_12321_l127_127235

-- Definitions based on the given conditions
def n := 12321
def a := 111
def p₁ := 3
def p₂ := 37

-- Given conditions as hypotheses
theorem largest_prime_factor_of_12321 (h1 : n = a^2) (h2 : a = p₁ * p₂) (hp₁_prime : Prime p₁) (hp₂_prime : Prime p₂) :
  p₂ = 37 ∧ ∀ p, Prime p → p ∣ n → p ≤ 37 := 
by 
  sorry

end largest_prime_factor_of_12321_l127_127235


namespace sum_inequality_l127_127712

noncomputable def f (x : ℝ) : ℝ :=
  (3 * x^2 - x) / (1 + x^2)

theorem sum_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 1) : 
  f x + f y + f z ≥ 0 :=
by
  sorry

end sum_inequality_l127_127712


namespace prob_business_less25_correct_l127_127738

def prob_male : ℝ := 0.4
def prob_female : ℝ := 0.6

def prob_science : ℝ := 0.3
def prob_arts : ℝ := 0.45
def prob_business : ℝ := 0.25

def prob_male_science_25plus : ℝ := 0.4
def prob_male_arts_25plus : ℝ := 0.5
def prob_male_business_25plus : ℝ := 0.35

def prob_female_science_25plus : ℝ := 0.3
def prob_female_arts_25plus : ℝ := 0.45
def prob_female_business_25plus : ℝ := 0.2

def prob_male_science_less25 : ℝ := 1 - prob_male_science_25plus
def prob_male_arts_less25 : ℝ := 1 - prob_male_arts_25plus
def prob_male_business_less25 : ℝ := 1 - prob_male_business_25plus

def prob_female_science_less25 : ℝ := 1 - prob_female_science_25plus
def prob_female_arts_less25 : ℝ := 1 - prob_female_arts_25plus
def prob_female_business_less25 : ℝ := 1 - prob_female_business_25plus

def prob_science_less25 : ℝ := prob_male * prob_science * prob_male_science_less25 + prob_female * prob_science * prob_female_science_less25
def prob_arts_less25 : ℝ := prob_male * prob_arts * prob_male_arts_less25 + prob_female * prob_arts * prob_female_arts_less25
def prob_business_less25 : ℝ := prob_male * prob_business * prob_male_business_less25 + prob_female * prob_business * prob_female_business_less25

theorem prob_business_less25_correct :
    prob_business_less25 = 0.185 :=
by
  -- Theorem statement to be proved (proof omitted)
  sorry

end prob_business_less25_correct_l127_127738


namespace parametric_equations_curveC2_minimum_distance_M_to_curveC_l127_127104

noncomputable def curveC1_param (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sin α)

def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (3 * x, 2 * y)

theorem parametric_equations_curveC2 (θ : ℝ) :
  scaling_transform (Real.cos θ) (Real.sin θ) = (3 * Real.cos θ, 2 * Real.sin θ) :=
sorry

noncomputable def curveC (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin θ + ρ * Real.cos θ = 10

noncomputable def distance_to_curveC (θ : ℝ) : ℝ :=
  abs (3 * Real.cos θ + 4 * Real.sin θ - 10) / Real.sqrt 5

theorem minimum_distance_M_to_curveC : 
  ∀ θ, distance_to_curveC θ >= Real.sqrt 5 :=
sorry

end parametric_equations_curveC2_minimum_distance_M_to_curveC_l127_127104


namespace sangwoo_gave_away_notebooks_l127_127765

variables (n : ℕ)

theorem sangwoo_gave_away_notebooks
  (h1 : 12 - n + 34 - 3 * n = 30) :
  n = 4 :=
by
  sorry

end sangwoo_gave_away_notebooks_l127_127765


namespace range_of_a_l127_127600

def A := {x : ℝ | x * (4 - x) ≥ 3}
def B (a : ℝ) := {x : ℝ | x > a}

theorem range_of_a (a : ℝ) : (A ∩ B a = A) ↔ (a < 1) := by
  sorry

end range_of_a_l127_127600


namespace product_N_l127_127213

theorem product_N (A D D1 A1 : ℤ) (N : ℤ) 
  (h1 : D = A - N)
  (h2 : D1 = D + 7)
  (h3 : A1 = A - 2)
  (h4 : |D1 - A1| = 8) : 
  N = 1 → N = 17 → N * 17 = 17 :=
by
  sorry

end product_N_l127_127213


namespace prime_product_2002_l127_127613

theorem prime_product_2002 {a b c d : ℕ} (ha_prime : Prime a) (hb_prime : Prime b) (hc_prime : Prime c) (hd_prime : Prime d)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : a + c = d)
  (h2 : a * (a + b + c + d) = c * (d - b))
  (h3 : 1 + b * c + d = b * d) :
  a * b * c * d = 2002 := 
by 
  sorry

end prime_product_2002_l127_127613


namespace triangle_inequality_l127_127639

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : True :=
  sorry

end triangle_inequality_l127_127639


namespace Kolya_result_l127_127361

-- Define the list of numbers
def numbers := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

-- Defining the missing decimal point case
def mistaken_number := 15

-- Calculate the sum correctly and with a missing decimal point
noncomputable def correct_sum := numbers.sum
noncomputable def mistaken_sum := correct_sum + (mistaken_number - 1.5)

-- Prove the result is as expected
theorem Kolya_result : mistaken_sum = 27 := by
  sorry

end Kolya_result_l127_127361


namespace sourball_candies_division_l127_127757

theorem sourball_candies_division (N J L : ℕ) (total_candies : ℕ) (remaining_candies : ℕ) :
  N = 12 →
  J = N / 2 →
  L = J - 3 →
  total_candies = 30 →
  remaining_candies = total_candies - (N + J + L) →
  (remaining_candies / 3) = 3 :=
by 
  sorry

end sourball_candies_division_l127_127757


namespace marbles_lost_l127_127330

def initial_marbles := 8
def current_marbles := 6

theorem marbles_lost : initial_marbles - current_marbles = 2 :=
by
  sorry

end marbles_lost_l127_127330


namespace binomial_20_10_l127_127538

open Nat

theorem binomial_20_10 :
  (binomial 18 8 = 43758) →
  (binomial 18 9 = 48620) →
  (binomial 18 10 = 43758) →
  binomial 20 10 = 184756 :=
by
  intros h1 h2 h3
  sorry

end binomial_20_10_l127_127538


namespace factorize_polynomial_l127_127824

theorem factorize_polynomial (a b : ℝ) : 
  a^3 * b - 9 * a * b = a * b * (a + 3) * (a - 3) :=
by sorry

end factorize_polynomial_l127_127824


namespace smallest_four_digit_divisible_by_primes_l127_127083

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end smallest_four_digit_divisible_by_primes_l127_127083


namespace paul_crayons_l127_127450

def initial_crayons : ℝ := 479.0
def additional_crayons : ℝ := 134.0
def total_crayons : ℝ := initial_crayons + additional_crayons

theorem paul_crayons : total_crayons = 613.0 :=
by
  sorry

end paul_crayons_l127_127450


namespace find_time_eating_dinner_l127_127371

def total_flight_time : ℕ := 11 * 60 + 20
def time_reading : ℕ := 2 * 60
def time_watching_movies : ℕ := 4 * 60
def time_listening_radio : ℕ := 40
def time_playing_games : ℕ := 1 * 60 + 10
def time_nap : ℕ := 3 * 60

theorem find_time_eating_dinner : 
  total_flight_time - (time_reading + time_watching_movies + time_listening_radio + time_playing_games + time_nap) = 30 := 
by
  sorry

end find_time_eating_dinner_l127_127371


namespace find_s_l127_127805

variable (x t s : ℝ)

-- Conditions
#check (0.75 * x) / 60  -- Time for the first part of the trip
#check 0.25 * x  -- Distance for the remaining part of the trip
#check t - (0.75 * x) / 60  -- Time for the remaining part of the trip
#check 40 * t  -- Solving for x from average speed relation

-- Prove the value of s
theorem find_s (h1 : x = 40 * t) (h2 : s = (0.25 * x) / (t - (0.75 * x) / 60)) : s = 20 := by sorry

end find_s_l127_127805


namespace molar_mass_of_compound_l127_127271

variable (total_weight : ℝ) (num_moles : ℝ)

theorem molar_mass_of_compound (h1 : total_weight = 2352) (h2 : num_moles = 8) :
    total_weight / num_moles = 294 :=
by
  rw [h1, h2]
  norm_num

end molar_mass_of_compound_l127_127271


namespace expression_in_parentheses_l127_127417

theorem expression_in_parentheses (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) :
  ∃ expr : ℝ, xy * expr = -x^3 * y^2 ∧ expr = -x^2 * y :=
by
  sorry

end expression_in_parentheses_l127_127417


namespace factor_expression_l127_127688

variable (x : ℝ)

theorem factor_expression :
  (4 * x ^ 3 + 100 * x ^ 2 - 28) - (-9 * x ^ 3 + 2 * x ^ 2 - 28) = 13 * x ^ 2 * (x + 7) :=
by
  sorry

end factor_expression_l127_127688


namespace triangle_area_l127_127620

theorem triangle_area (a b c : ℝ)
    (h1 : Polynomial.eval a (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (h2 : Polynomial.eval b (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (h3 : Polynomial.eval c (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (sum_roots : a + b + c = 4)
    (sum_prod_roots : a * b + a * c + b * c = 5)
    (prod_roots : a * b * c = 1):
    Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) = 1 :=
  sorry

end triangle_area_l127_127620


namespace vectors_parallel_eq_l127_127269

-- Defining the problem
variables {m : ℝ}

-- Main statement
theorem vectors_parallel_eq (h : ∃ k : ℝ, (k ≠ 0) ∧ (k * 1 = m) ∧ (k * m = 2)) :
  m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
sorry

end vectors_parallel_eq_l127_127269


namespace binary_to_base5_conversion_l127_127817

theorem binary_to_base5_conversion : 
  let n := 45 in -- decimal conversion of 101101_2
    (n = 2^5 + 2^3 + 2^2 + 2^0) ∧ -- condition for binary to decimal
    -- condition for decimal to base-5
    (n % 5 = 0) ∧ ((n / 5) % 5 = 4) ∧ ((n / 5 / 5) % 5 = 1) 
    → n.to_nat_repr 5 = "140" :=
by
  sorry

end binary_to_base5_conversion_l127_127817


namespace arrange_bulbs_l127_127905

-- Define the conditions
def blue_bulbs : ℕ := 7
def red_bulbs : ℕ := 6
def white_bulbs : ℕ := 10

-- Calculate the binomial coefficients
def binom1 : ℕ := Nat.choose (blue_bulbs + red_bulbs) blue_bulbs
def binom2 : ℕ := Nat.choose (blue_bulbs + red_bulbs + 1) white_bulbs

-- Main theorem to prove the number of arrangements equals 1717716
theorem arrange_bulbs : binom1 * binom2 = 1717716 := sorry

end arrange_bulbs_l127_127905


namespace find_Sn_find_Tn_l127_127955

def Sn (n : ℕ) : ℕ := n^2 + n

def Tn (n : ℕ) : ℚ := (n : ℚ) / (n + 1)

section
variables {a₁ d : ℕ}

-- Given conditions
axiom S5 : 5 * a₁ + 10 * d = 30
axiom S10 : 10 * a₁ + 45 * d = 110

-- Problem statement 1
theorem find_Sn (n : ℕ) : Sn n = n^2 + n :=
sorry

-- Problem statement 2
theorem find_Tn (n : ℕ) : Tn n = (n : ℚ) / (n + 1) :=
sorry

end

end find_Sn_find_Tn_l127_127955


namespace distinct_four_digit_integers_with_digit_product_18_l127_127566

theorem distinct_four_digit_integers_with_digit_product_18 : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in digits.prod = 18) → 
  (finset.univ.filter (λ m, (let mdigits := [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10] in mdigits.prod = 18))).card = 36 :=
by
  sorry

end distinct_four_digit_integers_with_digit_product_18_l127_127566


namespace combined_tax_rate_35_58_l127_127746

noncomputable def combined_tax_rate (john_income : ℝ) (john_tax_rate : ℝ) (ingrid_income : ℝ) (ingrid_tax_rate : ℝ) : ℝ :=
  let john_tax := john_tax_rate * john_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let total_tax := john_tax + ingrid_tax
  let total_income := john_income + ingrid_income
  (total_tax / total_income) * 100

theorem combined_tax_rate_35_58
  (john_income : ℝ) (john_tax_rate : ℝ) (ingrid_income : ℝ) (ingrid_tax_rate : ℝ)
  (h1 : john_income = 57000) (h2 : john_tax_rate = 0.3)
  (h3 : ingrid_income = 72000) (h4 : ingrid_tax_rate = 0.4) :
  combined_tax_rate john_income john_tax_rate ingrid_income ingrid_tax_rate = 35.58 :=
by
  simp [combined_tax_rate, h1, h2, h3, h4]
  sorry

end combined_tax_rate_35_58_l127_127746


namespace find_x_plus_inv_x_l127_127256

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + (1/x)^3 = 110) : x + (1/x) = 5 :=
sorry

end find_x_plus_inv_x_l127_127256


namespace remainder_18_l127_127642

theorem remainder_18 (x : ℤ) (k : ℤ) (h : x = 62 * k + 7) :
  (x + 11) % 31 = 18 :=
by
  sorry

end remainder_18_l127_127642


namespace forces_arithmetic_progression_ratio_l127_127900

theorem forces_arithmetic_progression_ratio 
  (a d : ℝ) 
  (h1 : ∀ (x y z : ℝ), IsArithmeticProgression x y z → x = a ∧ y = a + d ∧ z = a + 2d)
  (h2 : a^2 + (a + d)^2 = (a + 2d)^2)
  (h3 : a ≠ 0 ∧ d ≠ 0) :
  d / a = 1 / 3 :=
by
  sorry

end forces_arithmetic_progression_ratio_l127_127900


namespace units_digit_quotient_l127_127162

theorem units_digit_quotient (n : ℕ) :
  (2^1993 + 3^1993) % 5 = 0 →
  ((2^1993 + 3^1993) / 5) % 10 = 3 := by
  sorry

end units_digit_quotient_l127_127162


namespace certain_event_is_A_l127_127180

def conditions (option_A option_B option_C option_D : Prop) : Prop :=
  option_A ∧ ¬option_B ∧ ¬option_C ∧ ¬option_D

theorem certain_event_is_A 
  (option_A option_B option_C option_D : Prop)
  (hconditions : conditions option_A option_B option_C option_D) : 
  ∀ e, (e = option_A) := 
by
  sorry

end certain_event_is_A_l127_127180


namespace smallest_four_digit_divisible_by_2_3_5_7_11_l127_127081

theorem smallest_four_digit_divisible_by_2_3_5_7_11 :
  ∃ n : ℕ, n = 2310 ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ (n % 11 = 0) ∧ 1000 ≤ n ∧ n < 10000 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_2_3_5_7_11_l127_127081


namespace brick_weight_l127_127237

theorem brick_weight (b s : ℕ) (h1 : 5 * b = 4 * s) (h2 : 2 * s = 80) : b = 32 :=
by {
  sorry
}

end brick_weight_l127_127237


namespace number_of_subsets_l127_127525

open Finset

/-- Given that {2, 3} ⊆ Y ⊆ {1, 2, 3, 4, 5, 6}, prove that the number of such subsets Y is 16. -/
theorem number_of_subsets (Y : Finset ℕ) (h : {2, 3} ⊆ Y ∧ Y ⊆ {1, 2, 3, 4, 5, 6}) :
    (univ.filter (λ X : Finset ℕ, {2, 3} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5, 6})).card = 16 := by
  sorry

end number_of_subsets_l127_127525


namespace total_jumps_l127_127461

def taehyung_jumps_per_day : ℕ := 56
def taehyung_days : ℕ := 3
def namjoon_jumps_per_day : ℕ := 35
def namjoon_days : ℕ := 4

theorem total_jumps : taehyung_jumps_per_day * taehyung_days + namjoon_jumps_per_day * namjoon_days = 308 :=
by
  sorry

end total_jumps_l127_127461


namespace factorization_of_polynomial_l127_127825

theorem factorization_of_polynomial : ∀ x : ℝ, x^2 - x - 42 = (x + 6) * (x - 7) :=
by
  sorry

end factorization_of_polynomial_l127_127825


namespace sin_half_alpha_l127_127711

theorem sin_half_alpha (α : ℝ) (h_cos : Real.cos α = -2/3) (h_range : π < α ∧ α < 3 * π / 2) :
  Real.sin (α / 2) = Real.sqrt 30 / 6 :=
by
  sorry

end sin_half_alpha_l127_127711


namespace problem1_problem2_problem3_l127_127332

-- Problem 1
theorem problem1 (x : ℝ) (h : x^2 + x - 2 = 0) : x^2 + x + 2023 = 2025 := 
  sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h : a + b = 5) : 2 * (a + b) - 4 * a - 4 * b + 21 = 11 := 
  sorry

-- Problem 3
theorem problem3 (a b : ℝ) (h1 : a^2 + 3 * a * b = 20) (h2 : b^2 + 5 * a * b = 8) : 2 * a^2 - b^2 + a * b = 32 := 
  sorry

end problem1_problem2_problem3_l127_127332


namespace number_of_houses_with_neither_feature_l127_127307

variable (T G P B : ℕ)

theorem number_of_houses_with_neither_feature 
  (hT : T = 90)
  (hG : G = 50)
  (hP : P = 40)
  (hB : B = 35) : 
  T - (G + P - B) = 35 := 
    by
      sorry

end number_of_houses_with_neither_feature_l127_127307


namespace james_bike_ride_l127_127297

variable {D P : ℝ}

theorem james_bike_ride :
  (∃ D P, 3 * D + (18 + 18 * 0.25) = 55.5 ∧ (18 = D * (1 + P / 100))) → P = 20 := by
  sorry

end james_bike_ride_l127_127297


namespace paige_finished_problems_at_school_l127_127761

-- Definitions based on conditions
def math_problems : ℕ := 43
def science_problems : ℕ := 12
def total_problems : ℕ := math_problems + science_problems
def problems_left : ℕ := 11

-- The main theorem we need to prove
theorem paige_finished_problems_at_school : total_problems - problems_left = 44 := by
  sorry

end paige_finished_problems_at_school_l127_127761


namespace scientific_notation_43300000_l127_127322

theorem scientific_notation_43300000 : 43300000 = 4.33 * 10^7 :=
by
  sorry

end scientific_notation_43300000_l127_127322


namespace quadratic_distinct_roots_k_range_l127_127124

theorem quadratic_distinct_roots_k_range (k : ℝ) :
  (k - 1) * x^2 + 2 * x - 2 = 0 ∧ 
  ∀ Δ, Δ = 2^2 - 4*(k-1)*(-2) ∧ Δ > 0 ∧ (k ≠ 1) ↔ k > 1/2 ∧ k ≠ 1 :=
by
  sorry

end quadratic_distinct_roots_k_range_l127_127124


namespace repeating_decimal_product_l127_127702

def repeating_decimal_12 := 12 / 99
def repeating_decimal_34 := 34 / 99

theorem repeating_decimal_product : (repeating_decimal_12 * repeating_decimal_34) = 136 / 3267 := by
  sorry

end repeating_decimal_product_l127_127702


namespace race_distance_l127_127741

theorem race_distance (D : ℝ) (h1 : (D / 36) * 45 = D + 20) : D = 80 :=
by
  sorry

end race_distance_l127_127741


namespace gcd_84_294_315_l127_127026

def gcd_3_integers : ℕ := Nat.gcd (Nat.gcd 84 294) 315

theorem gcd_84_294_315 : gcd_3_integers = 21 :=
by
  sorry

end gcd_84_294_315_l127_127026


namespace ratio_AB_AD_l127_127457

theorem ratio_AB_AD (a x y : ℝ) (h1 : 0.3 * a^2 = 0.7 * x * y) (h2 : y = a / 10) : x / y = 43 :=
by
  sorry

end ratio_AB_AD_l127_127457


namespace largest_multiple_l127_127335

theorem largest_multiple (a b limit : ℕ) (ha : a = 3) (hb : b = 5) (h_limit : limit = 800) : 
  ∃ (n : ℕ), (lcm a b) * n < limit ∧ (lcm a b) * (n + 1) ≥ limit ∧ (lcm a b) * n = 795 := 
by 
  sorry

end largest_multiple_l127_127335


namespace problem_solution_l127_127714

theorem problem_solution (k x1 x2 y1 y2 : ℝ) 
  (h₁ : k ≠ 0) 
  (h₂ : y1 = k * x1) 
  (h₃ : y1 = -5 / x1) 
  (h₄ : y2 = k * x2) 
  (h₅ : y2 = -5 / x2) 
  (h₆ : x1 = -x2) 
  (h₇ : y1 = -y2) : 
  x1 * y2 - 3 * x2 * y1 = 10 := 
sorry

end problem_solution_l127_127714


namespace champagne_bottles_needed_l127_127209

-- Define the initial conditions of the problem
def num_guests : ℕ := 120
def glasses_per_guest : ℕ := 2
def servings_per_bottle : ℕ := 6

-- The statement we need to prove
theorem champagne_bottles_needed : 
  (num_guests * glasses_per_guest) / servings_per_bottle = 40 := 
by
  sorry

end champagne_bottles_needed_l127_127209


namespace bob_better_than_half_chance_l127_127930

noncomputable def bob_guesses_correctly (x y : ℝ) (hx : x < y) : Prop :=
  ∃ (T : ℝ), ∀ (num_told : ℝ), ((num_told < T ∧ num_told = x) ∨ (num_told ≥ T ∧ num_told = y))

theorem bob_better_than_half_chance (x y : ℝ) (hx : x < y) :
  ∃ (T : ℝ), ∀ (num_told : ℝ), bob_guesses_correctly x y hx →
  (0.5 < probability_of_guessing_correctly x y T) :=
sorry

end bob_better_than_half_chance_l127_127930


namespace combined_share_of_A_and_C_l127_127666

-- Definitions based on the conditions
def total_money : Float := 15800
def charity_investment : Float := 0.10 * total_money
def savings_investment : Float := 0.08 * total_money
def remaining_money : Float := total_money - charity_investment - savings_investment

def ratio_A : Nat := 5
def ratio_B : Nat := 9
def ratio_C : Nat := 6
def ratio_D : Nat := 5
def sum_of_ratios : Nat := ratio_A + ratio_B + ratio_C + ratio_D

def share_A : Float := (ratio_A.toFloat / sum_of_ratios.toFloat) * remaining_money
def share_C : Float := (ratio_C.toFloat / sum_of_ratios.toFloat) * remaining_money
def combined_share_A_C : Float := share_A + share_C

-- Statement to be proven
theorem combined_share_of_A_and_C : combined_share_A_C = 5700.64 := by
  sorry

end combined_share_of_A_and_C_l127_127666


namespace unique_x1_sequence_l127_127710

open Nat

theorem unique_x1_sequence (x1 : ℝ) (x : ℕ → ℝ)
  (h₀ : x 1 = x1)
  (h₁ : ∀ n, x (n + 1) = x n * (x n + 1 / (n + 1))) :
  (∃! x1, (0 < x1 ∧ x1 < 1) ∧ 
   (∀ n, 0 < x n ∧ x n < x (n + 1) ∧ x (n + 1) < 1)) := sorry

end unique_x1_sequence_l127_127710


namespace min_value_of_f_l127_127394

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - 2 * x + 16) / (2 * x - 1)

theorem min_value_of_f :
  ∃ x : ℝ, x ≥ 1 ∧ f x = 9 ∧ (∀ y : ℝ, y ≥ 1 → f y ≥ 9) :=
by { sorry }

end min_value_of_f_l127_127394


namespace molecular_weight_AlOH3_l127_127937

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

theorem molecular_weight_AlOH3 :
  (atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H) = 78.01 :=
by
  sorry

end molecular_weight_AlOH3_l127_127937


namespace volume_of_inscribed_tetrahedron_l127_127312

theorem volume_of_inscribed_tetrahedron (r h : ℝ) (V : ℝ) (tetrahedron_inscribed : Prop) 
  (cylinder_condition : π * r^2 * h = 1) 
  (inscribed : tetrahedron_inscribed → True) : 
  V ≤ 2 / (3 * π) :=
sorry

end volume_of_inscribed_tetrahedron_l127_127312


namespace relationship_between_a_and_b_l127_127280

theorem relationship_between_a_and_b : 
  ∀ (a b : ℝ), (∀ x y : ℝ, (x-a)^2 + (y-b)^2 = b^2 + 1 → (x+1)^2 + (y+1)^2 = 4 → (2 + 2*a)*x + (2 + 2*b)*y - a^2 - 1 = 0) → a^2 + 2*a + 2*b + 5 = 0 :=
by
  intros a b hyp
  sorry

end relationship_between_a_and_b_l127_127280


namespace sum_common_ratios_l127_127004

variable (k p r : ℝ)
variable (hp : p ≠ r)

theorem sum_common_ratios (h : k * p ^ 2 - k * r ^ 2 = 2 * (k * p - k * r)) : 
  p + r = 2 := by
  have hk : k ≠ 0 := sorry -- From the nonconstancy condition
  sorry

end sum_common_ratios_l127_127004


namespace train_speed_l127_127808

noncomputable def train_length : ℝ := 2500
noncomputable def time_to_cross_pole : ℝ := 35

noncomputable def speed_in_kmph (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed :
  speed_in_kmph train_length time_to_cross_pole = 257.14 := by
  sorry

end train_speed_l127_127808


namespace equations_solution_l127_127158

-- Definition of the conditions
def equation1 := ∀ x : ℝ, x^2 - 2 * x - 3 = 0 -> (x = 3 ∨ x = -1)
def equation2 := ∀ x : ℝ, x * (x - 2) + x - 2 = 0 -> (x = -1 ∨ x = 2)

-- The main statement combining both problems
theorem equations_solution :
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 -> (x = 3 ∨ x = -1)) ∧
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 -> (x = -1 ∨ x = 2)) := by
  sorry

end equations_solution_l127_127158


namespace pyramid_volume_eq_l127_127218

noncomputable def volume_of_pyramid (base_length1 base_length2 height : ℝ) : ℝ :=
  (1 / 3) * base_length1 * base_length2 * height

theorem pyramid_volume_eq (base_length1 base_length2 height : ℝ) (h1 : base_length1 = 1) (h2 : base_length2 = 2) (h3 : height = 1) :
  volume_of_pyramid base_length1 base_length2 height = 2 / 3 := by
  sorry

end pyramid_volume_eq_l127_127218


namespace defect_probability_l127_127929

variable (A : Event)
variable (H1 H2 H3 : Event)
variable (prod_first prod_second prod_third : ℝ)
variable (def_first def_second def_third : ℝ)

axiom H1_prod: prod_first = 3 * prod_second
axiom H2_prod: prod_third = prod_second / 2
axiom def_prob_first : def_first = 0.02
axiom def_prob_second : def_second = 0.03
axiom def_prob_third : def_third = 0.04

theorem defect_probability :
  let total_prod := 3 * prod_second + prod_second + prod_second / 2 in
  let P_H1 := (3 * prod_second) / total_prod in
  let P_H2 := prod_second / total_prod in
  let P_H3 := (prod_second / 2) / total_prod in
  let P_A := def_first * P_H1 + def_second * P_H2 + def_third * P_H3 in
  P_A = 0.024 := by
  sorry

end defect_probability_l127_127929


namespace pyarelal_loss_l127_127185

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (ashok_ratio pyarelal_ratio : ℝ)
  (h1 : ashok_ratio = 1/9) (h2 : pyarelal_ratio = 1)
  (h3 : total_loss = 2000) : (pyarelal_ratio / (ashok_ratio + pyarelal_ratio)) * total_loss = 1800 :=
by
  sorry

end pyarelal_loss_l127_127185


namespace find_a1_of_geom_series_l127_127901

noncomputable def geom_series_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem find_a1_of_geom_series (a₁ q : ℝ) (S : ℕ → ℝ)
  (h1 : S 6 = 9 * S 3)
  (h2 : S 5 = 62)
  (neq1 : q ≠ 1)
  (neqm1 : q ≠ -1) :
  a₁ = 2 :=
by
  have eq1 : S 6 = geom_series_sum a₁ q 6 := sorry
  have eq2 : S 3 = geom_series_sum a₁ q 3 := sorry
  have eq3 : S 5 = geom_series_sum a₁ q 5 := sorry
  sorry

end find_a1_of_geom_series_l127_127901


namespace greatest_possible_xy_value_l127_127057

-- Define the conditions
variables (a b c d x y : ℕ)
variables (h1 : a < b) (h2 : b < c) (h3 : c < d)
variables (sums : Finset ℕ) (hsums : sums = {189, 320, 287, 234, x, y})

-- Define the goal statement to prove
theorem greatest_possible_xy_value : x + y = 791 :=
sorry

end greatest_possible_xy_value_l127_127057


namespace root_in_interval_l127_127650

def f (x : ℝ) : ℝ := x^3 + 5 * x^2 - 3 * x + 1

theorem root_in_interval : ∃ A B : ℤ, B = A + 1 ∧ (∃ ξ : ℝ, f ξ = 0 ∧ (A : ℝ) < ξ ∧ ξ < (B : ℝ)) ∧ A = -6 ∧ B = -5 :=
by
  sorry

end root_in_interval_l127_127650


namespace digit_150_of_5_over_37_l127_127784

theorem digit_150_of_5_over_37 : (decimal_digit_at 150 (5 / 37)) = 5 :=
by
  sorry

end digit_150_of_5_over_37_l127_127784


namespace solution_set_of_inequality_l127_127167

variable (a b x : ℝ)
variable (h1 : ∀ x, ax + b > 0 ↔ 1 < x)

theorem solution_set_of_inequality : ∀ x, (ax + b) * (x - 2) < 0 ↔ (1 < x ∧ x < 2) :=
by sorry

end solution_set_of_inequality_l127_127167


namespace total_employees_with_advanced_degrees_l127_127867

theorem total_employees_with_advanced_degrees 
  (total_employees : ℕ) 
  (num_females : ℕ) 
  (num_males_college_only : ℕ) 
  (num_females_advanced_degrees : ℕ)
  (h1 : total_employees = 180)
  (h2 : num_females = 110)
  (h3 : num_males_college_only = 35)
  (h4 : num_females_advanced_degrees = 55) :
  ∃ num_employees_advanced_degrees : ℕ, num_employees_advanced_degrees = 90 :=
by
  have num_males := total_employees - num_females
  have num_males_advanced_degrees := num_males - num_males_college_only
  have num_employees_advanced_degrees := num_males_advanced_degrees + num_females_advanced_degrees
  use num_employees_advanced_degrees
  sorry

end total_employees_with_advanced_degrees_l127_127867


namespace february_first_day_of_week_l127_127419

theorem february_first_day_of_week 
  (feb13_is_wednesday : ∃ day, day = 13 ∧ day_of_week = "Wednesday") :
  ∃ day, day = 1 ∧ day_of_week = "Friday" :=
sorry

end february_first_day_of_week_l127_127419


namespace maria_workday_end_l127_127821

def time_in_minutes (h : ℕ) (m : ℕ) : ℕ := h * 60 + m

def start_time : ℕ := time_in_minutes 7 25
def lunch_break : ℕ := 45
def noon : ℕ := time_in_minutes 12 0
def work_hours : ℕ := 8 * 60
def end_time : ℕ := time_in_minutes 16 10

theorem maria_workday_end : start_time + (noon - start_time) + lunch_break + (work_hours - (noon - start_time)) = end_time := by
  sorry

end maria_workday_end_l127_127821


namespace prob_difference_l127_127777

noncomputable def probability_same_color (total_marbs : ℕ) (num_red : ℕ) (num_black : ℕ) (num_white : ℕ) : ℚ :=
  let ways_same_red := num_red.choose 2
  let ways_same_black := num_black.choose 2
  let ways_red_white := num_red
  let ways_black_white := num_black
  let num_ways_same := ways_same_red + ways_same_black + ways_red_white + ways_black_white
  let total_ways := total_marbs.choose 2
  num_ways_same / total_ways

noncomputable def probability_different_color (total_marbs : ℕ) (num_red : ℕ) (num_black : ℕ) : ℚ :=
  let ways_diff := num_red * num_black
  let total_ways := total_marbs.choose 2
  ways_diff / total_ways

theorem prob_difference (num_red num_black num_white : ℕ) (h_red : num_red = 1500) (h_black : num_black = 1500) (h_white : num_white = 1) :
  let total_marbs := num_red + num_black + num_white
  |probability_same_color total_marbs num_red num_black num_white - probability_different_color total_marbs num_red num_black| = 1/3 :=
by
  sorry

end prob_difference_l127_127777


namespace number_of_two_point_safeties_l127_127737

variables (f g s : ℕ)

theorem number_of_two_point_safeties (h1 : 4 * f = 6 * g) 
                                    (h2 : s = g + 2) 
                                    (h3 : 4 * f + 3 * g + 2 * s = 50) : 
                                    s = 6 := 
by sorry

end number_of_two_point_safeties_l127_127737


namespace system_of_equations_solution_l127_127889

theorem system_of_equations_solution (x y z : ℕ) :
  x + y + z = 6 ∧ xy + yz + zx = 11 ∧ xyz = 6 ↔
  (x, y, z) = (1, 2, 3) ∨ (x, y, z) = (1, 3, 2) ∨ 
  (x, y, z) = (2, 1, 3) ∨ (x, y, z) = (2, 3, 1) ∨ 
  (x, y, z) = (3, 1, 2) ∨ (x, y, z) = (3, 2, 1) := by
  sorry

end system_of_equations_solution_l127_127889


namespace distance_between_foci_l127_127520

theorem distance_between_foci (a b : ℝ) (h₁ : a^2 = 18) (h₂ : b^2 = 2) :
  2 * (Real.sqrt (a^2 + b^2)) = 4 * Real.sqrt 5 :=
by
  sorry

end distance_between_foci_l127_127520


namespace value_of_B_minus_3_plus_A_l127_127278

theorem value_of_B_minus_3_plus_A (A B : ℝ) (h : A + B = 5) : B - 3 + A = 2 :=
by 
  sorry

end value_of_B_minus_3_plus_A_l127_127278


namespace min_ab_square_is_four_l127_127693

noncomputable def min_ab_square : Prop :=
  ∃ a b : ℝ, (a^2 + b^2 = 4 ∧ ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0)

theorem min_ab_square_is_four : min_ab_square :=
  sorry

end min_ab_square_is_four_l127_127693


namespace probability_of_common_books_l127_127447

theorem probability_of_common_books (total_books : ℕ) (books_to_select : ℕ) :
  total_books = 12 → books_to_select = 6 →
  let total_ways := Nat.choose 12 6 * Nat.choose 12 6 in
  let successful_ways := Nat.choose 12 3 * Nat.choose 9 3 * Nat.choose 9 3 in
  (successful_ways : ℚ) / total_ways = 220 / 153 :=
by
  intros ht12 hs6
  let total_ways := Nat.choose 12 6 * Nat.choose 12 6
  let successful_ways := Nat.choose 12 3 * Nat.choose 9 3 * Nat.choose 9 3
  have htotal_ways : total_ways = 924 * 924 := by sorry
  have hsuccessful_ways : successful_ways = 220 * 84 * 84 := by sorry
  rw [ht12, hs6, htotal_ways, hsuccessful_ways]
  norm_num
  exact @eq.refl(ℚ) (220 / 153)

end probability_of_common_books_l127_127447


namespace correct_answer_l127_127810

-- Define the problem conditions and question
def equation (y : ℤ) : Prop := y + 2 = -3

-- Prove that the correct answer is y = -5
theorem correct_answer : ∀ y : ℤ, equation y → y = -5 :=
by
  intros y h
  unfold equation at h
  linarith

end correct_answer_l127_127810


namespace max_consecutive_integers_sum_48_l127_127910

-- Define the sum of consecutive integers
def sum_consecutive_integers (a N : ℤ) : ℤ :=
  (N * (2 * a + N - 1)) / 2

-- Define the main theorem
theorem max_consecutive_integers_sum_48 : 
  ∃ N a : ℤ, sum_consecutive_integers a N = 48 ∧ (∀ N' : ℤ, ((N' * (2 * a + N' - 1)) / 2 = 48) → N' ≤ N) :=
sorry

end max_consecutive_integers_sum_48_l127_127910


namespace work_days_l127_127344

theorem work_days (Dx Dy : ℝ) (H1 : Dy = 45) (H2 : 8 / Dx + 36 / Dy = 1) : Dx = 40 :=
by
  sorry

end work_days_l127_127344


namespace f_f_2_equals_l127_127731

def f (x : ℕ) : ℕ := 4 * x ^ 3 - 6 * x + 2

theorem f_f_2_equals :
  f (f 2) = 42462 :=
by
  sorry

end f_f_2_equals_l127_127731


namespace sin_double_angle_values_l127_127272

theorem sin_double_angle_values (α : ℝ) (hα : 0 < α ∧ α < π) (h : 3 * (Real.cos α)^2 = Real.sin ((π / 4) - α)) :
  Real.sin (2 * α) = 1 ∨ Real.sin (2 * α) = -17 / 18 :=
by
  sorry

end sin_double_angle_values_l127_127272


namespace min_value_of_f_l127_127527

noncomputable def f (x : ℝ) : ℝ := (real.exp x - 1) ^ 2 + (real.exp (-x) - 1) ^ 2

theorem min_value_of_f : ∃ x : ℝ, f x = 0 := 
begin
  use 0,
  -- Here we should show that f(0) = 0
  simp [f, real.exp_zero],
  norm_num,
end

end min_value_of_f_l127_127527


namespace value_of_b_l127_127128

variable (a b c y1 y2 : ℝ)

def equation1 := (y1 = 4 * a + 2 * b + c)
def equation2 := (y2 = 4 * a - 2 * b + c)
def difference := (y1 - y2 = 8)

theorem value_of_b 
  (h1 : equation1 a b c y1)
  (h2 : equation2 a b c y2)
  (h3 : difference y1 y2) : 
  b = 2 := 
by 
  sorry

end value_of_b_l127_127128


namespace negation_proposition_correct_l127_127384

theorem negation_proposition_correct : 
  (∀ x : ℝ, 0 < x → x + 4 / x ≥ 4) :=
by
  intro x hx
  sorry

end negation_proposition_correct_l127_127384


namespace cubed_identity_l127_127728

theorem cubed_identity (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := 
by 
  sorry

end cubed_identity_l127_127728


namespace reduction_in_jury_running_time_l127_127816

def week1_miles : ℕ := 2
def week2_miles : ℕ := 2 * week1_miles + 3
def week3_miles : ℕ := (9 * week2_miles) / 7
def week4_miles : ℕ := 4

theorem reduction_in_jury_running_time : week3_miles - week4_miles = 5 :=
by
  -- sorry specifies the proof is skipped
  sorry

end reduction_in_jury_running_time_l127_127816


namespace payment_difference_correct_l127_127007

noncomputable def initial_debt : ℝ := 12000

noncomputable def planA_interest_rate : ℝ := 0.08
noncomputable def planA_compounding_periods : ℕ := 2

noncomputable def planB_interest_rate : ℝ := 0.08

noncomputable def planA_payment_years : ℕ := 4
noncomputable def planA_remaining_years : ℕ := 4

noncomputable def planB_years : ℕ := 8

-- Amount accrued in Plan A after 4 years
noncomputable def planA_amount_after_first_period : ℝ :=
  initial_debt * (1 + planA_interest_rate / planA_compounding_periods) ^ (planA_compounding_periods * planA_payment_years)

-- Amount paid at the end of first period (two-thirds of total)
noncomputable def planA_first_payment : ℝ :=
  (2/3) * planA_amount_after_first_period

-- Remaining debt after first payment
noncomputable def planA_remaining_debt : ℝ :=
  planA_amount_after_first_period - planA_first_payment

-- Amount accrued on remaining debt after 8 years (second 4-year period)
noncomputable def planA_second_payment : ℝ :=
  planA_remaining_debt * (1 + planA_interest_rate / planA_compounding_periods) ^ (planA_compounding_periods * planA_remaining_years)

-- Total payment under Plan A
noncomputable def total_payment_planA : ℝ :=
  planA_first_payment + planA_second_payment

-- Total payment under Plan B
noncomputable def total_payment_planB : ℝ :=
  initial_debt * (1 + planB_interest_rate * planB_years)

-- Positive difference between payments
noncomputable def payment_difference : ℝ :=
  total_payment_planB - total_payment_planA

theorem payment_difference_correct :
  payment_difference = 458.52 :=
by
  sorry

end payment_difference_correct_l127_127007


namespace ratio_seniors_to_juniors_l127_127214

variable (j s : ℕ)

-- Condition: \(\frac{3}{7}\) of the juniors participated is equal to \(\frac{6}{7}\) of the seniors participated
def participation_condition (j s : ℕ) : Prop :=
  3 * j = 6 * s

-- Theorem to be proved: the ratio of seniors to juniors is \( \frac{1}{2} \)
theorem ratio_seniors_to_juniors (j s : ℕ) (h : participation_condition j s) : s / j = 1 / 2 :=
  sorry

end ratio_seniors_to_juniors_l127_127214


namespace race_distance_l127_127740

theorem race_distance (D : ℝ) (h1 : (D / 36) * 45 = D + 20) : D = 80 :=
by
  sorry

end race_distance_l127_127740


namespace digit_assignment_count_is_correct_l127_127136

open Finset

/-
In how many different ways can the digits 0, 1, 2, ..., 9 be placed into the following scheme so that a correct addition is obtained?

A
BC
DEF
———
CHJK
-/
noncomputable def count_valid_digit_assignments : ℕ := 60

theorem digit_assignment_count_is_correct :
  ∃ (A B C D E F H J K : ℕ), 
    {A, B, C, D, E, F, H, J, K}.card = 9 ∧
    A ∈ (range 10) ∧ B ∈ (range 10) ∧ C ∈ (range 10) ∧ D ∈ (range 10) ∧
    E ∈ (range 10) ∧ F ∈ (range 10) ∧ H ∈ (range 10) ∧ J ∈ (range 10) ∧ K ∈ (range 10) ∧
    0 < A ∧ 0 < B ∧ 0 < D ∧
    1000 * C + 100 * H + 10 * J + K = A + 10 * B + C + 100 * D + 10 * E + F ∧
    count_valid_digit_assignments = 60 :=
by
  sorry

end digit_assignment_count_is_correct_l127_127136


namespace volume_of_cube_surface_area_times_l127_127795

theorem volume_of_cube_surface_area_times (V1 : ℝ) (hV1 : V1 = 8) : 
  ∃ V2, V2 = 24 * Real.sqrt 3 :=
sorry

end volume_of_cube_surface_area_times_l127_127795


namespace Kayla_score_fifth_level_l127_127749

theorem Kayla_score_fifth_level :
  ∃ (a b c d e f : ℕ),
  a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 8 ∧ f = 17 ∧
  (b - a = 1) ∧ (c - b = 2) ∧ (d - c = 3) ∧ (e - d = 4) ∧ (f - e = 5) ∧ e = 12 :=
sorry

end Kayla_score_fifth_level_l127_127749


namespace birds_joined_l127_127047

variable (initialBirds : ℕ) (totalBirds : ℕ)

theorem birds_joined (h1 : initialBirds = 2) (h2 : totalBirds = 6) : (totalBirds - initialBirds) = 4 :=
by
  sorry

end birds_joined_l127_127047


namespace calculate_expression_l127_127093

variable {x y : ℝ}

theorem calculate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 3 / y) :
  (3 * x - 3 / x) * (3 * y + 3 / y) = 9 * x^2 - y^2 :=
by
  sorry

end calculate_expression_l127_127093


namespace teresa_marks_ratio_l127_127770

theorem teresa_marks_ratio (science music social_studies total_marks physics_ratio : ℝ) 
  (h_science : science = 70)
  (h_music : music = 80)
  (h_social_studies : social_studies = 85)
  (h_total_marks : total_marks = 275)
  (h_physics : science + music + social_studies + physics_ratio * music = total_marks) :
  physics_ratio = 1 / 2 :=
by
  subst h_science
  subst h_music
  subst h_social_studies
  subst h_total_marks
  have : 70 + 80 + 85 + physics_ratio * 80 = 275 := h_physics
  linarith

end teresa_marks_ratio_l127_127770


namespace smallest_x_gx_eq_g2023_l127_127006

def g (x : ℝ) : ℝ :=
  if 2 ≤ x ∧ x ≤ 4 then 2 - |x - 3| else sorry

theorem smallest_x_gx_eq_g2023
  (h1 : ∀ x > 0, g (4 * x) = 4 * g x)
  (h2 : ∀ x, 2 ≤ x ∧ x ≤ 4 → g x = 2 - |x - 3|) :
  ∃ x : ℝ, g x = g 2023 ∧ ∀ y : ℝ, g y = g 2023 → x ≤ y := 
sorry

end smallest_x_gx_eq_g2023_l127_127006


namespace area_of_figure_l127_127431
-- Import necessary libraries

-- Define the conditions as functions/constants
def length_left : ℕ := 7
def width_top : ℕ := 6
def height_middle : ℕ := 3
def width_middle : ℕ := 4
def height_right : ℕ := 5
def width_right : ℕ := 5

-- State the problem as a theorem
theorem area_of_figure : 
  (length_left * width_top) + 
  (width_middle * height_middle) + 
  (width_right * height_right) = 79 := 
  by
  sorry

end area_of_figure_l127_127431


namespace compute_expression_l127_127217

theorem compute_expression : 6^2 + 2 * 5 - 4^2 = 30 :=
by sorry

end compute_expression_l127_127217


namespace seventh_term_correct_l127_127706

noncomputable def seventh_term_geometric_sequence (a r : ℝ) (h1 : a = 5) (h2 : a * r = 1/5) : ℝ :=
  a * r ^ 6

theorem seventh_term_correct :
  seventh_term_geometric_sequence 5 (1/25) (by rfl) (by norm_num) = 1 / 48828125 :=
  by
    unfold seventh_term_geometric_sequence
    sorry

end seventh_term_correct_l127_127706


namespace boy_to_total_ratio_l127_127582

-- Problem Definitions
variables (b g : ℕ) -- number of boys and number of girls

-- Hypothesis: The probability of choosing a boy is (4/5) the probability of choosing a girl
def probability_boy := b / (b + g : ℕ)
def probability_girl := g / (b + g : ℕ)

theorem boy_to_total_ratio (h : probability_boy b g = (4 / 5) * probability_girl b g) : 
  b / (b + g : ℕ) = 4 / 9 :=
sorry

end boy_to_total_ratio_l127_127582


namespace couscous_problem_l127_127350

def total_couscous (S1 S2 S3 : ℕ) : ℕ :=
  S1 + S2 + S3

def couscous_per_dish (total : ℕ) (dishes : ℕ) : ℕ :=
  total / dishes

theorem couscous_problem 
  (S1 S2 S3 : ℕ) (dishes : ℕ) 
  (h1 : S1 = 7) (h2 : S2 = 13) (h3 : S3 = 45) (h4 : dishes = 13) :
  couscous_per_dish (total_couscous S1 S2 S3) dishes = 5 := by  
  sorry

end couscous_problem_l127_127350


namespace MrSami_sold_20_shares_of_stock_x_l127_127037

theorem MrSami_sold_20_shares_of_stock_x
    (shares_v : ℕ := 68)
    (shares_w : ℕ := 112)
    (shares_x : ℕ := 56)
    (shares_y : ℕ := 94)
    (shares_z : ℕ := 45)
    (additional_shares_y : ℕ := 23)
    (increase_in_range : ℕ := 14)
    : (shares_x - (shares_y + additional_shares_y - ((shares_w - shares_z + increase_in_range) - shares_y - additional_shares_y)) = 20) :=
by
  sorry

end MrSami_sold_20_shares_of_stock_x_l127_127037


namespace cubed_identity_l127_127729

theorem cubed_identity (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := 
by 
  sorry

end cubed_identity_l127_127729


namespace sequence_existence_l127_127075

theorem sequence_existence (n : ℕ) : 
  (∃ (x : ℕ → ℤ), 
    (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n ∧ i + j ≤ n ∧ ((x i - x j) % 3 = 0) → (x (i + j) + x i + x j + 1) % 3 = 0)) ↔ (n = 8) := 
by 
  sorry

end sequence_existence_l127_127075


namespace vector_addition_parallel_l127_127966

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem vector_addition_parallel:
  ∀ x : ℝ, parallel (2, 1) (x, -2) → a + b x = ((-2 : ℝ), -1) :=
by
  intros x h
  sorry

end vector_addition_parallel_l127_127966


namespace cats_in_house_l127_127739

-- Define the conditions
def total_cats (C : ℕ) : Prop :=
  let num_white_cats := 2
  let num_black_cats := C / 4
  let num_grey_cats := 10
  C = num_white_cats + num_black_cats + num_grey_cats

-- State the theorem
theorem cats_in_house : ∃ C : ℕ, total_cats C ∧ C = 16 := 
by
  sorry

end cats_in_house_l127_127739


namespace false_proposition_A_l127_127390

theorem false_proposition_A 
  (a b : ℝ)
  (root1_eq_1 : ∀ x, x^2 + a * x + b = 0 → x = 1)
  (root2_eq_3 : ∀ x, x^2 + a * x + b = 0 → x = 3)
  (sum_of_roots_eq_2 : -a = 2)
  (opposite_sign_roots : ∀ x1 x2, x1 * x2 < 0) :
  ∃ prop, prop = "A" :=
sorry

end false_proposition_A_l127_127390


namespace max_value_neg_domain_l127_127512

theorem max_value_neg_domain (x : ℝ) (h : x < 0) : 
  ∃ y, y = 2 * x + 2 / x ∧ y ≤ -4 :=
sorry

end max_value_neg_domain_l127_127512


namespace monomial_2015_l127_127010

def a (n : ℕ) : ℤ := (-1 : ℤ)^n * (2 * n - 1)

theorem monomial_2015 :
  a 2015 * (x : ℤ) ^ 2015 = -4029 * (x : ℤ) ^ 2015 :=
by
  sorry

end monomial_2015_l127_127010


namespace order_of_values_l127_127392

noncomputable def a : ℝ := 21.2
noncomputable def b : ℝ := Real.sqrt 450 - 0.8
noncomputable def c : ℝ := 2 * Real.logb 5 2

theorem order_of_values : c < b ∧ b < a := by 
  sorry

end order_of_values_l127_127392


namespace sum_not_divisible_by_10_iff_l127_127116

theorem sum_not_divisible_by_10_iff (n : ℕ) :
  ¬ (1981^n + 1982^n + 1983^n + 1984^n) % 10 = 0 ↔ n % 4 = 0 :=
sorry

end sum_not_divisible_by_10_iff_l127_127116


namespace parallel_vectors_determine_t_l127_127304

theorem parallel_vectors_determine_t (t : ℝ) (h : (t, -6) = (k * -3, k * 2)) : t = 9 :=
by
  sorry

end parallel_vectors_determine_t_l127_127304


namespace set_clock_correctly_l127_127664

noncomputable def correct_clock_time
  (T_depart T_arrive T_depart_friend T_return : ℕ) 
  (T_visit := T_depart_friend - T_arrive) 
  (T_return_err := T_return - T_depart) 
  (T_total_travel := T_return_err - T_visit) 
  (T_travel_oneway := T_total_travel / 2) : ℕ :=
  T_depart + T_visit + T_travel_oneway

theorem set_clock_correctly 
  (T_depart T_arrive T_depart_friend T_return : ℕ)
  (h1 : T_depart ≤ T_return) -- The clock runs without accounting for the time away
  (h2 : T_arrive ≤ T_depart_friend) -- The friend's times are correct
  (h3 : T_return ≠ T_depart) -- The man was away for some non-zero duration
: 
  (correct_clock_time T_depart T_arrive T_depart_friend T_return) = 
  (T_depart + (T_depart_friend - T_arrive) + ((T_return - T_depart - (T_depart_friend - T_arrive)) / 2)) :=
sorry

end set_clock_correctly_l127_127664


namespace simplify_expression_l127_127315

theorem simplify_expression : (1 / (1 + Real.sqrt 3) * 1 / (1 + Real.sqrt 3)) = 1 - Real.sqrt 3 / 2 :=
by
  sorry

end simplify_expression_l127_127315


namespace find_s_l127_127254

theorem find_s (a b r1 r2 : ℝ) (h1 : r1 + r2 = -a) (h2 : r1 * r2 = b) :
    let new_root1 := (r1 + r2) * (r1 + r2)
    let new_root2 := (r1 * r2) * (r1 + r2)
    let s := b * a - a * a
    s = ab - a^2 :=
  by
    -- the proof goes here
    sorry

end find_s_l127_127254


namespace chloe_profit_l127_127375

theorem chloe_profit 
  (cost_per_dozen : ℕ)
  (selling_price_per_half_dozen : ℕ)
  (dozens_sold : ℕ)
  (h1 : cost_per_dozen = 50)
  (h2 : selling_price_per_half_dozen = 30)
  (h3 : dozens_sold = 50) : 
  (selling_price_per_half_dozen - cost_per_dozen / 2) * (dozens_sold * 2) = 500 :=
by 
  sorry

end chloe_profit_l127_127375


namespace smallest_n_satisfying_mod_cond_l127_127695

theorem smallest_n_satisfying_mod_cond (n : ℕ) : (15 * n - 3) % 11 = 0 ↔ n = 9 := by
  sorry

end smallest_n_satisfying_mod_cond_l127_127695


namespace water_jugs_problem_l127_127045

-- Definitions based on the conditions
variables (m n : ℕ) (relatively_prime_m_n : Nat.gcd m n = 1)
variables (k : ℕ) (hk : 1 ≤ k ∧ k ≤ m + n)

-- Statement of the theorem
theorem water_jugs_problem : 
    ∃ (x y z : ℕ), 
    (x = m ∨ x = n ∨ x = m + n) ∧ 
    (y = m ∨ y = n ∨ y = m + n) ∧ 
    (z = m ∨ z = n ∨ z = m + n) ∧ 
    (x ≤ m + n) ∧ 
    (y ≤ m + n) ∧ 
    (z ≤ m + n) ∧ 
    x + y + z = m + n ∧ 
    (x = k ∨ y = k ∨ z = k) :=
sorry

end water_jugs_problem_l127_127045


namespace area_of_triangle_COD_l127_127148

theorem area_of_triangle_COD (x p : ℕ) (hx : 0 < x) (hx' : x < 12) (hp : 0 < p) :
  (∃ A : ℚ, A = (x * p : ℚ) / 2) :=
sorry

end area_of_triangle_COD_l127_127148


namespace repeating_decimal_product_l127_127703

def repeating_decimal_12 := 12 / 99
def repeating_decimal_34 := 34 / 99

theorem repeating_decimal_product : (repeating_decimal_12 * repeating_decimal_34) = 136 / 3267 := by
  sorry

end repeating_decimal_product_l127_127703


namespace min_distance_to_line_l127_127713

-- Given that a point P(x, y) lies on the line x - y - 1 = 0
-- We need to prove that the minimum value of (x - 2)^2 + (y - 2)^2 is 1/2
theorem min_distance_to_line (x y: ℝ) (h: x - y - 1 = 0) :
  ∃ P : ℝ, P = (x - 2)^2 + (y - 2)^2 ∧ P = 1 / 2 :=
by
  sorry

end min_distance_to_line_l127_127713


namespace polynomial_sum_equals_one_l127_127599

theorem polynomial_sum_equals_one (a a_1 a_2 a_3 a_4 : ℤ) :
  (∀ x : ℤ, (2*x + 1)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  a - a_1 + a_2 - a_3 + a_4 = 1 :=
by
  sorry

end polynomial_sum_equals_one_l127_127599


namespace correct_calculation_l127_127178

theorem correct_calculation (x a : Real) :
  (3 * x^2 - x^2 ≠ 3) → 
  (-3 * a^2 - 2 * a^2 ≠ -a^2) →
  (x^3 / x ≠ 3) → 
  ((-x)^3 = -x^3) → 
  true :=
by
  intros _ _ _ _
  trivial

end correct_calculation_l127_127178


namespace four_digit_flippies_div_by_4_l127_127355

def is_flippy (n : ℕ) : Prop := 
  let digits := [4, 6]
  n / 1000 ∈ digits ∧
  (n / 100 % 10) ∈ digits ∧
  ((n / 10 % 10) = if (n / 100 % 10) = 4 then 6 else 4) ∧
  (n % 10) = if (n / 1000) = 4 then 6 else 4

def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

theorem four_digit_flippies_div_by_4 : 
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ is_flippy n ∧ is_divisible_by_4 n :=
by
  sorry

end four_digit_flippies_div_by_4_l127_127355


namespace proof_theorem_l127_127652

noncomputable def proof_problem 
  (m n : ℕ) 
  (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ 1) 
  (h3 : 0 ≤ y) (h4 : y ≤ 1) 
  (h5 : 0 ≤ z) (h6 : z ≤ 1) 
  (h7 : m > 0) (h8 : n > 0) 
  (h9 : m + n = p) : Prop :=
0 ≤ x^p + y^p + z^p - x^m * y^n - y^m * z^n - z^m * x^n ∧ 
x^p + y^p + z^p - x^m * y^n - y^m * z^n - z^m * x^n ≤ 1

theorem proof_theorem (m n : ℕ) (x y z : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 1) 
  (h3 : 0 ≤ y) (h4 : y ≤ 1) 
  (h5 : 0 ≤ z) (h6 : z ≤ 1) 
  (h7 : m > 0) (h8 : n > 0) 
  (h9 : m + n = p) : 
  proof_problem m n x y z h1 h2 h3 h4 h5 h6 h7 h8 h9 :=
by {
  sorry
}

end proof_theorem_l127_127652


namespace shekar_average_is_81_9_l127_127013

def shekar_average_marks (marks : List ℕ) : ℚ :=
  (marks.sum : ℚ) / marks.length

theorem shekar_average_is_81_9 :
  shekar_average_marks [92, 78, 85, 67, 89, 74, 81, 95, 70, 88] = 81.9 :=
by
  sorry

end shekar_average_is_81_9_l127_127013


namespace problem_solution_l127_127717

def f (x m : ℝ) : ℝ :=
  3 * x ^ 2 + m * (m - 6) * x + 5

theorem problem_solution (m n : ℝ) :
  (f 1 m > 0) ∧ (∀ x : ℝ, -1 < x ∧ x < 4 → f x m < n) ↔ (m = 3 ∧ n = 17) :=
by sorry

end problem_solution_l127_127717


namespace inequality_ab_equals_bc_l127_127311

-- Define the given conditions and state the theorem as per the proof problem
theorem inequality_ab_equals_bc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a^b * b^c * c^a ≤ a^a * b^b * c^c :=
by
  sorry

end inequality_ab_equals_bc_l127_127311


namespace michael_choices_l127_127588

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem michael_choices : combination 10 4 = 210 := by
  sorry

end michael_choices_l127_127588


namespace solve_for_x_l127_127316

theorem solve_for_x (x : ℝ) (h : (x - 5)^4 = (1 / 16)⁻¹) : x = 7 :=
by
  sorry

end solve_for_x_l127_127316


namespace calculate_expression_l127_127509

theorem calculate_expression : ( (3 / 20 + 5 / 200 + 7 / 2000) * 2 = 0.357 ) :=
by
  sorry

end calculate_expression_l127_127509


namespace smallest_four_digit_divisible_five_smallest_primes_l127_127078

theorem smallest_four_digit_divisible_five_smallest_primes :
  let lcm_5_min_primes := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 (Nat.lcm 7 11))
  ∀ m : ℕ, (m ≥ 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ m)) → m = 2310 :=
by
  let lcm_5_min_primes := 2310
  intro m h
  have h1 : m = 2310 := sorry
  exact h1

end smallest_four_digit_divisible_five_smallest_primes_l127_127078


namespace number_of_machines_l127_127769

theorem number_of_machines (X : ℕ)
  (h1 : 20 = (10 : ℝ) * X * 0.4) :
  X = 5 := sorry

end number_of_machines_l127_127769


namespace find_divisor_l127_127308

theorem find_divisor 
  (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h_dividend : dividend = 190) (h_quotient : quotient = 9) (h_remainder : remainder = 1) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 21 := 
by
  sorry

end find_divisor_l127_127308


namespace find_speed_in_second_hour_l127_127324

-- Define the given conditions as hypotheses
def speed_in_first_hour : ℝ := 50
def average_speed : ℝ := 55
def total_time : ℝ := 2

-- Define a function that represents the speed in the second hour
def speed_second_hour (s2 : ℝ) := 
  (speed_in_first_hour + s2) / total_time = average_speed

-- The statement to prove: the speed in the second hour is 60 km/h
theorem find_speed_in_second_hour : speed_second_hour 60 :=
by sorry

end find_speed_in_second_hour_l127_127324


namespace factor_poly1_factor_poly2_factor_poly3_l127_127704

-- Define the three polynomial functions.
def poly1 (x : ℝ) : ℝ := 2 * x^4 - 2
def poly2 (x : ℝ) : ℝ := x^4 - 18 * x^2 + 81
def poly3 (y : ℝ) : ℝ := (y^2 - 1)^2 + 11 * (1 - y^2) + 24

-- Formulate the goals: proving that each polynomial equals its respective factored form.
theorem factor_poly1 (x : ℝ) : poly1 x = 2 * (x^2 + 1) * (x + 1) * (x - 1) :=
sorry

theorem factor_poly2 (x : ℝ) : poly2 x = (x + 3)^2 * (x - 3)^2 :=
sorry

theorem factor_poly3 (y : ℝ) : poly3 y = (y + 2) * (y - 2) * (y + 3) * (y - 3) :=
sorry

end factor_poly1_factor_poly2_factor_poly3_l127_127704


namespace tangency_lines_intersect_at_diagonal_intersection_point_l127_127154

noncomputable def point := Type
noncomputable def line := Type

noncomputable def tangency (C : point) (l : line) : Prop := sorry
noncomputable def circumscribed (Q : point × point × point × point) (C : point) : Prop := sorry
noncomputable def intersects (l1 l2 : line) (P : point) : Prop := sorry
noncomputable def connects_opposite_tangency (Q : point × point × point × point) (l1 l2 : line) : Prop := sorry
noncomputable def diagonals_intersect_at (Q : point × point × point × point) (P : point) : Prop := sorry

theorem tangency_lines_intersect_at_diagonal_intersection_point :
  ∀ (Q : point × point × point × point) (C P : point), 
  circumscribed Q C →
  diagonals_intersect_at Q P →
  ∀ (l1 l2 : line), connects_opposite_tangency Q l1 l2 →
  intersects l1 l2 P :=
sorry

end tangency_lines_intersect_at_diagonal_intersection_point_l127_127154


namespace find_general_term_a_l127_127396

-- Define the sequence and conditions
noncomputable def S (n : ℕ) : ℚ :=
  if n = 0 then 0 else (n - 1) / (n * (n + 1))

-- General term to prove
def a (n : ℕ) : ℚ := 1 / (2^n) - 1 / (n * (n + 1))

theorem find_general_term_a :
  ∀ n : ℕ, n > 0 → S n + a n = (n - 1) / (n * (n + 1)) :=
by
  intro n hn
  sorry -- Proof omitted

end find_general_term_a_l127_127396


namespace solve_range_m_l127_127957

variable (m : ℝ)
def p := m < 0
def q := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem solve_range_m (hpq : p m ∧ q m) : -2 < m ∧ m < 0 := 
  sorry

end solve_range_m_l127_127957


namespace factorize_expression_l127_127227

variable {R : Type} [CommRing R] (a x y : R)

theorem factorize_expression :
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l127_127227


namespace minute_hand_angle_is_pi_six_minute_hand_arc_length_is_2pi_third_l127_127548

theorem minute_hand_angle_is_pi_six (radius : ℝ) (fast_min : ℝ) (h1 : radius = 4) (h2 : fast_min = 5) :
  (fast_min / 60 * 2 * Real.pi = Real.pi / 6) :=
by sorry

theorem minute_hand_arc_length_is_2pi_third (radius : ℝ) (angle : ℝ) (fast_min : ℝ) (h1 : radius = 4) (h2 : angle = Real.pi / 6) (h3 : fast_min = 5) :
  (radius * angle = 2 * Real.pi / 3) :=
by sorry

end minute_hand_angle_is_pi_six_minute_hand_arc_length_is_2pi_third_l127_127548


namespace distinct_four_digit_integers_with_digit_product_18_l127_127567

theorem distinct_four_digit_integers_with_digit_product_18 : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in digits.prod = 18) → 
  (finset.univ.filter (λ m, (let mdigits := [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10] in mdigits.prod = 18))).card = 36 :=
by
  sorry

end distinct_four_digit_integers_with_digit_product_18_l127_127567


namespace vector_parallel_addition_l127_127967

theorem vector_parallel_addition 
  (x : ℝ)
  (a : ℝ × ℝ := (2, 1))
  (b : ℝ × ℝ := (x, -2)) 
  (h_parallel : 2 / x = 1 / -2) :
  a + b = (-2, -1) := 
by
  -- While the proof is omitted, the statement is complete and correct.
  sorry

end vector_parallel_addition_l127_127967


namespace binom_20_10_l127_127537

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end binom_20_10_l127_127537


namespace arun_weight_lower_limit_l127_127934

theorem arun_weight_lower_limit :
  ∃ (w : ℝ), w > 60 ∧ w <= 64 ∧ (∀ (a : ℝ), 60 < a ∧ a <= 64 → ((a + 64) / 2 = 63) → a = 62) :=
by
  sorry

end arun_weight_lower_limit_l127_127934


namespace combine_terms_l127_127179

theorem combine_terms (a b : ℕ) : 
  let lhs := (2 * a * b ^ 3)
  let rhs := (- a * b ^ 3)
  lhs + rhs = (2 - 1) * a * b ^ 3 :=
by sorry

end combine_terms_l127_127179


namespace force_magnitudes_ratio_l127_127899

theorem force_magnitudes_ratio (a d : ℝ) (h1 : (a + 2 * d)^2 = a^2 + (a + d)^2) :
  ∃ k : ℝ, k > 0 ∧ (a + d) = a * (4 / 3) ∧ (a + 2 * d) = a * (5 / 3) :=
by
  sorry

end force_magnitudes_ratio_l127_127899


namespace determine_a_l127_127106

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 - 2 * a * x + 1 

theorem determine_a (a : ℝ) (h : ¬ (∀ x : ℝ, 0 < x ∧ x < 1 → f a x ≠ 0)) : a > 1 :=
sorry

end determine_a_l127_127106


namespace triangle_inequality_l127_127598

variable {a b c : ℝ}

theorem triangle_inequality (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  (a / Real.sqrt (2*b^2 + 2*c^2 - a^2)) + (b / Real.sqrt (2*c^2 + 2*a^2 - b^2)) + 
  (c / Real.sqrt (2*a^2 + 2*b^2 - c^2)) ≥ Real.sqrt 3 := by
  sorry

end triangle_inequality_l127_127598


namespace robins_hair_cut_l127_127458

theorem robins_hair_cut (x : ℕ) : 16 - x + 12 = 17 → x = 11 := by
  sorry

end robins_hair_cut_l127_127458


namespace total_fruits_on_display_l127_127471

-- Declare the variables and conditions as hypotheses
variables (apples oranges bananas : ℕ)
variables (h1 : apples = 2 * oranges)
variables (h2 : oranges = 2 * bananas)
variables (h3 : bananas = 5)

-- Define what we want to prove
theorem total_fruits_on_display : apples + oranges + bananas = 35 :=
by sorry

end total_fruits_on_display_l127_127471


namespace power_mod_1000_l127_127456

theorem power_mod_1000 (N : ℤ) (h : Int.gcd N 10 = 1) : (N ^ 101 ≡ N [ZMOD 1000]) :=
  sorry

end power_mod_1000_l127_127456


namespace total_messages_sent_l127_127676

theorem total_messages_sent 
    (lucia_day1 : ℕ)
    (alina_day1_less : ℕ)
    (lucia_day1_messages : lucia_day1 = 120)
    (alina_day1_messages : alina_day1_less = 20)
    : (lucia_day2 : ℕ)
    (alina_day2 : ℕ)
    (lucia_day2_eq : lucia_day2 = lucia_day1 / 3)
    (alina_day2_eq : alina_day2 = (lucia_day1 - alina_day1_less) * 2)
    (messages_day3_eq : ∀ (lucia_day3 alina_day3 : ℕ), lucia_day3 + alina_day3 = lucia_day1 + (lucia_day1 - alina_day1_less))
    : lucia_day1 + alina_day1_less + (lucia_day2 + alina_day2) + messages_day3_eq 120 100 = 680 :=
    sorry

end total_messages_sent_l127_127676


namespace find_values_of_a_b_l127_127959

variable (a b : ℤ)

def A : Set ℤ := {1, b, a + b}
def B : Set ℤ := {a - b, a * b}
def common_set : Set ℤ := {-1, 0}

theorem find_values_of_a_b (h : A a b ∩ B a b = common_set) : (a, b) = (-1, 0) := by
  sorry

end find_values_of_a_b_l127_127959


namespace find_W_l127_127030

def digit_sum_eq (X Y Z W : ℕ) : Prop := X * 10 + Y + Z * 10 + X = W * 10 + X
def digit_diff_eq (X Y Z : ℕ) : Prop := X * 10 + Y - (Z * 10 + X) = X
def is_digit (n : ℕ) : Prop := n < 10

theorem find_W (X Y Z W : ℕ) (h1 : digit_sum_eq X Y Z W) (h2 : digit_diff_eq X Y Z) 
  (hX : is_digit X) (hY : is_digit Y) (hZ : is_digit Z) (hW : is_digit W) : W = 0 := 
sorry

end find_W_l127_127030


namespace problem1_problem2_l127_127879

-- Define propositions P and Q under the given conditions
def P (a x : ℝ) : Prop := 2 * x^2 - 5 * a * x - 3 * a^2 < 0

def Q (x : ℝ) : Prop := (2 * Real.sin x > 1) ∧ (x^2 - x - 2 < 0)

-- Problem 1: Prove that if a = 2 and p ∧ q holds true, then the range of x is (π/6, 2)
theorem problem1 (x : ℝ) (hx1 : P 2 x ∧ Q x) : (Real.pi / 6 < x ∧ x < 2) :=
sorry

-- Problem 2: Prove that if ¬P is a sufficient but not necessary condition for ¬Q, then the range of a is [2/3, ∞)
theorem problem2 (a : ℝ) (h₁ : ∀ x, Q x → P a x) (h₂ : ∃ x, Q x → ¬P a x) : a ≥ 2 / 3 :=
sorry

end problem1_problem2_l127_127879


namespace pure_imaginary_x_l127_127971

theorem pure_imaginary_x (x : ℝ) (h: (x - 2008) = 0) : x = 2008 :=
by
  sorry

end pure_imaginary_x_l127_127971


namespace kaleb_toys_can_buy_l127_127747

theorem kaleb_toys_can_buy (saved_money : ℕ) (allowance_received : ℕ) (allowance_increase_percent : ℕ) (toy_cost : ℕ) (half_total_spend : ℕ) :
  saved_money = 21 →
  allowance_received = 15 →
  allowance_increase_percent = 20 →
  toy_cost = 6 →
  half_total_spend = (saved_money + allowance_received) / 2 →
  (half_total_spend / toy_cost) = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end kaleb_toys_can_buy_l127_127747


namespace number_of_roots_l127_127122

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 2 * b * x + 3 * c

theorem number_of_roots (a b c x₁ x₂ : ℝ) (h_extreme : x₁ ≠ x₂)
    (h_fx1 : f a b c x₁ = x₁) :
    (∃ (r : ℝ), 3 * (f a b c r)^2 + 4 * a * (f a b c r) + 2 * b = 0) :=
sorry

end number_of_roots_l127_127122


namespace jared_current_age_jared_current_age_l127_127175

theorem jared_current_age (jared_age_two_years_ago : ℕ) (tom_age_in_five_years : ℕ) : 
  (jared_age_two_years_ago + 2) = 48 :=
by
  sorry

variables (tom_curr_age : ℕ) (tom_age_two_years_ago : ℕ) (jared_age_two_years_ago : ℕ)
(h1 : tom_curr_age = tom_age_in_five_years - 5)
(h2 : tom_age_two_years_ago = tom_curr_age - 2)
(h3 : jared_age_two_years_ago = 2 * tom_age_two_years_ago)

include h1 h2 h3

theorem jared_current_age (jared_age_two_years_ago : ℕ) (tom_age_in_five_years : ℕ) : 
  (jared_age_two_years_ago + 2) = 48 :=
by
  let tom_curr_age := tom_age_in_five_years - 5
  let tom_age_two_years_ago := tom_curr_age - 2
  let jared_age_two_years_ago := 2 * tom_age_two_years_ago
  exact Eq.refl 48

#check @jared_current_age

end jared_current_age_jared_current_age_l127_127175


namespace totalCarsProduced_is_29621_l127_127920

def numSedansNA    := 3884
def numSUVsNA      := 2943
def numPickupsNA   := 1568

def numSedansEU    := 2871
def numSUVsEU      := 2145
def numPickupsEU   := 643

def numSedansASIA  := 5273
def numSUVsASIA    := 3881
def numPickupsASIA := 2338

def numSedansSA    := 1945
def numSUVsSA      := 1365
def numPickupsSA   := 765

def totalCarsProduced : Nat :=
  numSedansNA + numSUVsNA + numPickupsNA +
  numSedansEU + numSUVsEU + numPickupsEU +
  numSedansASIA + numSUVsASIA + numPickupsASIA +
  numSedansSA + numSUVsSA + numPickupsSA

theorem totalCarsProduced_is_29621 : totalCarsProduced = 29621 :=
by
  sorry

end totalCarsProduced_is_29621_l127_127920


namespace prime_cube_plus_five_implies_prime_l127_127573

theorem prime_cube_plus_five_implies_prime (p : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime (p^3 + 5)) : p^5 - 7 = 25 := 
by
  sorry

end prime_cube_plus_five_implies_prime_l127_127573


namespace ad_lt_bc_l127_127708

theorem ad_lt_bc (a b c d : ℝ ) (h1a : a > 0) (h1b : b > 0) (h1c : c > 0) (h1d : d > 0)
  (h2 : a + d = b + c) (h3 : |a - d| < |b - c|) : a * d < b * c :=
  sorry

end ad_lt_bc_l127_127708


namespace log_relation_l127_127252

variable (a b : ℝ)

theorem log_relation (h : real.log 3 a < real.log 3 b ∧ real.log 3 b < 0) : 0 < a ∧ a < b ∧ b < 1 :=
by
  sorry

end log_relation_l127_127252


namespace harly_adopts_percentage_l127_127114

/-- Definitions for the conditions -/
def initial_dogs : ℝ := 80
def dogs_taken_back : ℝ := 5
def dogs_left : ℝ := 53

/-- Define the percentage of dogs adopted out -/
def percentage_adopted (P : ℝ) := P

/-- Lean 4 statement where we prove that if the given conditions are met, then the percentage of dogs initially adopted out is 40 -/
theorem harly_adopts_percentage : 
  ∃ P : ℝ, 
    (initial_dogs - (percentage_adopted P / 100 * initial_dogs) + dogs_taken_back = dogs_left) 
    ∧ P = 40 :=
by
  sorry

end harly_adopts_percentage_l127_127114


namespace tangent_y_intercept_l127_127939

theorem tangent_y_intercept :
  let C1 := (2, 4)
  let r1 := 5
  let C2 := (14, 9)
  let r2 := 10
  let m := 120 / 119
  m > 0 → ∃ b, b = 912 / 119 := by
  sorry

end tangent_y_intercept_l127_127939


namespace largest_prime_factor_12321_l127_127233

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p :=
begin
  use 83,
  split,
  { -- Prove that 83 is a prime number
    sorry },
  split,
  { -- Prove that 83 divides 12321
    sorry },
  { -- Prove that any other prime factor of 12321 is less than or equal to 83
    sorry }
end

end largest_prime_factor_12321_l127_127233


namespace find_m_l127_127751

theorem find_m (a : ℕ → ℝ) (m : ℕ) (h_pos : m > 0) 
  (h_a0 : a 0 = 37) (h_a1 : a 1 = 72) (h_am : a m = 0)
  (h_rec : ∀ k, 1 ≤ k ∧ k ≤ m - 1 → a (k + 1) = a (k - 1) - 3 / a k) :
  m = 889 :=
sorry

end find_m_l127_127751


namespace island_challenge_probability_l127_127622
open Nat

theorem island_challenge_probability :
  let total_ways := choose 20 3
  let ways_one_tribe := choose 10 3
  let combined_ways := 2 * ways_one_tribe
  let probability := combined_ways / total_ways
  probability = (20 : ℚ) / 95 :=
by
  sorry

end island_challenge_probability_l127_127622


namespace kim_saplings_left_l127_127596

def number_of_pits : ℕ := 80
def proportion_sprout : ℚ := 0.25
def saplings_sold : ℕ := 6

theorem kim_saplings_left : 
  (number_of_pits * proportion_sprout - saplings_sold = 14) :=
begin
  sorry
end

end kim_saplings_left_l127_127596


namespace remaining_movies_l127_127469

-- Definitions based on the problem's conditions
def total_movies : ℕ := 8
def watched_movies : ℕ := 4

-- Theorem statement to prove that you still have 4 movies left to watch
theorem remaining_movies : total_movies - watched_movies = 4 :=
by
  sorry

end remaining_movies_l127_127469


namespace explicit_form_l127_127848

-- Define the functional equation
def f (x : ℝ) : ℝ := sorry

-- Define the condition that f(x) satisfies
axiom functional_equation (x : ℝ) (h : x ≠ 0) : f x = 2 * f (1 / x) + 3 * x

-- State the theorem that we need to prove
theorem explicit_form (x : ℝ) (h : x ≠ 0) : f x = -x - (2 / x) :=
by
  sorry

end explicit_form_l127_127848


namespace find_x_complementary_l127_127871

-- Define the conditions.
def are_complementary (a b : ℝ) : Prop := a + b = 90

-- The main theorem statement with the condition and conclusion.
theorem find_x_complementary : ∀ x : ℝ, are_complementary (2*x) (3*x) → x = 18 := 
by
  intros x h
  -- sorry is a placeholder for the proof.
  sorry

end find_x_complementary_l127_127871


namespace equivalent_problem_l127_127342

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x^2 else sorry

theorem equivalent_problem 
  (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_periodic : ∀ x : ℝ, f (x + 2) = f x)
  (f_interval : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x^2)
  : f (-3/2) + f 1 = 3/4 :=
sorry

end equivalent_problem_l127_127342


namespace fair_hair_women_percentage_l127_127655

-- Definitions based on conditions
def total_employees (E : ℝ) := E
def women_with_fair_hair (E : ℝ) := 0.28 * E
def fair_hair_employees (E : ℝ) := 0.70 * E

-- Theorem to prove
theorem fair_hair_women_percentage (E : ℝ) (hE : E > 0) :
  (women_with_fair_hair E) / (fair_hair_employees E) * 100 = 40 :=
by 
  -- Sorry denotes the proof is omitted
  sorry

end fair_hair_women_percentage_l127_127655


namespace y_pow_x_eq_nine_l127_127277

theorem y_pow_x_eq_nine (x y : ℝ) (h : x^2 + y^2 - 4 * x + 6 * y + 13 = 0) : y^x = 9 := by
  sorry

end y_pow_x_eq_nine_l127_127277


namespace negation_of_P_is_exists_Q_l127_127033

def P (x : ℝ) : Prop := x^2 - x + 3 > 0

theorem negation_of_P_is_exists_Q :
  (¬ (∀ x : ℝ, P x)) ↔ (∃ x : ℝ, ¬ P x) :=
sorry

end negation_of_P_is_exists_Q_l127_127033


namespace total_cookies_eaten_l127_127782

-- Definitions of the cookies eaten
def charlie_cookies := 15
def father_cookies := 10
def mother_cookies := 5

-- The theorem to prove the total number of cookies eaten
theorem total_cookies_eaten : charlie_cookies + father_cookies + mother_cookies = 30 := by
  sorry

end total_cookies_eaten_l127_127782


namespace find_original_number_l127_127915

theorem find_original_number : ∃ (N : ℤ), (∃ (k : ℤ), N - 30 = 87 * k) ∧ N = 117 :=
by
  sorry

end find_original_number_l127_127915


namespace number_of_complete_decks_l127_127924

theorem number_of_complete_decks (total_cards : ℕ) (additional_cards : ℕ) (cards_per_deck : ℕ) 
(h1 : total_cards = 319) (h2 : additional_cards = 7) (h3 : cards_per_deck = 52) : 
total_cards - additional_cards = (cards_per_deck * 6) :=
by
  sorry

end number_of_complete_decks_l127_127924


namespace sofa_love_seat_cost_l127_127359

theorem sofa_love_seat_cost (love_seat_cost : ℕ) (sofa_cost : ℕ) 
    (h₁ : love_seat_cost = 148) (h₂ : sofa_cost = 2 * love_seat_cost) :
    love_seat_cost + sofa_cost = 444 := 
by
  sorry

end sofa_love_seat_cost_l127_127359


namespace largest_prime_factor_12321_l127_127232

theorem largest_prime_factor_12321 : ∃ p, prime p ∧ (∀ q, prime q ∧ q ∣ 12321 → q ≤ p) ∧ p = 19 :=
by {
  sorry
}

end largest_prime_factor_12321_l127_127232


namespace team_total_points_l127_127379

theorem team_total_points (Connor_score Amy_score Jason_score : ℕ) :
  Connor_score = 2 →
  Amy_score = Connor_score + 4 →
  Jason_score = 2 * Amy_score →
  Connor_score + Amy_score + Jason_score = 20 :=
by
  intros
  sorry

end team_total_points_l127_127379


namespace spherical_coordinates_equivalence_l127_127869

theorem spherical_coordinates_equivalence
  (ρ θ φ : ℝ)
  (h_ρ : ρ > 0)
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h_φ : φ = 2 * Real.pi - (7 * Real.pi / 4)) :
  (ρ, θ, φ) = (4, 3 * Real.pi / 4, Real.pi / 4) :=
by 
  sorry

end spherical_coordinates_equivalence_l127_127869


namespace vector_subtraction_result_l127_127964

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_subtraction_result :
  2 • a - b = (7, -2) :=
by
  simp [a, b]
  sorry

end vector_subtraction_result_l127_127964


namespace ball_reaches_height_l127_127351

theorem ball_reaches_height (h₀ : ℝ) (ratio : ℝ) (target_height : ℝ) (bounces : ℕ) 
  (initial_height : h₀ = 16) 
  (bounce_ratio : ratio = 1/3) 
  (target : target_height = 2) 
  (bounce_count : bounces = 7) :
  h₀ * (ratio ^ bounces) < target_height := 
sorry

end ball_reaches_height_l127_127351


namespace find_n_l127_127638

theorem find_n (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) = 14) : n = 2 :=
sorry

end find_n_l127_127638


namespace price_of_turbans_l127_127853

theorem price_of_turbans : 
  ∀ (salary_A salary_B salary_C : ℝ) (months_A months_B months_C : ℕ) (payment_A payment_B payment_C : ℝ)
    (prorated_salary_A prorated_salary_B prorated_salary_C : ℝ),
  salary_A = 120 → 
  salary_B = 150 → 
  salary_C = 180 → 
  months_A = 8 → 
  months_B = 7 → 
  months_C = 10 → 
  payment_A = 80 → 
  payment_B = 87.50 → 
  payment_C = 150 → 
  prorated_salary_A = (salary_A * (months_A / 12 : ℝ)) → 
  prorated_salary_B = (salary_B * (months_B / 12 : ℝ)) → 
  prorated_salary_C = (salary_C * (months_C / 12 : ℝ)) → 
  ∃ (price_A price_B price_C : ℝ),
  price_A = payment_A - prorated_salary_A ∧ 
  price_B = payment_B - prorated_salary_B ∧ 
  price_C = payment_C - prorated_salary_C ∧ 
  price_A = 0 ∧ price_B = 0 ∧ price_C = 0 := 
by
  sorry

end price_of_turbans_l127_127853


namespace distinct_four_digit_numbers_product_18_l127_127571

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end distinct_four_digit_numbers_product_18_l127_127571


namespace probability_three_even_dice_l127_127885

theorem probability_three_even_dice :
  let p_even := 1 / 2
  let combo := Nat.choose 5 3
  let probability := combo * (p_even ^ 3) * ((1 - p_even) ^ 2)
  probability = 5 / 16 := 
by
  sorry

end probability_three_even_dice_l127_127885


namespace f_at_2008_l127_127709

noncomputable def f : ℝ → ℝ := sorry
noncomputable def finv : ℝ → ℝ := sorry

axiom f_inverse : ∀ x, f (finv x) = x ∧ finv (f x) = x
axiom f_at_9 : f 9 = 18

theorem f_at_2008 : f 2008 = -1981 :=
by
  sorry

end f_at_2008_l127_127709


namespace smallest_divisible_four_digit_number_l127_127091

theorem smallest_divisible_four_digit_number :
  let n := 2310
  in ∃ k : ℕ, n * k >= 1000 ∧ n = 2 * 3 * 5 * 7 * 11 :=
begin
  let n := 2310,
  use 1,
  split,
  { norm_num },
  { norm_num },
  sorry
end

end smallest_divisible_four_digit_number_l127_127091


namespace find_probability_l127_127425

noncomputable def probability_within_0_80 (η : ℝ → ℝ) (δ : ℝ) : Prop :=
  ∀ η, (η : NormalDist 100 (δ ^ 2)) → (h₁ : ∀ x, 80 < x ∧ x < 120 → prob η x = 0.6) →
  (h₂ : δ > 0) → (prob η (0, 80) = 0.2)

-- Lean 4 statement expressing the equivalent of the mathematical problem
theorem find_probability (η : ℝ → ℝ) (δ : ℝ) (h₁ : ∀ x, 80 < x ∧ x < 120 → prob η x = 0.6)
  (h₂ : δ > 0) (h₃ : η = NormalDist 100 (δ ^ 2)) : 
  prob η (0, 80) = 0.2 :=
sorry  -- Proof not included, as per instructions

end find_probability_l127_127425


namespace restaurant_earnings_l127_127669

theorem restaurant_earnings :
  let set1 := 10 * 8 in
  let set2 := 5 * 10 in
  let set3 := 20 * 4 in
  set1 + set2 + set3 = 210 :=
by
  let set1 := 10 * 8
  let set2 := 5 * 10
  let set3 := 20 * 4
  exact (by ring : set1 + set2 + set3 = 210)

end restaurant_earnings_l127_127669


namespace f_inequality_l127_127774

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem f_inequality (x : ℝ) : f (3^x) ≥ f (2^x) := 
by 
  sorry

end f_inequality_l127_127774


namespace part1_part2_l127_127400

def quadratic_inequality_A (x m : ℝ) := -x^2 + 2 * m * x + 4 - m^2 ≥ 0
def quadratic_inequality_B (x : ℝ) := 2 * x^2 - 5 * x - 7 < 0

theorem part1 (m : ℝ) :
  (∀ x, quadratic_inequality_A x m ∧ quadratic_inequality_B x ↔ 0 ≤ x ∧ x < 7 / 2) →
  m = 2 := by sorry

theorem part2 (m : ℝ) :
  (∀ x, quadratic_inequality_B x → ¬ quadratic_inequality_A x m) →
  m ≤ -3 ∨ 11 / 2 ≤ m := by sorry

end part1_part2_l127_127400


namespace sufficient_condition_for_reciprocal_square_not_necessary_condition_for_reciprocal_square_l127_127102

variable {a b : ℝ}

theorem sufficient_condition_for_reciprocal_square :
  (b > a ∧ a > 0) → (1 / a^2 > 1 / b^2) :=
sorry

theorem not_necessary_condition_for_reciprocal_square :
  ¬((1 / a^2 > 1 / b^2) → (b > a ∧ a > 0)) :=
sorry

end sufficient_condition_for_reciprocal_square_not_necessary_condition_for_reciprocal_square_l127_127102


namespace intersection_of_A_and_B_l127_127963

-- Conditions: definitions of sets A and B
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def set_B : Set ℝ := {x | x < 1}

-- The proof goal: A ∩ B = {x | -1 ≤ x ∧ x < 1}
theorem intersection_of_A_and_B : set_A ∩ set_B = {x | -1 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l127_127963


namespace find_c_l127_127098

   variable {a b c : ℝ}
   
   theorem find_c (h1 : 4 * a - 3 * b + c = 0)
     (h2 : (a - 1)^2 + (b - 1)^2 = 4) :
     c = 9 ∨ c = -11 := 
   by
     sorry
   
end find_c_l127_127098


namespace cost_per_book_l127_127475

-- Definitions and conditions
def number_of_books : ℕ := 8
def amount_tommy_has : ℕ := 13
def amount_tommy_needs_to_save : ℕ := 27

-- Total money Tommy needs to buy the books
def total_amount_needed : ℕ := amount_tommy_has + amount_tommy_needs_to_save

-- Proven statement
theorem cost_per_book : (total_amount_needed / number_of_books) = 5 := by
  -- Skip proof
  sorry

end cost_per_book_l127_127475


namespace combined_salaries_of_A_B_C_E_is_correct_l127_127035

-- Given conditions
def D_salary : ℕ := 7000
def average_salary : ℕ := 8800
def n_individuals : ℕ := 5

-- Combined salary of A, B, C, and E
def combined_salaries : ℕ := 37000

theorem combined_salaries_of_A_B_C_E_is_correct :
  (average_salary * n_individuals - D_salary) = combined_salaries :=
by
  sorry

end combined_salaries_of_A_B_C_E_is_correct_l127_127035


namespace batting_average_is_60_l127_127352

-- Definitions for conditions:
def highest_score : ℕ := 179
def difference_highest_lowest : ℕ := 150
def average_44_innings : ℕ := 58
def innings_excluding_highest_lowest : ℕ := 44
def total_innings : ℕ := 46

-- Lowest score
def lowest_score : ℕ := highest_score - difference_highest_lowest

-- Total runs in 44 innings
def total_runs_44 : ℕ := average_44_innings * innings_excluding_highest_lowest

-- Total runs in 46 innings
def total_runs_46 : ℕ := total_runs_44 + highest_score + lowest_score

-- Batting average in 46 innings
def batting_average_46 : ℕ := total_runs_46 / total_innings

-- The theorem to prove
theorem batting_average_is_60 :
  batting_average_46 = 60 :=
sorry

end batting_average_is_60_l127_127352


namespace animal_sale_money_l127_127759

theorem animal_sale_money (G S : ℕ) (h1 : G + S = 360) (h2 : 5 * S = 7 * G) : 
  (1/2 * G * 40) + (2/3 * S * 30) = 7200 := 
by
  sorry

end animal_sale_money_l127_127759


namespace ashley_champagne_bottles_l127_127210

theorem ashley_champagne_bottles (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) 
  (h1 : guests = 120) (h2 : glasses_per_guest = 2) (h3 : servings_per_bottle = 6) : 
  (guests * glasses_per_guest) / servings_per_bottle = 40 :=
by
  -- The proof will go here
  sorry

end ashley_champagne_bottles_l127_127210


namespace factorize_expression_l127_127229

variable (a x y : ℝ)

theorem factorize_expression : a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l127_127229


namespace product_of_two_numbers_l127_127465

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : a * b = 875 :=
sorry

end product_of_two_numbers_l127_127465


namespace intersection_lg_1_x_squared_zero_t_le_one_l127_127753

theorem intersection_lg_1_x_squared_zero_t_le_one  :
  let M := {x | 0 ≤ x ∧ x ≤ 2}
  let N := {x | -1 < x ∧ x < 1}
  M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_lg_1_x_squared_zero_t_le_one_l127_127753


namespace scientific_notation_of_393000_l127_127891

theorem scientific_notation_of_393000 : 
  ∃ (a : ℝ) (n : ℤ), a = 3.93 ∧ n = 5 ∧ (393000 = a * 10^n) := 
by
  use 3.93
  use 5
  sorry

end scientific_notation_of_393000_l127_127891


namespace seating_problem_smallest_n_l127_127328

   theorem seating_problem_smallest_n (k : ℕ) (n : ℕ) (h1 : 2 ≤ k) (h2 : k < n)
     (h3 : 2 * (nat.factorial (n-1) / nat.factorial (n-k)) = 
            (nat.factorial n / (nat.factorial (k-2) * nat.factorial (n-k+2))) * nat.factorial (k-2)) :
     n = 12 :=
   sorry
   
end seating_problem_smallest_n_l127_127328


namespace race_distance_l127_127742

theorem race_distance (D : ℝ)
  (A_time : D / 36 * 45 = D + 20) : 
  D = 80 :=
by
  sorry

end race_distance_l127_127742


namespace solution_of_valve_problem_l127_127697

noncomputable def valve_filling_problem : Prop :=
  ∃ (x y z : ℝ), 
    (x + y + z = 1 / 2) ∧    -- Condition when all three valves are open
    (x + z = 1 / 3) ∧        -- Condition when valves X and Z are open
    (y + z = 1 / 4) ∧        -- Condition when valves Y and Z are open
    (1 / (x + y) = 2.4)      -- Required condition for valves X and Y

theorem solution_of_valve_problem : valve_filling_problem :=
sorry

end solution_of_valve_problem_l127_127697


namespace find_k_l127_127496

theorem find_k (k : ℚ) : 
  ((3, -8) ≠ (k, 20)) ∧ 
  (∃ m, (4 * m = -3) ∧ (20 - (-8) = m * (k - 3))) → 
  k = -103/3 := 
by
  sorry

end find_k_l127_127496


namespace parallel_vectors_x_value_l127_127554

variable {x : ℝ}

theorem parallel_vectors_x_value (h : (1 / x) = (2 / -6)) : x = -3 := sorry

end parallel_vectors_x_value_l127_127554


namespace find_number_l127_127012

theorem find_number (x : ℝ) (h : (5/4) * x = 40) : x = 32 := 
sorry

end find_number_l127_127012


namespace average_time_to_win_permit_l127_127660

theorem average_time_to_win_permit :
  let p n := (9/10)^(n-1) * (1/10)
  ∑' n, n * p n = 10 :=
sorry

end average_time_to_win_permit_l127_127660


namespace average_letters_per_day_l127_127764

theorem average_letters_per_day 
  (letters_tuesday : ℕ)
  (letters_wednesday : ℕ)
  (days : ℕ := 2) 
  (letters_total : ℕ := letters_tuesday + letters_wednesday) :
  letters_tuesday = 7 → letters_wednesday = 3 → letters_total / days = 5 :=
by
  -- The proof is omitted
  sorry

end average_letters_per_day_l127_127764


namespace systematic_sampling_distance_l127_127781

-- Conditions
def total_students : ℕ := 1200
def sample_size : ℕ := 30

-- Problem: Compute sampling distance
def sampling_distance (n : ℕ) (m : ℕ) : ℕ := n / m

-- The formal proof statement
theorem systematic_sampling_distance :
  sampling_distance total_students sample_size = 40 := by
  sorry

end systematic_sampling_distance_l127_127781


namespace sin_squared_equiv_cosine_l127_127067

theorem sin_squared_equiv_cosine :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = (Real.sqrt 2) / 2 :=
by sorry

end sin_squared_equiv_cosine_l127_127067


namespace logarithm_base_l127_127166

theorem logarithm_base (x : ℝ) (b : ℝ) : (9 : ℝ)^(x + 5) = (16 : ℝ)^x → b = 16 / 9 → x = Real.log 9^5 / Real.log b := by sorry

end logarithm_base_l127_127166


namespace abs_expression_eq_five_l127_127813

theorem abs_expression_eq_five : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry -- proof omitted

end abs_expression_eq_five_l127_127813


namespace simplify_expression_l127_127768

theorem simplify_expression (k : ℤ) : 
  let a := 1
  let b := 3
  (6 * k + 18) / 6 = k + 3 ∧ a / b = 1 / 3 :=
by
  sorry

end simplify_expression_l127_127768


namespace trajectory_of_M_l127_127024

theorem trajectory_of_M (M : ℝ × ℝ) (h : (M.2 < 0 → M.1 = 0) ∧ (M.2 ≥ 0 → M.1 ^ 2 = 8 * M.2)) :
  (M.2 < 0 → M.1 = 0) ∧ (M.2 ≥ 0 → M.1 ^ 2 = 8 * M.2) :=
by
  sorry

end trajectory_of_M_l127_127024


namespace stratified_sampling_correct_l127_127188

def total_employees : ℕ := 150
def senior_titles : ℕ := 15
def intermediate_titles : ℕ := 45
def general_staff : ℕ := 90
def sample_size : ℕ := 30

def stratified_sampling (total_employees senior_titles intermediate_titles general_staff sample_size : ℕ) : (ℕ × ℕ × ℕ) :=
  (senior_titles * sample_size / total_employees, 
   intermediate_titles * sample_size / total_employees, 
   general_staff * sample_size / total_employees)

theorem stratified_sampling_correct :
  stratified_sampling total_employees senior_titles intermediate_titles general_staff sample_size = (3, 9, 18) :=
  by sorry

end stratified_sampling_correct_l127_127188


namespace harmonic_mean_pairs_count_l127_127239

theorem harmonic_mean_pairs_count :
  ∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, p.1 < p.2 ∧ 2 * p.1 * p.2 = 4^15 * (p.1 + p.2)) ∧ s.card = 29 :=
sorry

end harmonic_mean_pairs_count_l127_127239


namespace initial_students_count_l127_127772

theorem initial_students_count (n : ℕ) (W : ℝ) :
  (W = n * 28) →
  (W + 4 = (n + 1) * 27.2) →
  n = 29 :=
by
  intros hW hw_avg
  -- Proof goes here
  sorry

end initial_students_count_l127_127772


namespace find_a_l127_127105

variable (a x : ℝ)

noncomputable def curve1 (x : ℝ) := x + Real.log x
noncomputable def curve2 (a x : ℝ) := a * x^2 + (a + 2) * x + 1

theorem find_a : (curve1 1 = 1 ∧ curve1 1 = curve2 a 1) → a = 8 :=
by
  sorry

end find_a_l127_127105


namespace distinct_four_digit_positive_integers_product_18_l127_127557

theorem distinct_four_digit_positive_integers_product_18 :
  Finset.card {n | ∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
                             (1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9) ∧
                             a * b * c * d = 18} = 24 :=
by
  sorry

end distinct_four_digit_positive_integers_product_18_l127_127557


namespace rhombus_area_l127_127022

def diagonal1 : ℝ := 24
def diagonal2 : ℝ := 16

theorem rhombus_area : 0.5 * diagonal1 * diagonal2 = 192 :=
by
  sorry

end rhombus_area_l127_127022


namespace store_A_cheaper_than_store_B_l127_127908

noncomputable def store_A_full_price : ℝ := 125
noncomputable def store_A_discount_pct : ℝ := 0.08
noncomputable def store_B_full_price : ℝ := 130
noncomputable def store_B_discount_pct : ℝ := 0.10

noncomputable def final_price_A : ℝ :=
  store_A_full_price * (1 - store_A_discount_pct)

noncomputable def final_price_B : ℝ :=
  store_B_full_price * (1 - store_B_discount_pct)

theorem store_A_cheaper_than_store_B :
  final_price_B - final_price_A = 2 :=
by
  sorry

end store_A_cheaper_than_store_B_l127_127908


namespace sin_cos_relation_l127_127707

theorem sin_cos_relation (α : ℝ) (h : Real.tan (π / 4 + α) = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1 / 2 :=
by
  sorry

end sin_cos_relation_l127_127707


namespace non_empty_solution_set_range_l127_127863

theorem non_empty_solution_set_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| - |x - 1| < a) → a > -3 :=
sorry

end non_empty_solution_set_range_l127_127863


namespace paula_paint_cans_needed_l127_127309

-- Let's define the initial conditions and required computations in Lean.
def initial_rooms : ℕ := 48
def cans_lost : ℕ := 4
def remaining_rooms : ℕ := 36
def large_rooms_to_paint : ℕ := 8
def normal_rooms_to_paint : ℕ := 20
def paint_per_large_room : ℕ := 2 -- as each large room requires twice as much paint

-- Define a function to compute the number of cans required.
def cans_needed (initial_rooms remaining_rooms large_rooms_to_paint normal_rooms_to_paint paint_per_large_room : ℕ) : ℕ :=
  let rooms_lost := initial_rooms - remaining_rooms
  let cans_per_room := rooms_lost / cans_lost
  let total_room_equivalents := large_rooms_to_paint * paint_per_large_room + normal_rooms_to_paint
  total_room_equivalents / cans_per_room

theorem paula_paint_cans_needed : cans_needed initial_rooms remaining_rooms large_rooms_to_paint normal_rooms_to_paint paint_per_large_room = 12 :=
by
  -- The proof would go here
  sorry

end paula_paint_cans_needed_l127_127309


namespace math_problem_l127_127705

theorem math_problem 
  (m n : ℕ) 
  (h1 : (m^2 - n) ∣ (m + n^2))
  (h2 : (n^2 - m) ∣ (m^2 + n)) : 
  (m, n) = (2, 2) ∨ (m, n) = (3, 3) ∨ (m, n) = (1, 2) ∨ (m, n) = (2, 1) ∨ (m, n) = (2, 3) ∨ (m, n) = (3, 2) := 
sorry

end math_problem_l127_127705


namespace initial_amount_l127_127184

theorem initial_amount (bread_price : ℝ) (bread_qty : ℝ) (pb_price : ℝ) (leftover : ℝ) :
  bread_price = 2.25 → bread_qty = 3 → pb_price = 2 → leftover = 5.25 →
  bread_qty * bread_price + pb_price + leftover = 14 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num


end initial_amount_l127_127184


namespace problem_expression_eq_zero_l127_127378

variable {x y : ℝ}

theorem problem_expression_eq_zero (h : x * y ≠ 0) : 
    ( ( (x^2 - 1) / x ) * ( (y^2 - 1) / y ) ) - 
    ( ( (x^2 - 1) / y ) * ( (y^2 - 1) / x ) ) = 0 :=
by
  sorry

end problem_expression_eq_zero_l127_127378


namespace kim_boxes_sold_on_tuesday_l127_127875

theorem kim_boxes_sold_on_tuesday :
  ∀ (T W Th F : ℕ),
  (T = 3 * W) →
  (W = 2 * Th) →
  (Th = 3 / 2 * F) →
  (F = 600) →
  T = 5400 :=
by
  intros T W Th F h1 h2 h3 h4
  sorry

end kim_boxes_sold_on_tuesday_l127_127875


namespace wheels_in_garage_l127_127326

theorem wheels_in_garage :
  let bicycles := 9
  let cars := 16
  let single_axle_trailers := 5
  let double_axle_trailers := 3
  let wheels_per_bicycle := 2
  let wheels_per_car := 4
  let wheels_per_single_axle_trailer := 2
  let wheels_per_double_axle_trailer := 4
  let total_wheels := bicycles * wheels_per_bicycle + cars * wheels_per_car + single_axle_trailers * wheels_per_single_axle_trailer + double_axle_trailers * wheels_per_double_axle_trailer
  total_wheels = 104 := by
  sorry

end wheels_in_garage_l127_127326


namespace find_x_l127_127275

theorem find_x (a x : ℝ) (ha : 1 < a) (hx : 0 < x)
  (h : (3 * x)^(Real.log 3 / Real.log a) - (4 * x)^(Real.log 4 / Real.log a) = 0) : 
  x = 1 / 4 := 
by 
  sorry

end find_x_l127_127275


namespace reimbursement_proof_l127_127876

-- Define the rates
def rate_industrial_weekday : ℝ := 0.36
def rate_commercial_weekday : ℝ := 0.42
def rate_weekend : ℝ := 0.45

-- Define the distances for each day
def distance_monday : ℝ := 18
def distance_tuesday : ℝ := 26
def distance_wednesday : ℝ := 20
def distance_thursday : ℝ := 20
def distance_friday : ℝ := 16
def distance_saturday : ℝ := 12

-- Calculate the reimbursement for each day
def reimbursement_monday : ℝ := distance_monday * rate_industrial_weekday
def reimbursement_tuesday : ℝ := distance_tuesday * rate_commercial_weekday
def reimbursement_wednesday : ℝ := distance_wednesday * rate_industrial_weekday
def reimbursement_thursday : ℝ := distance_thursday * rate_commercial_weekday
def reimbursement_friday : ℝ := distance_friday * rate_industrial_weekday
def reimbursement_saturday : ℝ := distance_saturday * rate_weekend

-- Calculate the total reimbursement
def total_reimbursement : ℝ :=
  reimbursement_monday + reimbursement_tuesday + reimbursement_wednesday +
  reimbursement_thursday + reimbursement_friday + reimbursement_saturday

-- State the theorem to be proven
theorem reimbursement_proof : total_reimbursement = 44.16 := by
  sorry

end reimbursement_proof_l127_127876


namespace men_employed_l127_127481

theorem men_employed (M : ℕ) (W : ℕ)
  (h1 : W = M * 9)
  (h2 : W = (M + 10) * 6) : M = 20 := by
  sorry

end men_employed_l127_127481


namespace coolant_left_l127_127773

theorem coolant_left (initial_volume : ℝ) (initial_concentration : ℝ) (x : ℝ) (replacement_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 19 ∧ 
  initial_concentration = 0.30 ∧ 
  replacement_concentration = 0.80 ∧ 
  final_concentration = 0.50 ∧ 
  (0.30 * initial_volume - 0.30 * x + 0.80 * x = 0.50 * initial_volume) →
  initial_volume - x = 11.4 :=
by sorry

end coolant_left_l127_127773


namespace divisibility_by_3_l127_127265

theorem divisibility_by_3 (a b c : ℤ) (h1 : c ≠ b)
    (h2 : ∃ x : ℂ, (a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0)) :
    3 ∣ (a + b + 2 * c) :=
by
  sorry

end divisibility_by_3_l127_127265


namespace car_rent_per_day_leq_30_l127_127048

variable (D : ℝ) -- daily rental rate
variable (cost_per_mile : ℝ := 0.23) -- cost per mile
variable (daily_budget : ℝ := 76) -- daily budget
variable (distance : ℝ := 200) -- distance driven

theorem car_rent_per_day_leq_30 :
  D + cost_per_mile * distance ≤ daily_budget → D ≤ 30 :=
sorry

end car_rent_per_day_leq_30_l127_127048


namespace average_of_new_set_l127_127617

theorem average_of_new_set (s : List ℝ) (h₁ : s.length = 10) (h₂ : (s.sum / 10) = 7) : 
  ((s.map (λ x => x * 12)).sum / 10) = 84 :=
by
  sorry

end average_of_new_set_l127_127617


namespace hundred_fiftieth_digit_of_fraction_l127_127786

theorem hundred_fiftieth_digit_of_fraction :
  let repeating_block := "135"
  let decimal_pos := 150
  (decimal_pos - 1) % 3 = 2 ->
  "3" = repeating_block[(decimal_pos - 1) % 3] :=
by
  sorry

end hundred_fiftieth_digit_of_fraction_l127_127786


namespace cole_round_trip_time_l127_127940

theorem cole_round_trip_time :
  ∀ (speed_to_work speed_return : ℝ) (time_to_work_minutes : ℝ),
  speed_to_work = 75 ∧ speed_return = 105 ∧ time_to_work_minutes = 210 →
  (time_to_work_minutes / 60 + (speed_to_work * (time_to_work_minutes / 60)) / speed_return) = 6 := 
by
  sorry

end cole_round_trip_time_l127_127940


namespace find_a_l127_127321

theorem find_a 
  (a : ℝ) 
  (h1 : ∀ x : ℝ, (x - 3) ^ 2 + 5 = a * x^2 + bx + c) 
  (h2 : (3, 5) = (3, a * 3 ^ 2 + b * 3 + c))
  (h3 : (-2, -20) = (-2, a * (-2)^2 + b * (-2) + c)) : a = -1 :=
by
  sorry

end find_a_l127_127321


namespace number_of_ways_to_choose_one_book_l127_127468

-- Defining the conditions
def num_chinese_books : ℕ := 5
def num_math_books : ℕ := 4

-- Statement of the theorem
theorem number_of_ways_to_choose_one_book : num_chinese_books + num_math_books = 9 :=
by
  -- Skipping the proof as instructed
  sorry

end number_of_ways_to_choose_one_book_l127_127468


namespace expected_carrot_yield_l127_127756

-- Condition definitions
def num_steps_width : ℕ := 16
def num_steps_length : ℕ := 22
def step_length : ℝ := 1.75
def avg_yield_per_sqft : ℝ := 0.75

-- Theorem statement
theorem expected_carrot_yield : 
  (num_steps_width * step_length) * (num_steps_length * step_length) * avg_yield_per_sqft = 808.5 :=
by
  sorry

end expected_carrot_yield_l127_127756


namespace find_angle_C_l127_127433

noncomputable def angle_C_value (A B : ℝ) : ℝ :=
  180 - A - B

theorem find_angle_C (A B : ℝ) 
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) :
  angle_C_value A B = 30 :=
sorry

end find_angle_C_l127_127433


namespace function_tangent_and_max_k_l127_127846

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 2 * x - 1

theorem function_tangent_and_max_k 
  (x : ℝ) (h1 : 0 < x) 
  (h2 : 3 * x - y - 2 = 0) : 
  (∀ k : ℤ, (∀ x : ℝ, 1 < x → k < (f x) / (x - 1)) → k ≤ 4) := 
sorry

end function_tangent_and_max_k_l127_127846


namespace factorization1_factorization2_l127_127225

-- Definitions for the first problem
def expr1 (x : ℝ) := 3 * x^2 - 12
def factorized_form1 (x : ℝ) := 3 * (x + 2) * (x - 2)

-- Theorem for the first problem
theorem factorization1 (x : ℝ) : expr1 x = factorized_form1 x :=
  sorry

-- Definitions for the second problem
def expr2 (a x y : ℝ) := a * x^2 - 4 * a * x * y + 4 * a * y^2
def factorized_form2 (a x y : ℝ) := a * (x - 2 * y) * (x - 2 * y)

-- Theorem for the second problem
theorem factorization2 (a x y : ℝ) : expr2 a x y = factorized_form2 a x y :=
  sorry

end factorization1_factorization2_l127_127225


namespace division_of_exponents_l127_127183

-- Define the conditions as constants and statements that we are concerned with
variables (x : ℝ)

-- The Lean 4 statement of the equivalent proof problem
theorem division_of_exponents (h₁ : x ≠ 0) : x^8 / x^2 = x^6 := 
sorry

end division_of_exponents_l127_127183


namespace volume_of_pond_rect_prism_l127_127428

-- Define the problem as a proposition
theorem volume_of_pond_rect_prism :
  let l := 28
  let w := 10
  let h := 5
  V = l * w * h →
  V = 1400 :=
by
  intros l w h h1
  -- Here, the theorem states the equivalence of the volume given the defined length, width, and height being equal to 1400 cubic meters.
  have : V = 28 * 10 * 5 := by sorry
  exact this

end volume_of_pond_rect_prism_l127_127428


namespace number_is_divisible_by_divisor_l127_127195

-- Defining the number after replacing y with 3
def number : ℕ := 7386038

-- Defining the divisor which we need to prove 
def divisor : ℕ := 7

-- Stating the property that 7386038 is divisible by 7
theorem number_is_divisible_by_divisor : number % divisor = 0 := by
  sorry

end number_is_divisible_by_divisor_l127_127195


namespace solve_system_of_inequalities_l127_127015

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 3 ≤ x + 2) ∧ ((x + 1) / 3 > x - 1) → x ≤ -1 := by
  sorry

end solve_system_of_inequalities_l127_127015


namespace exactly_one_team_correct_l127_127659

variable (A B C : Event Ω) -- Define events for teams answering correctly
variables [probability : ℙ] -- Probability space

-- Conditions given in the problem
axiom prob_A : ℙ[A] = 3 / 4
axiom prob_B : ℙ[B] = 2 / 3
axiom prob_C : ℙ[C] = 2 / 3

-- Independence of the events
axiom indep_AB : IndepEvents A B
axiom indep_AC : IndepEvents A C
axiom indep_BC : IndepEvents B C

theorem exactly_one_team_correct :
  ℙ[(A ∧ ¬B ∧ ¬C) ∨ (¬A ∧ B ∧ ¬C) ∨ (¬A ∧ ¬B ∧ C)] = 7 / 36 :=
by
  sorry -- Detailed proof not needed

end exactly_one_team_correct_l127_127659


namespace alley_width_l127_127868

noncomputable def calculate_width (l k h : ℝ) : ℝ :=
  l / 2

theorem alley_width (k h l w : ℝ) (h1 : k = (l * (Real.sin (Real.pi / 3)))) (h2 : h = (l * (Real.sin (Real.pi / 6)))) :
  w = calculate_width l k h :=
by
  sorry

end alley_width_l127_127868


namespace repeating_decimal_to_fraction_l127_127072

theorem repeating_decimal_to_fraction : (0.7 + 23 / 99 / 10) = (62519 / 66000) := by
  sorry

end repeating_decimal_to_fraction_l127_127072


namespace distinct_four_digit_integers_with_product_18_l127_127565

theorem distinct_four_digit_integers_with_product_18 :
  ∃ n : ℕ, n = 24 ∧ ∀ (d1 d2 d3 d4 : ℕ), (d1 * d2 * d3 * d4 = 18 ∧ 1000 ≤ 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ 1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10000) →
    set.finite { x | ∃ (d1 d2 d3 d4 : ℕ), x = 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ d1 * d2 * d3 * d4 = 18 ∧ ∀ i ∈ [d1, d2, d3, d4], 1 ≤ i ∧ i ≤ 9 } :=
begin
  sorry
end

end distinct_four_digit_integers_with_product_18_l127_127565


namespace champagne_bottles_needed_l127_127207

-- Define the initial conditions of the problem
def num_guests : ℕ := 120
def glasses_per_guest : ℕ := 2
def servings_per_bottle : ℕ := 6

-- The statement we need to prove
theorem champagne_bottles_needed : 
  (num_guests * glasses_per_guest) / servings_per_bottle = 40 := 
by
  sorry

end champagne_bottles_needed_l127_127207


namespace day_of_20th_is_Thursday_l127_127021

noncomputable def day_of_week (d : ℕ) : String :=
  match d % 7 with
  | 0 => "Saturday"
  | 1 => "Sunday"
  | 2 => "Monday"
  | 3 => "Tuesday"
  | 4 => "Wednesday"
  | 5 => "Thursday"
  | 6 => "Friday"
  | _ => "Unknown"

theorem day_of_20th_is_Thursday (s1 s2 s3: ℕ) (h1: 2 ≤ s1) (h2: s1 ≤ 30) (h3: s2 = s1 + 14) (h4: s3 = s2 + 14) (h5: s3 ≤ 30) (h6: day_of_week s1 = "Sunday") : 
  day_of_week 20 = "Thursday" :=
by
  sorry

end day_of_20th_is_Thursday_l127_127021


namespace num_valid_lists_l127_127441

-- Define a predicate for a list to satisfy the given constraints
def valid_list (l : List ℕ) : Prop :=
  l = List.range' 1 12 ∧ ∀ i, 1 < i ∧ i ≤ 12 → (l.indexOf (l.get! (i - 1) + 1) < i - 1 ∨ l.indexOf (l.get! (i - 1) - 1) < i - 1) ∧ ¬(l.indexOf (l.get! (i - 1) + 1) < i - 1 ∧ l.indexOf (l.get! (i - 1) - 1) < i - 1)

-- Prove that there is exactly one valid list of such nature
theorem num_valid_lists : ∃! l : List ℕ, valid_list l :=
  sorry

end num_valid_lists_l127_127441


namespace value_of_2_pow_5_plus_5_l127_127776

theorem value_of_2_pow_5_plus_5 : 2^5 + 5 = 37 := by
  sorry

end value_of_2_pow_5_plus_5_l127_127776


namespace team_includes_john_peter_mary_prob_l127_127585

/-- In a group of 12 players where John, Peter, and Mary are among them, 
if a coach randomly selects a 6-player team, 
the probability of choosing a team that includes John, Peter, and Mary is 1/11. -/
def probability_team_includes_john_peter_mary (total_players : ℕ) (team_size : ℕ) (selected_players : ℕ) : ℚ :=
  if h1 : total_players = 12 ∧ team_size = 6 ∧ selected_players = 3 then
    let ways_to_choose := Nat.choose 9 3 in
    let total_ways := Nat.choose 12 6 in
    ways_to_choose / total_ways
  else 
    0

theorem team_includes_john_peter_mary_prob :
  probability_team_includes_john_peter_mary 12 6 3 = 1 / 11 :=
by 
  simp [probability_team_includes_john_peter_mary, h1]
  sorry

end team_includes_john_peter_mary_prob_l127_127585


namespace range_of_a_minus_b_l127_127412

theorem range_of_a_minus_b {a b : ℝ} (h1 : -2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) : 
  -4 < a - b ∧ a - b < 2 := 
by
  sorry

end range_of_a_minus_b_l127_127412


namespace find_number_l127_127649

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 8) : x = 4 :=
by
  sorry

end find_number_l127_127649


namespace sum_reciprocals_five_l127_127902

theorem sum_reciprocals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : 
  1/x + 1/y = 5 :=
begin
  sorry
end

end sum_reciprocals_five_l127_127902


namespace inequality_of_weighted_squares_equality_conditions_of_weighted_squares_l127_127383

theorem inequality_of_weighted_squares
  (x y a b : ℝ)
  (h_sum : a + b = 1)
  (h_nonneg_a : a ≥ 0)
  (h_nonneg_b : b ≥ 0) :
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 :=
by sorry

theorem equality_conditions_of_weighted_squares
  (x y a b : ℝ)
  (h_sum : a + b = 1)
  (h_nonneg_a : a ≥ 0)
  (h_nonneg_b : b ≥ 0) :
  (a * x + b * y)^2 = a * x^2 + b * y^2
  ↔ (a = 0 ∨ b = 0 ∨ x = y) :=
by sorry

end inequality_of_weighted_squares_equality_conditions_of_weighted_squares_l127_127383


namespace smaller_circle_radius_l127_127495

open Real

def is_geometric_progression (a b c : ℝ) : Prop :=
  (b / a = c / b)

theorem smaller_circle_radius 
  (B1 B2 : ℝ) 
  (r2 : ℝ) 
  (h1 : B1 + B2 = π * r2^2) 
  (h2 : r2 = 5) 
  (h3 : is_geometric_progression B1 B2 (B1 + B2)) :
  sqrt ((-1 + sqrt (1 + 100 * π)) / (2 * π)) = sqrt (B1 / π) :=
by
  sorry

end smaller_circle_radius_l127_127495


namespace possible_values_of_a_l127_127722

def A (a : ℤ) : Set ℤ := {2, 4, a^3 - 2 * a^2 - a + 7}
def B (a : ℤ) : Set ℤ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem possible_values_of_a (a : ℤ) :
  A a ∩ B a = {2, 5} ↔ a = -1 ∨ a = 2 :=
by
  sorry

end possible_values_of_a_l127_127722


namespace profit_share_difference_correct_l127_127044

noncomputable def profit_share_difference (a_capital b_capital c_capital b_profit : ℕ) : ℕ :=
  let total_parts := 4 + 5 + 6
  let part_size := b_profit / 5
  let a_profit := 4 * part_size
  let c_profit := 6 * part_size
  c_profit - a_profit

theorem profit_share_difference_correct :
  profit_share_difference 8000 10000 12000 1600 = 640 :=
by
  sorry

end profit_share_difference_correct_l127_127044


namespace chloe_profit_l127_127374

theorem chloe_profit 
  (cost_per_dozen : ℕ)
  (selling_price_per_half_dozen : ℕ)
  (dozens_sold : ℕ)
  (h1 : cost_per_dozen = 50)
  (h2 : selling_price_per_half_dozen = 30)
  (h3 : dozens_sold = 50) : 
  (selling_price_per_half_dozen - cost_per_dozen / 2) * (dozens_sold * 2) = 500 :=
by 
  sorry

end chloe_profit_l127_127374


namespace probability_two_blue_l127_127189

-- Define the conditions of the problem
def total_balls : ℕ := 15
def blue_balls : ℕ := 5
def red_balls : ℕ := 10
def balls_drawn : ℕ := 6
def blue_needed : ℕ := 2
def red_needed : ℕ := 4

-- Calculate the total number of ways to choose 6 balls out of 15
def total_outcomes : ℕ := Nat.choose total_balls balls_drawn

-- Calculate the number of favorable outcomes (2 blue, 4 red)
def blue_combinations : ℕ := Nat.choose blue_balls blue_needed
def red_combinations : ℕ := Nat.choose red_balls red_needed
def favorable_outcomes : ℕ := blue_combinations * red_combinations

-- Calculate the probability of 2 blue balls out of 6 drawn
def probability : ℚ := favorable_outcomes /. total_outcomes

theorem probability_two_blue :
  probability = 2100 /. 5005 := by
  sorry

end probability_two_blue_l127_127189


namespace evaluate_expression_l127_127413

variables (a b c d m : ℝ)

lemma opposite_is_zero (h1 : a + b = 0) : a + b = 0 := h1

lemma reciprocals_equal_one (h2 : c * d = 1) : c * d = 1 := h2

lemma abs_value_two (h3 : |m| = 2) : |m| = 2 := h3

theorem evaluate_expression (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 2) :
  m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 :=
by
  sorry

end evaluate_expression_l127_127413


namespace num_comfortable_butterflies_final_state_l127_127604

noncomputable def num_comfortable_butterflies (n : ℕ) : ℕ :=
  if h : 0 < n then
    n
  else
    0

theorem num_comfortable_butterflies_final_state {n : ℕ} (h : 0 < n):
  num_comfortable_butterflies n = n := by
  sorry

end num_comfortable_butterflies_final_state_l127_127604


namespace distinct_four_digit_positive_integers_product_18_l127_127559

theorem distinct_four_digit_positive_integers_product_18 :
  Finset.card {n | ∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
                             (1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9) ∧
                             a * b * c * d = 18} = 24 :=
by
  sorry

end distinct_four_digit_positive_integers_product_18_l127_127559


namespace candies_division_l127_127758

theorem candies_division :
  let nellie_eats := 12
  let jacob_eats := nellie_eats / 2
  let lana_eats := jacob_eats - 3
  let total_candies := 30
  let total_eaten := nellie_eats + jacob_eats + lana_eats
  let remaining_candies := total_candies - total_eaten
  let each_gets := remaining_candies / 3
  in each_gets = 3 :=
by
  let nellie_eats := 12
  let jacob_eats := nellie_eats / 2
  let lana_eats := jacob_eats - 3
  let total_candies := 30
  let total_eaten := nellie_eats + jacob_eats + lana_eats
  let remaining_candies := total_candies - total_eaten
  let each_gets := remaining_candies / 3
  show each_gets = 3
  sorry

end candies_division_l127_127758


namespace students_wanted_fruit_l127_127626

theorem students_wanted_fruit (red_apples green_apples extra_fruit : ℕ)
  (h_red : red_apples = 42)
  (h_green : green_apples = 7)
  (h_extra : extra_fruit = 40) :
  red_apples + green_apples + extra_fruit - (red_apples + green_apples) = 40 :=
by
  sorry

end students_wanted_fruit_l127_127626


namespace percent_decrease_to_original_price_l127_127671

variable (x : ℝ) (p : ℝ)

def new_price (x : ℝ) : ℝ := 1.35 * x

theorem percent_decrease_to_original_price :
  ∀ (x : ℝ), x ≠ 0 → (1 - (7 / 27)) * (new_price x) = x := 
sorry

end percent_decrease_to_original_price_l127_127671


namespace function_neither_even_nor_odd_l127_127592

noncomputable def f (x : ℝ) : ℝ := (4 * x ^ 3 - 3) / (x ^ 6 + 2)

theorem function_neither_even_nor_odd : 
  (∀ x : ℝ, f (-x) ≠ f x) ∧ (∀ x : ℝ, f (-x) ≠ -f x) :=
by
  sorry

end function_neither_even_nor_odd_l127_127592


namespace alpha_beta_sum_l127_127329

theorem alpha_beta_sum (α β : ℝ) (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 102 * x + 2021) / (x^2 + 89 * x - 3960)) : α + β = 176 := by
  sorry

end alpha_beta_sum_l127_127329


namespace range_a_le_2_l127_127579
-- Import everything from Mathlib

-- Define the hypothesis and the conclusion in Lean 4
theorem range_a_le_2 (a : ℝ) : 
  (∀ x > 0, Real.log x + a * x + 1 - x * Real.exp (2 * x) ≤ 0) ↔ a ≤ 2 := 
sorry

end range_a_le_2_l127_127579


namespace cost_price_of_one_toy_l127_127499

-- Definitions translating the conditions into Lean
def total_revenue (toys_sold : ℕ) (price_per_toy : ℕ) : ℕ := toys_sold * price_per_toy
def gain (cost_per_toy : ℕ) (toys_gained : ℕ) : ℕ := cost_per_toy * toys_gained

-- Given the conditions in the problem
def total_cost_price_of_sold_toys := 18 * (1300 : ℕ)
def gain_from_sale := 3 * (1300 : ℕ)
def selling_price := total_cost_price_of_sold_toys + gain_from_sale

-- The target theorem we want to prove
theorem cost_price_of_one_toy : (selling_price = 27300) → (1300 = 27300 / 21) :=
by
  intro h
  sorry

end cost_price_of_one_toy_l127_127499


namespace truck_capacity_rental_plan_l127_127493

-- Define the variables for the number of boxes each type of truck can carry
variables {x y : ℕ}

-- Define the conditions for the number of boxes carried by trucks
axiom cond1 : 15 * x + 25 * y = 750
axiom cond2 : 10 * x + 30 * y = 700

-- Problem 1: Prove x = 25 and y = 15
theorem truck_capacity : x = 25 ∧ y = 15 :=
by
  sorry

-- Define the variables for the number of each type of truck
variables {m : ℕ}

-- Define the conditions for the total number of trucks and boxes to be carried
axiom cond3 : 25 * m + 15 * (70 - m) ≤ 1245
axiom cond4 : 70 - m ≤ 3 * m

-- Problem 2: Prove there is one valid rental plan with m = 18 and 70-m = 52
theorem rental_plan : 17 ≤ m ∧ m ≤ 19 ∧ 70 - m ≤ 3 * m ∧ (70-m = 52 → m = 18) :=
by
  sorry

end truck_capacity_rental_plan_l127_127493


namespace distinct_four_digit_integers_with_product_18_l127_127563

theorem distinct_four_digit_integers_with_product_18 :
  ∃ n : ℕ, n = 24 ∧ ∀ (d1 d2 d3 d4 : ℕ), (d1 * d2 * d3 * d4 = 18 ∧ 1000 ≤ 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ 1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10000) →
    set.finite { x | ∃ (d1 d2 d3 d4 : ℕ), x = 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ d1 * d2 * d3 * d4 = 18 ∧ ∀ i ∈ [d1, d2, d3, d4], 1 ≤ i ∧ i ≤ 9 } :=
begin
  sorry
end

end distinct_four_digit_integers_with_product_18_l127_127563


namespace similar_triangle_angles_l127_127427

theorem similar_triangle_angles (α β γ : ℝ) (h1 : α + β + γ = Real.pi) (h2 : α + β/2 + γ/2 = Real.pi):
  ∃ (k : ℝ), α = k ∧ β = 2 * k ∧ γ = 4 * k ∧ k = Real.pi / 7 := 
sorry

end similar_triangle_angles_l127_127427


namespace find_flour_amount_l127_127606

variables (F S C : ℕ)

-- Condition 1: Proportions must remain constant
axiom proportion : 11 * S = 7 * F ∧ 7 * C = 5 * S

-- Condition 2: Mary needs 2 more cups of flour than sugar
axiom flour_sugar : F = S + 2

-- Condition 3: Mary needs 1 more cup of sugar than cocoa powder
axiom sugar_cocoa : S = C + 1

-- Question: How many cups of flour did she put in?
theorem find_flour_amount : F = 8 :=
by
  sorry

end find_flour_amount_l127_127606


namespace total_messages_sent_l127_127675

theorem total_messages_sent 
    (lucia_day1 : ℕ)
    (alina_day1_less : ℕ)
    (lucia_day1_messages : lucia_day1 = 120)
    (alina_day1_messages : alina_day1_less = 20)
    : (lucia_day2 : ℕ)
    (alina_day2 : ℕ)
    (lucia_day2_eq : lucia_day2 = lucia_day1 / 3)
    (alina_day2_eq : alina_day2 = (lucia_day1 - alina_day1_less) * 2)
    (messages_day3_eq : ∀ (lucia_day3 alina_day3 : ℕ), lucia_day3 + alina_day3 = lucia_day1 + (lucia_day1 - alina_day1_less))
    : lucia_day1 + alina_day1_less + (lucia_day2 + alina_day2) + messages_day3_eq 120 100 = 680 :=
    sorry

end total_messages_sent_l127_127675


namespace general_formula_sequence_l127_127249

theorem general_formula_sequence (a : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h_rec : ∀ n : ℕ, n > 0 → a (n + 1) = 4 * a n + 3) :
  ∀ n : ℕ, n > 0 → a n = 4^n - 1 :=
by 
  sorry

end general_formula_sequence_l127_127249


namespace count_1000_pointed_stars_l127_127941

/--
A regular n-pointed star is defined by:
1. The points P_1, P_2, ..., P_n are coplanar and no three of them are collinear.
2. Each of the n line segments intersects at least one other segment at a point other than an endpoint.
3. All of the angles at P_1, P_2, ..., P_n are congruent.
4. All of the n line segments P_2P_3, ..., P_nP_1 are congruent.
5. The path P_1P_2, P_2P_3, ..., P_nP_1 turns counterclockwise at an angle of less than 180 degrees at each vertex.

There are no regular 3-pointed, 4-pointed, or 6-pointed stars.
All regular 5-pointed stars are similar.
There are two non-similar regular 7-pointed stars.

Prove that the number of non-similar regular 1000-pointed stars is 199.
-/
theorem count_1000_pointed_stars : ∀ (n : ℕ), n = 1000 → 
  -- Points P_1, P_2, ..., P_1000 are coplanar, no three are collinear.
  -- Each of the 1000 segments intersects at least one other segment not at an endpoint.
  -- Angles at P_1, P_2, ..., P_1000 are congruent.
  -- Line segments P_2P_3, ..., P_1000P_1 are congruent.
  -- Path P_1P_2, P_2P_3, ..., P_1000P_1 turns counterclockwise at < 180 degrees each.
  -- No 3-pointed, 4-pointed, or 6-pointed regular stars.
  -- All regular 5-pointed stars are similar.
  -- There are two non-similar regular 7-pointed stars.
  -- Proven: The number of non-similar regular 1000-pointed stars is 199.
  n = 1000 ∧ (∀ m : ℕ, 1 ≤ m ∧ m < 1000 → (gcd m 1000 = 1 → (m ≠ 1 ∧ m ≠ 999))) → 
    -- Because 1000 = 2^3 * 5^3 and we exclude 1 and 999.
    (2 * 5 * 2 * 5 * 2 * 5) / 2 - 1 - 1 / 2 = 199 :=
by
  -- Pseudo-proof steps for the problem.
  sorry

end count_1000_pointed_stars_l127_127941


namespace find_P_coordinates_l127_127533

-- Define points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -3)

-- Define the theorem
theorem find_P_coordinates :
  ∃ P : ℝ × ℝ, P = (8, -15) ∧ (P.1 - A.1, P.2 - A.2) = (3 * (B.1 - A.1), 3 * (B.2 - A.2)) :=
sorry

end find_P_coordinates_l127_127533


namespace monotonicity_and_range_of_a_l127_127716

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * x + a * Real.log x

theorem monotonicity_and_range_of_a (a : ℝ) (t : ℝ) (ht : t ≥ 1) :
  (∀ x, x > 0 → f x a ≥ f t a - 3) → a ≤ 2 := 
sorry

end monotonicity_and_range_of_a_l127_127716


namespace area_of_circle_l127_127334

theorem area_of_circle (x y : ℝ) :
  x^2 + y^2 - 4*x - 6*y = -3 →
  ∃ A : ℝ, A = 10 * Real.pi :=
by
  intro h
  sorry

end area_of_circle_l127_127334


namespace sum_of_x_values_l127_127286

noncomputable def arithmetic_angles_triangle (x : ℝ) : Prop :=
  let α := 30 * Real.pi / 180
  let β := (30 + 40) * Real.pi / 180
  let γ := (30 + 80) * Real.pi / 180
  (x = 6) ∨ (x = 8) ∨ (x = (7 + Real.sqrt 36 + Real.sqrt 83))

theorem sum_of_x_values : ∀ x : ℝ, 
  arithmetic_angles_triangle x → 
  (∃ p q r : ℝ, x = p + Real.sqrt q + Real.sqrt r ∧ p = 7 ∧ q = 36 ∧ r = 83) := 
by
  sorry

end sum_of_x_values_l127_127286


namespace carter_family_children_l127_127890

variable (f m x y : ℕ)

theorem carter_family_children 
  (avg_family : (3 * y + m + x * y) / (2 + x) = 25)
  (avg_mother_children : (m + x * y) / (1 + x) = 18)
  (father_age : f = 3 * y)
  (simplest_case : y = x) :
  x = 8 :=
by
  -- Proof to be provided
  sorry

end carter_family_children_l127_127890


namespace total_distance_swam_l127_127883

theorem total_distance_swam (molly_swam_saturday : ℕ) (molly_swam_sunday : ℕ) (h1 : molly_swam_saturday = 400) (h2 : molly_swam_sunday = 300) : molly_swam_saturday + molly_swam_sunday = 700 := by 
    sorry

end total_distance_swam_l127_127883


namespace dance_team_members_l127_127366

theorem dance_team_members (a b c : ℕ)
  (h1 : a + b + c = 100)
  (h2 : b = 2 * a)
  (h3 : c = 2 * a + 10) :
  c = 46 := by
  sorry

end dance_team_members_l127_127366


namespace joshua_share_is_30_l127_127874

-- Definitions based on the conditions
def total_amount_shared : ℝ := 40
def ratio_joshua_justin : ℝ := 3

-- Proposition to prove
theorem joshua_share_is_30 (J : ℝ) (Joshua_share : ℝ) :
  J + ratio_joshua_justin * J = total_amount_shared → 
  Joshua_share = ratio_joshua_justin * J → 
  Joshua_share = 30 :=
sorry

end joshua_share_is_30_l127_127874


namespace water_volume_per_minute_l127_127918

theorem water_volume_per_minute (depth width : ℝ) (flow_rate_kmph : ℝ) 
  (H_depth : depth = 5) 
  (H_width : width = 35) 
  (H_flow_rate_kmph : flow_rate_kmph = 2) : 
  (depth * width * (flow_rate_kmph * 1000 / 60)) = 5832.75 :=
by
  sorry

end water_volume_per_minute_l127_127918


namespace smallest_possible_other_integer_l127_127897

theorem smallest_possible_other_integer (x m n : ℕ) (h1 : x > 0) (h2 : m = 70) 
  (h3 : gcd m n = x + 7) (h4 : lcm m n = x * (x + 7)) : n = 20 :=
sorry

end smallest_possible_other_integer_l127_127897


namespace restaurant_sales_l127_127668

theorem restaurant_sales :
  let meals_sold_8 := 10
  let price_per_meal_8 := 8
  let meals_sold_10 := 5
  let price_per_meal_10 := 10
  let meals_sold_4 := 20
  let price_per_meal_4 := 4
  let total_sales := meals_sold_8 * price_per_meal_8 + meals_sold_10 * price_per_meal_10 + meals_sold_4 * price_per_meal_4
  total_sales = 210 :=
by
  sorry

end restaurant_sales_l127_127668


namespace solve_floor_equation_l127_127830

theorem solve_floor_equation (x : ℝ) :
  (⌊⌊2 * x⌋ - 1 / 2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
sorry

end solve_floor_equation_l127_127830


namespace cut_scene_length_proof_l127_127358

noncomputable def original_length : ℕ := 60
noncomputable def final_length : ℕ := 57
noncomputable def cut_scene_length := original_length - final_length

theorem cut_scene_length_proof : cut_scene_length = 3 := by
  sorry

end cut_scene_length_proof_l127_127358


namespace find_m_if_divisible_by_11_l127_127421

theorem find_m_if_divisible_by_11 : ∃ m : ℕ, m < 10 ∧ (734000000 + m*100000 + 8527) % 11 = 0 ↔ m = 6 :=
by {
    sorry
}

end find_m_if_divisible_by_11_l127_127421


namespace battery_lasts_12_hours_more_l127_127443

-- Define the battery consumption rates
def standby_consumption_rate : ℚ := 1 / 36
def active_consumption_rate : ℚ := 1 / 4

-- Define the usage times
def total_time_hours : ℚ := 12
def active_use_time_hours : ℚ := 1.5
def standby_time_hours : ℚ := total_time_hours - active_use_time_hours

-- Define the total battery used during standby and active use
def standby_battery_used : ℚ := standby_time_hours * standby_consumption_rate
def active_battery_used : ℚ := active_use_time_hours * active_consumption_rate
def total_battery_used : ℚ := standby_battery_used + active_battery_used

-- Define the remaining battery
def remaining_battery : ℚ := 1 - total_battery_used

-- Define how long the remaining battery will last on standby
def remaining_standby_time : ℚ := remaining_battery / standby_consumption_rate

-- Theorem stating the correct answer
theorem battery_lasts_12_hours_more :
  remaining_standby_time = 12 := 
sorry

end battery_lasts_12_hours_more_l127_127443


namespace distance_between_circle_centers_l127_127320

open Real

theorem distance_between_circle_centers :
  let center1 := (1 / 2, 0)
  let center2 := (0, 1 / 2)
  dist center1 center2 = sqrt 2 / 2 :=
by
  sorry

end distance_between_circle_centers_l127_127320


namespace sum_division_l127_127927

theorem sum_division (x y z : ℝ) (total_share_y : ℝ) 
  (Hx : x = 1) 
  (Hy : y = 0.45) 
  (Hz : z = 0.30) 
  (share_y : total_share_y = 36) 
  : (x + y + z) * (total_share_y / y) = 140 := by
  sorry

end sum_division_l127_127927


namespace A_n_is_interval_limit_a_n_l127_127988

open Real

noncomputable def A_n (n : ℕ) (Hn : n ≥ 2) : Set ℝ :=
  {s | ∃ (x : Fin n → ℝ), (∀ k, x k ∈ Icc 0 1) ∧ (∑ i, x i = 1) ∧ (s = ∑ i, arcsin (x i)) }

def a_n (n : ℕ) (Hn : n ≥ 2) : ℝ :=
  (Icc (n * arcsin (1 / n.toReal)) (π / 2)).toReal

theorem A_n_is_interval (n : ℕ) (Hn : n ≥ 2) : ∃ l u, A_n n Hn = Icc l u :=
sorry

theorem limit_a_n : Tendsto (fun n => a_n n (by decide)) atTop (𝓝 (π / 2 - 1)) :=
sorry

end A_n_is_interval_limit_a_n_l127_127988


namespace expected_difference_l127_127812

noncomputable def fair_eight_sided_die := [2, 3, 4, 5, 6, 7, 8]

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_composite (n : ℕ) : Prop := 
  n = 4 ∨ n = 6 ∨ n = 8

def unsweetened_cereal_days := (4 / 7) * 365
def sweetened_cereal_days := (3 / 7) * 365

theorem expected_difference :
  unsweetened_cereal_days - sweetened_cereal_days = 53 := by
  sorry

end expected_difference_l127_127812


namespace boat_goes_6_km_upstream_l127_127288

variable (speed_in_still_water : ℕ) (distance_downstream : ℕ) (time_downstream : ℕ) (effective_speed_downstream : ℕ) (speed_of_stream : ℕ)

-- Given conditions
def condition1 : Prop := speed_in_still_water = 11
def condition2 : Prop := distance_downstream = 16
def condition3 : Prop := time_downstream = 1
def condition4 : Prop := effective_speed_downstream = speed_in_still_water + speed_of_stream
def condition5 : Prop := effective_speed_downstream = 16

-- Prove that the boat goes 6 km against the stream in one hour.
theorem boat_goes_6_km_upstream : speed_of_stream = 5 →
  11 - 5 = 6 :=
by
  intros
  sorry

end boat_goes_6_km_upstream_l127_127288


namespace proof_one_third_of_seven_times_nine_subtract_three_l127_127231

def one_third_of_seven_times_nine_subtract_three : ℕ :=
  let product := 7 * 9
  let one_third := product / 3
  one_third - 3

theorem proof_one_third_of_seven_times_nine_subtract_three : one_third_of_seven_times_nine_subtract_three = 18 := by
  sorry

end proof_one_third_of_seven_times_nine_subtract_three_l127_127231


namespace sum_of_all_elements_in_T_binary_l127_127991

def T : Set ℕ := { n | ∃ a b c d : Bool, n = (1 * 2^4) + (a.toNat * 2^3) + (b.toNat * 2^2) + (c.toNat * 2^1) + d.toNat }

theorem sum_of_all_elements_in_T_binary :
  (∑ n in T, n) = 0b1001110000 :=
by
  sorry

end sum_of_all_elements_in_T_binary_l127_127991


namespace store_a_cheaper_by_two_l127_127907

def store_a_price : ℝ := 125
def store_b_price : ℝ := 130
def store_a_discount : ℝ := 0.08
def store_b_discount : ℝ := 0.10

def final_price_store_a : ℝ := store_a_price - (store_a_price * store_a_discount)
def final_price_store_b : ℝ := store_b_price - (store_b_price * store_b_discount)

theorem store_a_cheaper_by_two :
  final_price_store_b - final_price_store_a = 2 :=
by
  unfold final_price_store_b final_price_store_a store_a_price store_b_price store_a_discount store_b_discount
  have h₁ : store_b_price - store_b_price * store_b_discount = 117 := by norm_num
  have h₂ : store_a_price - store_a_price * store_a_discount = 115 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end store_a_cheaper_by_two_l127_127907


namespace number_of_rational_solutions_l127_127323

namespace RationalSolutions

def system_of_equations (x y z : ℚ) : Prop :=
  x + y + z = 0 ∧
  xyz + z = 0 ∧
  xy + yz + xz + y = 0

theorem number_of_rational_solutions : 
  (∃ x y z : ℚ, system_of_equations x y z) ∧ 
  ∀ (x₁ y₁ z₁) (x₂ y₂ z₂ : ℚ), 
    system_of_equations x₁ y₁ z₁ →
    system_of_equations x₂ y₂ z₂ →
    (x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂) ↔ 
    (x₁, y₁, z₁) = (0, 0, 0) ∨ (x₁, y₁, z₁) = (-1, 1, 0) ∨ 
    (x₂, y₂, z₂) = (0, 0, 0) ∨ (x₂, y₂, z₂) = (-1, 1, 0)
:=
begin
  sorry
end

end RationalSolutions

end number_of_rational_solutions_l127_127323


namespace montague_fraction_l127_127429

noncomputable def fraction_montague (M C : ℝ) : Prop :=
  M + C = 1 ∧
  (0.70 * C) / (0.20 * M + 0.70 * C) = 7 / 11

theorem montague_fraction : ∃ M C : ℝ, fraction_montague M C ∧ M = 2 / 3 :=
by sorry

end montague_fraction_l127_127429


namespace range_of_2a_plus_3b_inequality_between_expressions_l127_127097

-- First proof problem
theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) (h3 : -1 ≤ a - b) (h4 : a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
sorry

-- Second proof problem
theorem inequality_between_expressions (a b c : ℝ) (h : a^2 + b^2 + c^2 = 6) :
  (1 / (a^2 + 1) + 1 / (b^2 + 2)) > (1 / 2 - 1 / (c^2 + 3)) :=
sorry

end range_of_2a_plus_3b_inequality_between_expressions_l127_127097


namespace longest_diagonal_length_l127_127357

-- Defining conditions
variable (d1 d2 : ℝ)
variable (x : ℝ)
variable (area : ℝ)
variable (h_area : area = 144)
variable (h_ratio : d1 = 4 * x)
variable (h_ratio' : d2 = 3 * x)
variable (h_area_eq : area = 1 / 2 * d1 * d2)

-- The Lean statement, asserting the length of the longest diagonal is 8 * sqrt(6)
theorem longest_diagonal_length (x : ℝ) (h_area : 1 / 2 * (4 * x) * (3 * x) = 144) :
  4 * x = 8 * Real.sqrt 6 := by
sorry

end longest_diagonal_length_l127_127357


namespace cone_lsa_15pi_l127_127028

variable (r l : ℝ)

def cone_lateral_surface_area (r l : ℝ) : ℝ :=
  Real.pi * r * l

theorem cone_lsa_15pi (h₁ : r = 3) (h₂ : l = 5) :
  cone_lateral_surface_area r l = 15 * Real.pi := by
  rw [h₁, h₂, cone_lateral_surface_area]
  sorry

end cone_lsa_15pi_l127_127028


namespace simplify_expression_l127_127687

variable {R : Type*} [CommRing R]

theorem simplify_expression (x y : R) : (x + 2 * y) * (x - 2 * y) - y * (3 - 4 * y) = x^2 - 3 * y :=
by
  sorry

end simplify_expression_l127_127687


namespace isosceles_triangle_vertex_angle_l127_127586

theorem isosceles_triangle_vertex_angle (θ : ℝ) (h₀ : θ = 80) (h₁ : ∃ (x y z : ℝ), (x = y ∨ y = z ∨ z = x) ∧ x + y + z = 180) : θ = 80 ∨ θ = 20 := 
sorry

end isosceles_triangle_vertex_angle_l127_127586


namespace scientific_notation_of_393000_l127_127892

theorem scientific_notation_of_393000 :
  ∃ a n : ℝ, (1 ≤ a ∧ a < 10) ∧ n ∈ ℤ ∧ 393000 = a * 10^n ∧ a = 3.93 ∧ n = 5 :=
by
  use 3.93
  use 5
  split
  { split
    { norm_num }
    { norm_num } }
  split
  { norm_num }
  split
  { exact rfl }
  split
  { exact rfl }
  { exact rfl }
  sorry

end scientific_notation_of_393000_l127_127892


namespace binom_20_10_l127_127543

-- Definitions for the provided conditions
def binom_18_8 := 43758
def binom_18_9 := 48620
def binom_18_10 := 43758

-- The theorem we need to prove
theorem binom_20_10 : ∀
  (binom_18_8 = 43758)
  (binom_18_9 = 48620)
  (binom_18_10 = 43758),
  binomial 20 10 = 184756 :=
by
  sorry

end binom_20_10_l127_127543


namespace shaded_region_area_l127_127683

noncomputable def shaded_area (π_approx : ℝ := 3.14) (r : ℝ := 1) : ℝ :=
  let square_area := (r / Real.sqrt 2) ^ 2
  let quarter_circle_area := (π_approx * r ^ 2) / 4
  quarter_circle_area - square_area

theorem shaded_region_area :
  shaded_area = 0.285 :=
by
  sorry

end shaded_region_area_l127_127683


namespace hexagon_rotation_angle_l127_127796

theorem hexagon_rotation_angle (θ : ℕ) : θ = 90 → ¬ ∃ k, k * 60 = θ ∨ θ = 360 :=
by
  sorry

end hexagon_rotation_angle_l127_127796


namespace vector_BC_correct_l127_127113

-- Define the conditions
def vector_AB : ℝ × ℝ := (-3, 2)
def vector_AC : ℝ × ℝ := (1, -2)

-- Define the problem to be proved
theorem vector_BC_correct :
  let vector_BC := (vector_AC.1 - vector_AB.1, vector_AC.2 - vector_AB.2)
  vector_BC = (4, -4) :=
by
  sorry -- The proof is not required, but the structure indicates where it would go

end vector_BC_correct_l127_127113


namespace chandler_bike_purchase_l127_127065

theorem chandler_bike_purchase : 
    ∀ (x : ℕ), (200 + 20 * x = 800) → (x = 30) :=
by
  intros x h
  sorry

end chandler_bike_purchase_l127_127065


namespace smallest_b_for_perfect_square_l127_127338

theorem smallest_b_for_perfect_square : ∃ b : ℕ, b > 5 ∧ (∃ n : ℕ, 4 * b + 5 = n^2) ∧ ∀ b' : ℕ, b' > 5 → (∃ n' : ℕ, 4 * b' + 5 = n'^2) → b ≤ b' :=
by { use 11, split, linarith, split, use 7, norm_num, intros b' hb' hb'n', rcases hb'n' with ⟨n', hn'⟩, linarith }

end smallest_b_for_perfect_square_l127_127338


namespace trapezium_area_l127_127519

theorem trapezium_area (a b h : ℝ) (h₁ : a = 20) (h₂ : b = 16) (h₃ : h = 15) : 
  (1/2 * (a + b) * h = 270) :=
by
  rw [h₁, h₂, h₃]
  -- The following lines of code are omitted as they serve as solving this proof, and the requirement is to provide the statement only. 
  sorry

end trapezium_area_l127_127519


namespace final_price_correct_l127_127199

open BigOperators

-- Define the constants used in the problem
def original_price : ℝ := 500
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.10
def state_tax : ℝ := 0.05

-- Define the calculation steps
def price_after_first_discount : ℝ := original_price * (1 - first_discount)
def price_after_second_discount : ℝ := price_after_first_discount * (1 - second_discount)
def final_price : ℝ := price_after_second_discount * (1 + state_tax)

-- Prove that the final price is 354.375
theorem final_price_correct : final_price = 354.375 :=
by
  sorry

end final_price_correct_l127_127199


namespace squirrel_pine_cones_l127_127486

theorem squirrel_pine_cones (x y : ℕ) (hx : 26 - 10 + 9 + (x + 14)/2 = x/2) (hy : y + 5 - 18 + 9 + (x + 14)/2 = x/2) :
  x = 86 := sorry

end squirrel_pine_cones_l127_127486


namespace K_3_15_10_eq_151_30_l127_127245

def K (a b c : ℕ) : ℚ := (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a

theorem K_3_15_10_eq_151_30 : K 3 15 10 = 151 / 30 := 
by
  sorry

end K_3_15_10_eq_151_30_l127_127245


namespace lisa_eats_one_candy_on_other_days_l127_127444

def candies_total : ℕ := 36
def candies_per_day_on_mondays_and_wednesdays : ℕ := 2
def weeks : ℕ := 4
def days_in_a_week : ℕ := 7
def mondays_and_wednesdays_in_4_weeks : ℕ := 2 * weeks
def total_candies_mondays_and_wednesdays : ℕ := mondays_and_wednesdays_in_4_weeks * candies_per_day_on_mondays_and_wednesdays
def total_other_candies : ℕ := candies_total - total_candies_mondays_and_wednesdays
def total_other_days : ℕ := weeks * (days_in_a_week - 2)
def candies_per_other_day : ℕ := total_other_candies / total_other_days

theorem lisa_eats_one_candy_on_other_days :
  candies_per_other_day = 1 :=
by
  -- Prove the theorem with conditions defined
  sorry

end lisa_eats_one_candy_on_other_days_l127_127444


namespace sum_of_angles_is_990_l127_127223

noncomputable def z₁ : ℂ := 2 * complex.exp (complex.I * real.pi * (54 / 180))
noncomputable def z₂ : ℂ := 2 * complex.exp (complex.I * real.pi * (126 / 180))
noncomputable def z₃ : ℂ := 2 * complex.exp (complex.I * real.pi * (198 / 180))
noncomputable def z₄ : ℂ := 2 * complex.exp (complex.I * real.pi * (270 / 180))
noncomputable def z₅ : ℂ := 2 * complex.exp (complex.I * real.pi * (342 / 180))

theorem sum_of_angles_is_990 :
  (54 + 126 + 198 + 270 + 342 : ℝ) = 990 :=
by
  sorry

end sum_of_angles_is_990_l127_127223


namespace minimum_value_of_f_l127_127574

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x + 1)

theorem minimum_value_of_f (x : ℝ) (h : x > -1) : f x = 1 ↔ x = 0 :=
by
  sorry

end minimum_value_of_f_l127_127574


namespace num_common_divisors_60_90_l127_127856

theorem num_common_divisors_60_90 : 
  let n1 := 60
  let n2 := 90
  let gcd := 30 -- GCD calculated from prime factorizations
  let divisors_of_gcd := [1, 2, 3, 5, 6, 10, 15, 30]
  in divisors_of_gcd.length = 8 :=
by
  sorry

end num_common_divisors_60_90_l127_127856


namespace seats_in_hall_l127_127426

theorem seats_in_hall (S : ℝ) (h1 : 0.50 * S = 300) : S = 600 :=
by
  sorry

end seats_in_hall_l127_127426


namespace real_roots_range_real_roots_specific_value_l127_127266

-- Part 1
theorem real_roots_range (a b m : ℝ) (h_eq : a ≠ 0) (h_discriminant : b^2 - 4 * a * m ≥ 0) :
  m ≤ (b^2) / (4 * a) :=
sorry

-- Part 2
theorem real_roots_specific_value (x1 x2 m : ℝ) (h_sum : x1 + x2 = 4) (h_product : x1 * x2 = m)
  (h_condition : x1^2 + x2^2 + (x1 * x2)^2 = 40) (h_range : m ≤ 4) :
  m = -4 :=
sorry

end real_roots_range_real_roots_specific_value_l127_127266


namespace parabola_chord_length_l127_127386

theorem parabola_chord_length (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) (h2 : y2^2 = 4 * x2) 
  (hx : x1 + x2 = 9) 
  (focus_line : ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b → y^2 = 4 * x) :
  |(x1 - 1, y1) - (x2 - 1, y2)| = 11 := 
sorry

end parabola_chord_length_l127_127386


namespace min_value_f_l127_127526

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1)^2 + (Real.exp (-x) - 1)^2

theorem min_value_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -2 :=
sorry

end min_value_f_l127_127526


namespace digit_after_decimal_l127_127792

theorem digit_after_decimal (n : ℕ) : 
  ∀ n, n > 0 → n % 3 = 0 → 150 = n → "135"[2] = '5' := 
sorry

end digit_after_decimal_l127_127792


namespace stella_annual_income_after_tax_l127_127160

-- Definitions of the conditions
def base_salary_per_month : ℝ := 3500
def bonuses : List ℝ := [1200, 600, 1500, 900, 1200]
def months_paid : ℝ := 10
def tax_rate : ℝ := 0.05

-- Calculations derived from the conditions
def total_base_salary : ℝ := base_salary_per_month * months_paid
def total_bonuses : ℝ := bonuses.sum
def total_income_before_tax : ℝ := total_base_salary + total_bonuses
def tax_deduction : ℝ := total_income_before_tax * tax_rate
def annual_income_after_tax : ℝ := total_income_before_tax - tax_deduction

-- The theorem to prove
theorem stella_annual_income_after_tax :
  annual_income_after_tax = 38380 := by
  sorry

end stella_annual_income_after_tax_l127_127160


namespace find_number_l127_127389

theorem find_number (x : ℕ) (h : x * 9999 = 724817410) : x = 72492 :=
sorry

end find_number_l127_127389


namespace zeros_of_f_l127_127325

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem zeros_of_f : (f (-1) = 0) ∧ (f 1 = 0) ∧ (f 2 = 0) :=
by 
  -- Placeholder for the proof
  sorry

end zeros_of_f_l127_127325


namespace unattainable_y_l127_127836

theorem unattainable_y (x : ℚ) (y : ℚ) (h : y = (1 - 2 * x) / (3 * x + 4)) (hx : x ≠ -4 / 3) : y ≠ -2 / 3 :=
by {
  sorry
}

end unattainable_y_l127_127836


namespace range_of_m_l127_127404

variable {f : ℝ → ℝ}

theorem range_of_m 
  (even_f : ∀ x : ℝ, f x = f (-x))
  (mono_f : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h : f (m + 1) < f (3 * m - 1)) :
  m > 1 ∨ m < 0 :=
sorry

end range_of_m_l127_127404


namespace count_four_digit_integers_with_product_18_l127_127560

def valid_digits (n : ℕ) : Prop := 
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def digit_product_18 (a b c d : ℕ) : Prop := 
  a * b * c * d = 18

def four_digit_integer (a b c d : ℕ) : Prop := 
  valid_digits a ∧ valid_digits b ∧ valid_digits c ∧ valid_digits d

theorem count_four_digit_integers_with_product_18 : 
  (∑ a b c d in {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    ite (four_digit_integer a b c d ∧ digit_product_18 a b c d) 1 0) = 48 := 
sorry

end count_four_digit_integers_with_product_18_l127_127560


namespace flatville_additional_plates_max_count_l127_127092

noncomputable def flatville_initial_plate_count : Nat :=
  6 * 4 * 5

noncomputable def flatville_max_plate_count : Nat :=
  6 * 6 * 6

theorem flatville_additional_plates_max_count : flatville_max_plate_count - flatville_initial_plate_count = 96 :=
by
  sorry

end flatville_additional_plates_max_count_l127_127092


namespace monomial_properties_l127_127020

theorem monomial_properties (a b : ℕ) (h : a = 2 ∧ b = 1) : 
  (2 * a ^ 2 * b = 2 * (a ^ 2) * b) ∧ (2 = 2) ∧ ((2 + 1) = 3) :=
by
  sorry

end monomial_properties_l127_127020


namespace bottles_needed_l127_127204

-- Define specific values provided in conditions
def servings_per_guest : ℕ := 2
def number_of_guests : ℕ := 120
def servings_per_bottle : ℕ := 6

-- Define total servings needed
def total_servings : ℕ := servings_per_guest * number_of_guests

-- Define the number of bottles needed (as a proof statement)
theorem bottles_needed : total_servings / servings_per_bottle = 40 := by
  /-
    The proof will go here. For now we place a sorry to mark the place where
    a proof would be required. The statement should check the equivalence of 
    number of bottles needed being 40 given the total servings divided by 
    servings per bottle.
  -/
  sorry

end bottles_needed_l127_127204


namespace distinct_four_digit_integers_with_digit_product_18_l127_127568

theorem distinct_four_digit_integers_with_digit_product_18 : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in digits.prod = 18) → 
  (finset.univ.filter (λ m, (let mdigits := [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10] in mdigits.prod = 18))).card = 36 :=
by
  sorry

end distinct_four_digit_integers_with_digit_product_18_l127_127568


namespace sequence_formula_l127_127962

theorem sequence_formula (a : ℕ → ℕ) :
  (a 1 = 1) ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 3 * a n + 3 ^ n) → 
  ∀ n : ℕ, 0 < n → a n = n * 3 ^ (n - 1) :=
by
  sorry

end sequence_formula_l127_127962


namespace ratio_of_teaspoons_to_knives_is_2_to_1_l127_127936

-- Define initial conditions based on the problem
def initial_knives : ℕ := 24
def initial_teaspoons (T : ℕ) : Prop := 
  initial_knives + T + (1 / 3 : ℚ) * initial_knives + (2 / 3 : ℚ) * T = 112

-- Define the ratio to be proved
def ratio_teaspoons_to_knives (T : ℕ) : Prop :=
  initial_teaspoons T ∧ T = 48 ∧ 48 / initial_knives = 2

theorem ratio_of_teaspoons_to_knives_is_2_to_1 : ∃ T, ratio_teaspoons_to_knives T :=
by
  -- Proof would follow here
  sorry

end ratio_of_teaspoons_to_knives_is_2_to_1_l127_127936


namespace convex_functions_exist_l127_127801

noncomputable def exponential_function (x : ℝ) : ℝ :=
  4 - 5 * (1 / 2) ^ x

noncomputable def inverse_tangent_function (x : ℝ) : ℝ :=
  (10 / Real.pi) * Real.arctan x - 1

theorem convex_functions_exist :
  ∃ (f1 f2 : ℝ → ℝ),
    (∀ x, 0 < x → f1 x = exponential_function x) ∧
    (∀ x, 0 < x → f2 x = inverse_tangent_function x) ∧
    (∀ x, 0 < x → f1 x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ x, 0 < x → f2 x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 →
      f1 x1 + f1 x2 < 2 * f1 ((x1 + x2) / 2)) ∧
    (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 →
      f2 x1 + f2 x2 < 2 * f2 ((x1 + x2) / 2)) :=
sorry

end convex_functions_exist_l127_127801


namespace balloon_total_l127_127986

def total_balloons (joan_balloons melanie_balloons : ℕ) : ℕ :=
  joan_balloons + melanie_balloons

theorem balloon_total :
  total_balloons 40 41 = 81 :=
by
  sorry

end balloon_total_l127_127986


namespace exponentiation_problem_l127_127479

theorem exponentiation_problem : 2^3 * 2^2 * 3^3 * 3^2 = 6^5 :=
by sorry

end exponentiation_problem_l127_127479


namespace a2_plus_b2_minus_abc_is_perfect_square_l127_127572

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem a2_plus_b2_minus_abc_is_perfect_square {a b c : ℕ} (h : 0 < a^2 + b^2 - a * b * c ∧ a^2 + b^2 - a * b * c ≤ c) :
  is_perfect_square (a^2 + b^2 - a * b * c) :=
by
  sorry

end a2_plus_b2_minus_abc_is_perfect_square_l127_127572


namespace geometric_sequence_ratio_l127_127840

-- Definitions and conditions from part a)
def q : ℚ := 1 / 2

def sum_of_first_n (a1 : ℚ) (n : ℕ) : ℚ :=
  a1 * (1 - q ^ n) / (1 - q)

def a_n (a1 : ℚ) (n : ℕ) : ℚ :=
  a1 * q ^ (n - 1)

-- Theorem representing the proof problem from part c)
theorem geometric_sequence_ratio (a1 : ℚ) : 
  (sum_of_first_n a1 4) / (a_n a1 3) = 15 / 2 := 
sorry

end geometric_sequence_ratio_l127_127840


namespace abs_diff_of_two_numbers_l127_127463

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 504) : |x - y| = 3 :=
by
  sorry

end abs_diff_of_two_numbers_l127_127463


namespace constant_term_in_expansion_l127_127872

theorem constant_term_in_expansion :
  let f := (x - (2 / x^2))
  let expansion := f^9
  ∃ c: ℤ, expansion = c ∧ c = -672 :=
sorry

end constant_term_in_expansion_l127_127872


namespace union_of_A_and_B_l127_127442

def A := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 3)}
def B := {y : ℝ | ∃ (x : ℝ), y = Real.exp x}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 0} := by
sorry

end union_of_A_and_B_l127_127442


namespace original_price_l127_127909

theorem original_price (P : ℕ) (h : (1 / 8) * P = 8) : P = 64 :=
sorry

end original_price_l127_127909


namespace three_digit_solutions_modulo_l127_127970

def three_digit_positive_integers (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999

theorem three_digit_solutions_modulo (h : ∃ x : ℕ, three_digit_positive_integers x ∧ 
  (2597 * x + 763) % 17 = 1459 % 17) : 
  ∃ (count : ℕ), count = 53 :=
by sorry

end three_digit_solutions_modulo_l127_127970


namespace slope_of_line_l127_127336

theorem slope_of_line : 
  ∀ (x1 y1 x2 y2 : ℝ), 
  x1 = 1 → y1 = 3 → x2 = 6 → y2 = -7 → 
  (x1 ≠ x2) → ((y2 - y1) / (x2 - x1) = -2) :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2 hx1_ne_x2
  rw [hx1, hy1, hx2, hy2]
  sorry

end slope_of_line_l127_127336


namespace brownie_to_bess_ratio_l127_127230

-- Define daily milk production
def bess_daily_milk : ℕ := 2
def daisy_daily_milk : ℕ := bess_daily_milk + 1

-- Calculate weekly milk production
def bess_weekly_milk : ℕ := bess_daily_milk * 7
def daisy_weekly_milk : ℕ := daisy_daily_milk * 7

-- Given total weekly milk production
def total_weekly_milk : ℕ := 77
def combined_bess_daisy_weekly_milk : ℕ := bess_weekly_milk + daisy_weekly_milk
def brownie_weekly_milk : ℕ := total_weekly_milk - combined_bess_daisy_weekly_milk

-- Main proof statement
theorem brownie_to_bess_ratio : brownie_weekly_milk / bess_weekly_milk = 3 :=
by
  -- Skip the proof
  sorry

end brownie_to_bess_ratio_l127_127230


namespace sufficient_but_not_necessary_condition_l127_127852

-- Define set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define set Q
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem statement
theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ∈ P → x ∈ Q) ∧ (¬(x ∈ Q → x ∈ P)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l127_127852


namespace system_solution_correct_l127_127261

theorem system_solution_correct (b : ℝ) : (∃ x y : ℝ, (y = 3 * x - 5) ∧ (y = 2 * x + b) ∧ (x = 1) ∧ (y = -2)) ↔ b = -4 :=
by
  sorry

end system_solution_correct_l127_127261


namespace n_cubed_plus_two_not_divisible_by_nine_l127_127762

theorem n_cubed_plus_two_not_divisible_by_nine (n : ℕ) : ¬ (9 ∣ n^3 + 2) :=
sorry

end n_cubed_plus_two_not_divisible_by_nine_l127_127762


namespace ashley_champagne_bottles_l127_127212

theorem ashley_champagne_bottles (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) 
  (h1 : guests = 120) (h2 : glasses_per_guest = 2) (h3 : servings_per_bottle = 6) : 
  (guests * glasses_per_guest) / servings_per_bottle = 40 :=
by
  -- The proof will go here
  sorry

end ashley_champagne_bottles_l127_127212


namespace batsman_average_increase_l127_127804

theorem batsman_average_increase 
    (A : ℝ) 
    (h1 : 11 * A + 80 = 12 * 47) : 
    47 - A = 3 := 
by 
  -- Proof goes here
  sorry

end batsman_average_increase_l127_127804


namespace reducible_fraction_l127_127517

theorem reducible_fraction (l : ℤ) : ∃ k : ℤ, l = 13 * k + 4 ↔ (∃ d > 1, d ∣ (5 * l + 6) ∧ d ∣ (8 * l + 7)) :=
sorry

end reducible_fraction_l127_127517


namespace find_k_l127_127411

variable (k : ℝ)
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, 2)

theorem find_k 
  (h : (k * a.1 - b.1, k * a.2 - b.2) = (k - 1, k - 2)) 
  (perp_cond : (k * a.1 - b.1, k * a.2 - b.2).fst * (b.1 + a.1) + (k * a.1 - b.1, k * a.2 - b.2).snd * (b.2 + a.2) = 0) :
  k = 8 / 5 :=
sorry

end find_k_l127_127411


namespace ratio_nonupgraded_to_upgraded_l127_127197

-- Define the initial conditions and properties
variable (S : ℝ) (N : ℝ)
variable (h1 : ∀ N, N = S / 32)
variable (h2 : ∀ S, 0.25 * S = 0.25 * S)
variable (h3 : S > 0)

-- Define the theorem to show the required ratio
theorem ratio_nonupgraded_to_upgraded (h3 : 24 * N = 0.75 * S) : (N / (0.25 * S) = 1 / 8) :=
by
  sorry

end ratio_nonupgraded_to_upgraded_l127_127197


namespace sector_triangle_radii_l127_127056

theorem sector_triangle_radii 
  (r : ℝ) (theta : ℝ) (radius : ℝ) 
  (h_theta_eq: theta = 60)
  (h_radius_eq: radius = 10) :
  let R := (radius * Real.sqrt 3) / 3
  let r_in := (radius * Real.sqrt 3) / 6
  R = 10 * (Real.sqrt 3) / 3 ∧ r_in = 10 * (Real.sqrt 3) / 6 := 
by
  sorry

end sector_triangle_radii_l127_127056


namespace find_x_l127_127859

theorem find_x (x : ℝ) (h : (3 * x - 4) / 7 = 15) : x = 109 / 3 :=
by sorry

end find_x_l127_127859


namespace perpendicular_lines_l127_127281

theorem perpendicular_lines (a : ℝ) : (x + 2*y + 1 = 0) ∧ (ax + y - 2 = 0) → a = -2 :=
by
  sorry

end perpendicular_lines_l127_127281


namespace find_a_l127_127724

def setA (a : ℝ) : Set ℝ := {2, 4, a^3 - 2 * a^2 - a + 7}
def setB (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem find_a (a : ℝ) : 
  (setA a ∩ setB a = {2, 5}) → (a = -1 ∨ a = 2) :=
sorry

end find_a_l127_127724


namespace solve_equation_l127_127831

theorem solve_equation (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ↔ (x = 13 ∨ x = -2) :=
by
  sorry

end solve_equation_l127_127831


namespace correct_conclusions_l127_127904

-- Definitions for the events and probabilities
variable {Ω : Type*} [ProbabilitySpace Ω]
variables (A1 A2 A3 B : Event Ω)

-- Given conditions
variable [h1 : Probabilityable A1 = 5/10]
variable [h2 : Probabilityable A2 = 2/10]
variable [h3 : Probabilityable A3 = 3/10]
variable [h4 : Probabilityable B = 9/22]
variable [h5 : Probability B A1 = 2/5]
variable [h6 : Independent B A1]
variable [h7 : MutuallyExclusive A1 A2 A3]

-- Goal to prove
theorem correct_conclusions : (A1 = 1/2) ∧ (A2 = 1/5) ∧ (A3 = 3/10) ∧ (B = 9/22) :=
by
-- Proof not required, so we skip it with sorry
sorry

end correct_conclusions_l127_127904


namespace max_value_of_expression_l127_127982

-- Define the variables and constraints
variables {a b c d : ℤ}
variables (S : finset ℤ) (a_val b_val c_val d_val : ℤ)

axiom h1 : S = {0, 1, 2, 4, 5}
axiom h2 : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S
axiom h3 : ∀ x ∈ S, x = a → x ≠ b ∧ x ≠ c ∧ x ≠ d
axiom h4 : ∀ x ∈ S, x = b → x ≠ a ∧ x ≠ c ∧ x ≠ d
axiom h5 : ∀ x ∈ S, x = c → x ≠ a ∧ x ≠ b ∧ x ≠ d
axiom h6 : ∀ x ∈ S, x = d → x ≠ a ∧ x ≠ b ∧ x ≠ c

-- The main theorem to be proven
theorem max_value_of_expression : (∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
  (∀ x ∈ S, (x = a → x ≠ b ∧ x ≠ c ∧ x ≠ d) ∧ 
             (x = b → x ≠ a ∧ x ≠ c ∧ x ≠ d) ∧ 
             (x = c → x ≠ a ∧ x ≠ b ∧ x ≠ d) ∧ 
             (x = d → x ≠ a ∧ x ≠ b ∧ x ≠ c)) ∧
  (c * a^b - d = 20)) :=
sorry

end max_value_of_expression_l127_127982


namespace compare_neg_one_neg_sqrt_two_l127_127511

theorem compare_neg_one_neg_sqrt_two : -1 > -Real.sqrt 2 :=
  by
    sorry

end compare_neg_one_neg_sqrt_two_l127_127511


namespace find_a_for_positive_root_l127_127422

theorem find_a_for_positive_root (h : ∃ x > 0, (1 - x) / (x - 2) = a / (2 - x) - 2) : a = 1 :=
sorry

end find_a_for_positive_root_l127_127422


namespace workers_time_together_l127_127632

theorem workers_time_together (T : ℝ) (h1 : ∀ t : ℝ, (T + 8) = t → 1 / t = 1 / (T + 8))
                                (h2 : ∀ t : ℝ, (T + 4.5) = t → 1 / t = 1 / (T + 4.5))
                                (h3 : 1 / (T + 8) + 1 / (T + 4.5) = 1 / T) : T = 6 :=
sorry

end workers_time_together_l127_127632


namespace min_expr_value_l127_127994

theorem min_expr_value (α β : ℝ) :
  ∃ (c : ℝ), c = 36 ∧ ((3 * Real.cos α + 4 * Real.sin β - 5)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = c) :=
sorry

end min_expr_value_l127_127994


namespace find_a_minus_b_l127_127549

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x + 1

theorem find_a_minus_b (a b : ℝ)
  (h1 : deriv (f a b) 1 = -2)
  (h2 : deriv (f a b) (2 / 3) = 0) :
  a - b = 10 :=
sorry

end find_a_minus_b_l127_127549


namespace factorization_problem_l127_127025

theorem factorization_problem (a b : ℤ) : 
  (∀ y : ℤ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) → a - b = 1 := 
by
  sorry

end factorization_problem_l127_127025


namespace tablets_of_medicine_A_l127_127489

-- Given conditions as definitions
def B_tablets : ℕ := 16

def min_extracted_tablets : ℕ := 18

-- Question and expected answer encapsulated in proof statement
theorem tablets_of_medicine_A (A_tablets : ℕ) (h : A_tablets + B_tablets - 2 >= min_extracted_tablets) : A_tablets = 3 :=
sorry

end tablets_of_medicine_A_l127_127489


namespace triangle_with_consecutive_sides_and_angle_property_l127_127296

theorem triangle_with_consecutive_sides_and_angle_property :
  ∃ (a b c : ℕ), (b = a + 1) ∧ (c = b + 1) ∧
    (∃ (α β γ : ℝ), 2 * α = γ ∧
      (a * a + b * b = c * c + 2 * a * b * α.cos) ∧
      (b * b + c * c = a * a + 2 * b * c * β.cos) ∧
      (c * c + a * a = b * b + 2 * c * a * γ.cos) ∧
      (a = 4) ∧ (b = 5) ∧ (c = 6) ∧
      (γ.cos = 1 / 8)) :=
sorry

end triangle_with_consecutive_sides_and_angle_property_l127_127296


namespace worm_length_difference_l127_127017

theorem worm_length_difference
  (worm1 worm2: ℝ)
  (h_worm1: worm1 = 0.8)
  (h_worm2: worm2 = 0.1) :
  worm1 - worm2 = 0.7 :=
by
  -- starting the proof
  sorry

end worm_length_difference_l127_127017


namespace original_height_l127_127656

theorem original_height (h : ℝ) (h_rebound : ∀ n : ℕ, h / (4/3)^(n+1) > 0) (total_distance : ∀ h : ℝ, h*(1 + 1.5 + 1.5*(0.75) + 1.5*(0.75)^2 + 1.5*(0.75)^3 + (0.75)^4) = 305) :
  h = 56.3 := 
sorry

end original_height_l127_127656


namespace max_value_point_l127_127719

noncomputable def f (x : ℝ) : ℝ := x + Real.cos (2 * x)

theorem max_value_point : ∃ x ∈ Set.Ioo 0 Real.pi, (∀ y ∈ Set.Ioo 0 Real.pi, f x ≥ f y) ∧ x = Real.pi / 12 :=
by sorry

end max_value_point_l127_127719


namespace isosceles_triangle_perimeter_l127_127119

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : (a = 3 ∨ a = 7)) (h2 : (b = 3 ∨ b = 7)) (h3 : a ≠ b) : 
  ∃ (c : ℕ), (a = 7 ∧ b = 3 ∧ c = 17) ∨ (a = 3 ∧ b = 7 ∧ c = 17) := 
by
  sorry

end isosceles_triangle_perimeter_l127_127119


namespace heart_beats_during_marathon_l127_127201

theorem heart_beats_during_marathon :
  (∃ h_per_min t1 t2 total_time,
    h_per_min = 140 ∧
    t1 = 15 * 6 ∧
    t2 = 15 * 5 ∧
    total_time = t1 + t2 ∧
    23100 = h_per_min * total_time) :=
  sorry

end heart_beats_during_marathon_l127_127201


namespace p_range_l127_127302

def h (x : ℝ) : ℝ := 2 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem p_range :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → 29 ≤ p x ∧ p x ≤ 93 :=
by
  intros x hx
  sorry

end p_range_l127_127302


namespace find_n_find_m_constant_term_find_m_max_coefficients_l127_127553

-- 1. Prove that if the sum of the binomial coefficients is 256, then n = 8.
theorem find_n (n : ℕ) (h : 2^n = 256) : n = 8 :=
by sorry

-- 2. Prove that if the constant term is 35/8, then m = ±1/2.
theorem find_m_constant_term (m : ℚ) (h : m^4 * (Nat.choose 8 4) = 35/8) : m = 1/2 ∨ m = -1/2 :=
by sorry

-- 3. Prove that if only the 6th and 7th terms have the maximum coefficients, then m = 2.
theorem find_m_max_coefficients (m : ℚ) (h1 : m ≠ 0) (h2 : m^5 * (Nat.choose 8 5) = m^6 * (Nat.choose 8 6)) : m = 2 :=
by sorry

end find_n_find_m_constant_term_find_m_max_coefficients_l127_127553


namespace river_width_l127_127500

def boat_width : ℕ := 3
def num_boats : ℕ := 8
def space_between_boats : ℕ := 2
def riverbank_space : ℕ := 2

theorem river_width : 
  let boat_space := num_boats * boat_width
  let between_boat_space := (num_boats - 1) * space_between_boats
  let riverbank_space_total := 2 * riverbank_space
  boat_space + between_boat_space + riverbank_space_total = 42 :=
by
  sorry

end river_width_l127_127500


namespace probability_unique_rolls_l127_127238

theorem probability_unique_rolls :
  let num_people := 5 in
  let die_faces := 6 in
  let total_ways := die_faces ^ num_people in
  let successful_ways := die_faces * (die_faces - 1) * (die_faces - 2) * (die_faces - 3) * (die_faces - 4) in
  let probability := (successful_ways : ℚ) / (total_ways : ℚ) in
  probability = 5 / 54 :=
by 
  sorry

end probability_unique_rolls_l127_127238


namespace total_messages_three_days_l127_127677

theorem total_messages_three_days :
  ∀ (A1 A2 A3 L1 L2 L3 : ℕ),
  A1 = L1 - 20 →
  L1 = 120 →
  L2 = (1 / 3 : ℚ) * L1 →
  A2 = 2 * A1 →
  A1 + L1 = A3 + L3 →
  (A1 + L1 + A2 + L2 + A3 + L3 = 680) := by
  intros A1 A2 A3 L1 L2 L3 h1 h2 h3 h4 h5
  sorry

end total_messages_three_days_l127_127677


namespace N_def_M_intersection_CU_N_def_M_union_N_def_l127_127487

section Sets

variable {α : Type}

-- Declarations of conditions
def U := {x : ℝ | -3 ≤ x ∧ x ≤ 3}
def M := {x : ℝ | -1 < x ∧ x < 1}
def CU (N : Set ℝ) := {x : ℝ | 0 < x ∧ x < 2}

-- Problem statements
theorem N_def (N : Set ℝ) : N = {x : ℝ | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)} ↔ CU N = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

theorem M_intersection_CU_N_def (N : Set ℝ) : (M ∩ CU N) = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

theorem M_union_N_def (N : Set ℝ) : (M ∪ N) = {x : ℝ | (-3 ≤ x ∧ x < 1) ∨ (2 ≤ x ∧ x ≤ 3)} :=
by sorry

end Sets

end N_def_M_intersection_CU_N_def_M_union_N_def_l127_127487


namespace sum_fraction_series_l127_127948

theorem sum_fraction_series :
  ( ∑ i in finset.range (6), (1 : ℚ) / ((i + 3) * (i + 4)) ) = 2 / 9 :=
by
  sorry

end sum_fraction_series_l127_127948


namespace number_of_tshirts_sold_l127_127318

theorem number_of_tshirts_sold 
    (original_price discounted_price revenue : ℕ)
    (discount : ℕ) 
    (no_of_tshirts: ℕ)
    (h1 : original_price = 51)
    (h2 : discount = 8)
    (h3 : discounted_price = original_price - discount)
    (h4 : revenue = 5590)
    (h5 : revenue = no_of_tshirts * discounted_price) : 
    no_of_tshirts = 130 :=
by
  sorry

end number_of_tshirts_sold_l127_127318


namespace find_k_l127_127521

theorem find_k (k : ℝ) : 2 + (2 + k) / 3 + (2 + 2 * k) / 3^2 + (2 + 3 * k) / 3^3 + 
  ∑' (n : ℕ), (2 + (n + 1) * k) / 3^(n + 1) = 7 ↔ k = 16 / 3 := 
sorry

end find_k_l127_127521


namespace max_a_value_l127_127112

theorem max_a_value : ∃ a b : ℕ, 1 < a ∧ a < b ∧
  (∀ x y : ℝ, y = -2 * x + 4033 ∧ y = |x - 1| + |x + a| + |x - b| → 
  a = 4031) := sorry

end max_a_value_l127_127112


namespace smallest_base_10_integer_l127_127041

theorem smallest_base_10_integer :
  ∃ (c d : ℕ), 3 < c ∧ 3 < d ∧ (3 * c + 4 = 4 * d + 3) ∧ (3 * c + 4 = 19) :=
by {
 sorry
}

end smallest_base_10_integer_l127_127041


namespace Adam_total_candy_l127_127365

theorem Adam_total_candy :
  (2 + 5) * 4 = 28 := 
by 
  sorry

end Adam_total_candy_l127_127365


namespace john_has_388_pennies_l127_127748

theorem john_has_388_pennies (k : ℕ) (j : ℕ) (hk : k = 223) (hj : j = k + 165) : j = 388 := by
  sorry

end john_has_388_pennies_l127_127748


namespace derek_bought_more_cars_l127_127942

-- Define conditions
variables (d₆ c₆ d₁₆ c₁₆ : ℕ)

-- Given conditions
def initial_conditions :=
  (d₆ = 90) ∧
  (d₆ = 3 * c₆) ∧
  (d₁₆ = 120) ∧
  (c₁₆ = 2 * d₁₆)

-- Prove the number of cars Derek bought in ten years
theorem derek_bought_more_cars (h : initial_conditions d₆ c₆ d₁₆ c₁₆) : c₁₆ - c₆ = 210 :=
by sorry

end derek_bought_more_cars_l127_127942


namespace min_value_of_xy_l127_127974

theorem min_value_of_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h : 2 * x + y + 6 = x * y) : 18 ≤ x * y :=
by
  sorry

end min_value_of_xy_l127_127974


namespace number_mod_conditions_l127_127196

theorem number_mod_conditions :
  ∃ N, (N % 10 = 9) ∧ (N % 9 = 8) ∧ (N % 8 = 7) ∧ (N % 7 = 6) ∧
       (N % 6 = 5) ∧ (N % 5 = 4) ∧ (N % 4 = 3) ∧ (N % 3 = 2) ∧ (N % 2 = 1) ∧
       N = 2519 :=
by
  sorry

end number_mod_conditions_l127_127196


namespace smallest_divisible_four_digit_number_l127_127089

theorem smallest_divisible_four_digit_number :
  let n := 2310
  in ∃ k : ℕ, n * k >= 1000 ∧ n = 2 * 3 * 5 * 7 * 11 :=
begin
  let n := 2310,
  use 1,
  split,
  { norm_num },
  { norm_num },
  sorry
end

end smallest_divisible_four_digit_number_l127_127089


namespace find_y_coordinate_of_first_point_l127_127922

theorem find_y_coordinate_of_first_point :
  ∃ y1 : ℝ, ∀ k : ℝ, (k = 0.8) → (k = (0.8 - y1) / (5 - (-1))) → y1 = 4 :=
by
  sorry

end find_y_coordinate_of_first_point_l127_127922


namespace min_rice_proof_l127_127931

noncomputable def minRicePounds : ℕ := 2

theorem min_rice_proof (o r : ℕ) (h1 : o ≥ 8 + 3 * r / 4) (h2 : o ≤ 5 * r) :
  r ≥ 2 :=
by
  sorry

end min_rice_proof_l127_127931


namespace rubles_exchange_l127_127736

theorem rubles_exchange (x : ℕ) : 
  (3000 * x - 7000 = 2950 * x) → x = 140 := by
  sorry

end rubles_exchange_l127_127736


namespace highest_score_not_necessarily_at_least_12_l127_127052

section

-- Define the number of teams
def teams : ℕ := 12

-- Define the number of games each team plays
def games_per_team : ℕ := teams - 1

-- Define the total number of games
def total_games : ℕ := (teams * games_per_team) / 2

-- Define the points system
def points_for_win : ℕ := 2
def points_for_draw : ℕ := 1

-- Define the total points in the tournament
def total_points : ℕ := total_games * points_for_win

-- The highest score possible statement
def highest_score_must_be_at_least_12_statement : Prop :=
  ∀ (scores : Fin teams → ℕ), (∃ i, scores i ≥ 12)

-- Theorem stating that the statement "The highest score must be at least 12" is false
theorem highest_score_not_necessarily_at_least_12 (h : ∀ (scores : Fin teams → ℕ), (∃ i, scores i ≥ 12)) : False :=
  sorry

end

end highest_score_not_necessarily_at_least_12_l127_127052


namespace range_of_m_l127_127862

theorem range_of_m {f : ℝ → ℝ} (h : ∀ x, f x = x^2 - 6*x - 16)
  {a b : ℝ} (h_domain : ∀ x, 0 ≤ x ∧ x ≤ a → ∃ y, f y ≤ b) 
  (h_range : ∀ y, -25 ≤ y ∧ y ≤ -16 → ∃ x, f x = y) : 3 ≤ a ∧ a ≤ 6 := 
sorry

end range_of_m_l127_127862


namespace hulk_jump_distance_exceeds_1000_l127_127616

theorem hulk_jump_distance_exceeds_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m < n → 3^m ≤ 1000) ∧ 3^n > 1000 :=
sorry

end hulk_jump_distance_exceeds_1000_l127_127616


namespace simplify_expression_l127_127514

theorem simplify_expression (a : ℝ) (h : a ≠ 1/2) : 1 - (2 / (1 + (2 * a) / (1 - 2 * a))) = 4 * a - 1 :=
by
  sorry

end simplify_expression_l127_127514


namespace classify_triangle_l127_127406

theorem classify_triangle (m : ℕ) (h₁ : m > 1) (h₂ : 3 * m + 3 = 180) :
  (m < 60) ∧ (m + 1 < 90) ∧ (m + 2 < 90) :=
by
  sorry

end classify_triangle_l127_127406


namespace no_integer_right_triangle_side_x_l127_127522

theorem no_integer_right_triangle_side_x :
  ∀ (x : ℤ), (12 + 30 > x ∧ 12 + x > 30 ∧ 30 + x > 12) →
             (12^2 + 30^2 = x^2 ∨ 12^2 + x^2 = 30^2 ∨ 30^2 + x^2 = 12^2) →
             (¬ (∃ x : ℤ, 18 < x ∧ x < 42)) :=
by
  sorry

end no_integer_right_triangle_side_x_l127_127522


namespace width_of_shop_l127_127032

theorem width_of_shop 
  (monthly_rent : ℕ) 
  (length : ℕ) 
  (annual_rent_per_sqft : ℕ) 
  (h1 : monthly_rent = 3600) 
  (h2 : length = 18) 
  (h3 : annual_rent_per_sqft = 120) :
  ∃ width : ℕ, width = 20 :=
by
  sorry

end width_of_shop_l127_127032


namespace find_original_faculty_count_l127_127449

variable (F : ℝ)
variable (final_count : ℝ := 195)
variable (first_year_reduction : ℝ := 0.075)
variable (second_year_increase : ℝ := 0.125)
variable (third_year_reduction : ℝ := 0.0325)
variable (fourth_year_increase : ℝ := 0.098)
variable (fifth_year_reduction : ℝ := 0.1465)

theorem find_original_faculty_count (h : F * (1 - first_year_reduction)
                                        * (1 + second_year_increase)
                                        * (1 - third_year_reduction)
                                        * (1 + fourth_year_increase)
                                        * (1 - fifth_year_reduction) = final_count) :
  F = 244 :=
by sorry

end find_original_faculty_count_l127_127449


namespace pets_beds_calculation_l127_127445

theorem pets_beds_calculation
  (initial_beds : ℕ)
  (additional_beds : ℕ)
  (total_pets : ℕ)
  (H1 : initial_beds = 12)
  (H2 : additional_beds = 8)
  (H3 : total_pets = 10) :
  (initial_beds + additional_beds) / total_pets = 2 := 
by 
  sorry

end pets_beds_calculation_l127_127445


namespace find_x_plus_inv_x_l127_127258

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end find_x_plus_inv_x_l127_127258


namespace smallest_n_for_sqrt_50n_is_integer_l127_127403

theorem smallest_n_for_sqrt_50n_is_integer :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (50 * n) = k * k) ∧ n = 2 :=
by
  sorry

end smallest_n_for_sqrt_50n_is_integer_l127_127403


namespace B_representation_l127_127838

def A : Set ℤ := {-1, 2, 3, 4}

def f (x : ℤ) : ℤ := x^2 - 2*x + 2

def B : Set ℤ := {y | ∃ x ∈ A, y = f x}

theorem B_representation : B = {2, 5, 10} :=
by {
  -- Proof to be provided
  sorry
}

end B_representation_l127_127838


namespace angle_B_range_l127_127843

theorem angle_B_range (A B C : ℝ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : A + B + C = 180) (h4 : 2 * B = 5 * A) :
  0 < B ∧ B ≤ 75 :=
by
  sorry

end angle_B_range_l127_127843


namespace part1_part2_l127_127099

-- The quadratic equation of interest
def quadratic_eq (k x : ℝ) : ℝ :=
  x^2 + (2 * k - 1) * x + k^2 - k

-- Part 1: Proof that the equation has two distinct real roots
theorem part1 (k : ℝ) : (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic_eq k x1 = 0 ∧ quadratic_eq k x2 = 0) := 
  sorry

-- Part 2: Given x = 2 is a root, prove the value of the expression
theorem part2 (k : ℝ) (h : quadratic_eq k 2 = 0) : -2 * k^2 - 6 * k - 5 = -1 :=
  sorry

end part1_part2_l127_127099


namespace count_valid_outfits_l127_127726

/-
Problem:
I have 5 shirts, 3 pairs of pants, and 5 hats. The pants come in red, green, and blue. 
The shirts and hats come in those colors, plus orange and purple. 
I refuse to wear an outfit where the shirt and the hat are the same color. 
How many choices for outfits, consisting of one shirt, one hat, and one pair of pants, do I have?
-/

def num_shirts := 5
def num_pants := 3
def num_hats := 5
def valid_outfits := 66

-- The set of colors available for shirts and hats
inductive color
| red | green | blue | orange | purple

-- Conditions and properties translated into Lean
def pants_colors : List color := [color.red, color.green, color.blue]
def shirt_hat_colors : List color := [color.red, color.green, color.blue, color.orange, color.purple]

theorem count_valid_outfits (h1 : num_shirts = 5) 
                            (h2 : num_pants = 3) 
                            (h3 : num_hats = 5) 
                            (h4 : ∀ (s : color), s ∈ shirt_hat_colors) 
                            (h5 : ∀ (p : color), p ∈ pants_colors) 
                            (h6 : ∀ (s h : color), s ≠ h) :
  valid_outfits = 66 :=
by
  sorry

end count_valid_outfits_l127_127726


namespace perpendicular_slope_l127_127031

theorem perpendicular_slope (k : ℝ) : (∀ x, y = k*x) ∧ (∀ x, y = 2*x + 1) → k = -1 / 2 :=
by
  intro h
  sorry

end perpendicular_slope_l127_127031


namespace pipe_A_fills_tank_in_28_hours_l127_127797

variable (A B C : ℝ)
-- Conditions
axiom h1 : C = 2 * B
axiom h2 : B = 2 * A
axiom h3 : A + B + C = 1 / 4

theorem pipe_A_fills_tank_in_28_hours : 1 / A = 28 := by
  -- proof omitted for the exercise
  sorry

end pipe_A_fills_tank_in_28_hours_l127_127797


namespace optimal_strategy_for_father_l127_127194

-- Define the individual players
inductive player
| Father 
| Mother 
| Son

open player

-- Define the probabilities of player defeating another
def prob_defeat (p1 p2 : player) : ℝ := sorry  -- These will be defined as per the problem's conditions.

-- Define the probability of father winning given the first matchups
def P_father_vs_mother : ℝ :=
  prob_defeat Father Mother * prob_defeat Father Son +
  prob_defeat Father Mother * prob_defeat Son Father * prob_defeat Mother Son * prob_defeat Father Mother +
  prob_defeat Mother Father * prob_defeat Son Mother * prob_defeat Father Son * prob_defeat Father Mother

def P_father_vs_son : ℝ :=
  prob_defeat Father Son * prob_defeat Father Mother +
  prob_defeat Father Son * prob_defeat Mother Father * prob_defeat Son Mother * prob_defeat Father Son +
  prob_defeat Son Father * prob_defeat Mother Son * prob_defeat Father Mother * prob_defeat Father Son

-- Define the optimality condition
theorem optimal_strategy_for_father :
  P_father_vs_mother > P_father_vs_son :=
sorry

end optimal_strategy_for_father_l127_127194


namespace find_a_and_b_l127_127975

theorem find_a_and_b (a b : ℝ) (h1 : b - 1/4 = (a + b) / 4 + b / 2) (h2 : 4 * a / 3 = (a + b) / 2)  :
  a = 3/2 ∧ b = 5/2 :=
by
  sorry

end find_a_and_b_l127_127975


namespace books_in_special_collection_l127_127053

theorem books_in_special_collection (B : ℕ) :
  (∃ returned not_returned loaned_out_end  : ℝ, 
    loaned_out_end = 54 ∧ 
    returned = 0.65 * 60.00000000000001 ∧ 
    not_returned = 60.00000000000001 - returned ∧ 
    B = loaned_out_end + not_returned) → 
  B = 75 :=
by 
  intro h
  sorry

end books_in_special_collection_l127_127053


namespace percentage_discount_l127_127156

theorem percentage_discount (discounted_price original_price : ℝ) (h1 : discounted_price = 560) (h2 : original_price = 700) :
  (original_price - discounted_price) / original_price * 100 = 20 :=
by
  simp [h1, h2]
  sorry

end percentage_discount_l127_127156


namespace balloon_total_l127_127987

def total_balloons (joan_balloons melanie_balloons : ℕ) : ℕ :=
  joan_balloons + melanie_balloons

theorem balloon_total :
  total_balloons 40 41 = 81 :=
by
  sorry

end balloon_total_l127_127987


namespace find_number_of_appliances_l127_127818

-- Declare the constants related to the problem.
def commission_per_appliance : ℝ := 50
def commission_percent : ℝ := 0.1
def total_selling_price : ℝ := 3620
def total_commission : ℝ := 662

-- Define the theorem to solve for the number of appliances sold.
theorem find_number_of_appliances (n : ℝ) 
  (H : n * commission_per_appliance + commission_percent * total_selling_price = total_commission) : 
  n = 6 := 
sorry

end find_number_of_appliances_l127_127818


namespace quadratic_roots_real_distinct_l127_127126

theorem quadratic_roots_real_distinct (k : ℝ) :
  (k > (1/2)) ∧ (k ≠ 1) ↔
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k-1) * x1^2 + 2 * x1 - 2 = 0) ∧ ((k-1) * x2^2 + 2 * x2 - 2 = 0)) :=
by
  sorry

end quadratic_roots_real_distinct_l127_127126


namespace team_total_points_l127_127380

theorem team_total_points (Connor_score Amy_score Jason_score : ℕ) :
  Connor_score = 2 →
  Amy_score = Connor_score + 4 →
  Jason_score = 2 * Amy_score →
  Connor_score + Amy_score + Jason_score = 20 :=
by
  intros
  sorry

end team_total_points_l127_127380


namespace positive_difference_enrollment_l127_127476

theorem positive_difference_enrollment 
  (highest_enrollment : ℕ)
  (lowest_enrollment : ℕ)
  (h_highest : highest_enrollment = 2150)
  (h_lowest : lowest_enrollment = 980) :
  highest_enrollment - lowest_enrollment = 1170 :=
by {
  -- Proof to be added here
  sorry
}

end positive_difference_enrollment_l127_127476


namespace problem_l127_127401

theorem problem (x y : ℝ) (h1 : 2 * x + y = 4) (h2 : x + 2 * y = 5) : 5 * x ^ 2 + 8 * x * y + 5 * y ^ 2 = 41 := 
by 
  sorry

end problem_l127_127401


namespace smallest_integer_l127_127076

theorem smallest_integer (x : ℤ) (h : 3 * (Int.natAbs x)^3 + 5 < 56) : x = -2 :=
sorry

end smallest_integer_l127_127076


namespace droid_weekly_coffee_consumption_l127_127069

noncomputable def weekly_consumption_A : ℕ :=
  (3 * 5) + 4 + 2 + 1 -- Weekdays + Saturday + Sunday + Monday increase

noncomputable def weekly_consumption_B : ℕ :=
  (2 * 5) + 3 + (1 - 1 / 2) -- Weekdays + Saturday + Sunday decrease

noncomputable def weekly_consumption_C : ℕ :=
  (1 * 5) + 2 + 1 -- Weekdays + Saturday + Sunday

theorem droid_weekly_coffee_consumption :
  weekly_consumption_A = 22 ∧ weekly_consumption_B = 14 ∧ weekly_consumption_C = 8 :=
by 
  sorry

end droid_weekly_coffee_consumption_l127_127069


namespace factor_of_increase_l127_127345

noncomputable def sum_arithmetic_progression (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem factor_of_increase (a1 d n : ℕ) (h1 : a1 > 0) (h2 : (sum_arithmetic_progression a1 (3 * d) n = 2 * sum_arithmetic_progression a1 d n)) :
  sum_arithmetic_progression a1 (4 * d) n = (5 / 2) * sum_arithmetic_progression a1 d n :=
sorry

end factor_of_increase_l127_127345


namespace total_messages_three_days_l127_127678

theorem total_messages_three_days :
  ∀ (A1 A2 A3 L1 L2 L3 : ℕ),
  A1 = L1 - 20 →
  L1 = 120 →
  L2 = (1 / 3 : ℚ) * L1 →
  A2 = 2 * A1 →
  A1 + L1 = A3 + L3 →
  (A1 + L1 + A2 + L2 + A3 + L3 = 680) := by
  intros A1 A2 A3 L1 L2 L3 h1 h2 h3 h4 h5
  sorry

end total_messages_three_days_l127_127678


namespace number_of_selections_l127_127886

-- Define the given conditions
def num_colors : ℕ := 6
def num_select : ℕ := 2

-- Formalize the problem in Lean
theorem number_of_selections : Nat.choose num_colors num_select = 15 := by
  -- Proof omitted
  sorry

end number_of_selections_l127_127886


namespace numbers_must_be_equal_l127_127247

theorem numbers_must_be_equal
  (n : ℕ) (nums : Fin n → ℕ)
  (hn_pos : n = 99)
  (hbound : ∀ i, nums i < 100)
  (hdiv : ∀ (s : Finset (Fin n)) (hs : 2 ≤ s.card), ¬ 100 ∣ s.sum nums) :
  ∀ i j, nums i = nums j := 
sorry

end numbers_must_be_equal_l127_127247


namespace number_of_students_l127_127134

-- Definitions based on problem conditions
def age_condition (a n : ℕ) : Prop :=
  7 * (a - 1) + 2 * (a + 2) + (n - 9) * a = 330

-- Main theorem to prove the correct number of students
theorem number_of_students (a n : ℕ) (h : age_condition a n) : n = 37 :=
  sorry

end number_of_students_l127_127134


namespace alok_age_l127_127641

theorem alok_age (B A C : ℕ) (h1 : B = 6 * A) (h2 : B + 10 = 2 * (C + 10)) (h3 : C = 10) : A = 5 :=
by
  -- proof would go here
  sorry

end alok_age_l127_127641


namespace range_of_m_l127_127262

-- Define the conditions based on the problem statement
def equation (x m : ℝ) : Prop := (2 * x + m) = (x - 1)

-- The goal is to prove that if there exists a positive solution x to the equation, then m < -1
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, equation x m ∧ x > 0) → m < -1 :=
by
  sorry

end range_of_m_l127_127262


namespace max_value_b_minus_inv_a_is_minus_one_min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one_l127_127534

open Real

noncomputable def max_value_b_minus_inv_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : ℝ :=
b - (1 / a)

noncomputable def min_value_inv_3a_plus_1_plus_inv_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : ℝ :=
(1 / (3 * a + 1)) + (1 / (a + b))

theorem max_value_b_minus_inv_a_is_minus_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : 
  max_value_b_minus_inv_a a b ha hb h = -1 :=
sorry

theorem min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : 
  min_value_inv_3a_plus_1_plus_inv_a_plus_b a b ha hb h = 1 :=
sorry

end max_value_b_minus_inv_a_is_minus_one_min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one_l127_127534


namespace find_a_value_l127_127268

def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + (a + 2) * y + 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := a * x - y + 2

-- Define what it means for two lines to be not parallel
def not_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, (line1 a x y ≠ 0 ∧ line2 a x y ≠ 0)

theorem find_a_value (a : ℝ) (h : not_parallel a) : a = 0 ∨ a = -3 :=
  sorry

end find_a_value_l127_127268


namespace smallest_four_digit_number_divisible_by_smallest_primes_l127_127086

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end smallest_four_digit_number_divisible_by_smallest_primes_l127_127086


namespace cost_price_of_watch_l127_127928

theorem cost_price_of_watch (C SP1 SP2 : ℝ)
    (h1 : SP1 = 0.90 * C)
    (h2 : SP2 = 1.02 * C)
    (h3 : SP2 = SP1 + 140) :
    C = 1166.67 :=
by
  sorry

end cost_price_of_watch_l127_127928


namespace age_sum_is_47_l127_127480

theorem age_sum_is_47 (a b c : ℕ) (b_def : b = 18) 
  (a_def : a = b + 2) (c_def : c = b / 2) : a + b + c = 47 :=
by
  sorry

end age_sum_is_47_l127_127480


namespace andrew_purchase_grapes_l127_127508

theorem andrew_purchase_grapes (G : ℕ) (h : 70 * G + 495 = 1055) : G = 8 :=
by
  sorry

end andrew_purchase_grapes_l127_127508


namespace inequality_transfers_l127_127844

variables (a b c d : ℝ)

theorem inequality_transfers (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end inequality_transfers_l127_127844


namespace team_total_score_l127_127382

theorem team_total_score (Connor_score Amy_score Jason_score : ℕ)
  (h1 : Connor_score = 2)
  (h2 : Amy_score = Connor_score + 4)
  (h3 : Jason_score = 2 * Amy_score) :
  Connor_score + Amy_score + Jason_score = 20 :=
by
  sorry

end team_total_score_l127_127382


namespace g_triple_apply_l127_127605

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 15

theorem g_triple_apply : g (g (g 20)) = 1 :=
by
  sorry

end g_triple_apply_l127_127605


namespace smallest_four_digit_divisible_by_primes_l127_127084

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end smallest_four_digit_divisible_by_primes_l127_127084


namespace find_p_q_l127_127877

noncomputable def roots_of_polynomial (a b c : ℝ) :=
  a^3 - 2018 * a + 2018 = 0 ∧ b^3 - 2018 * b + 2018 = 0 ∧ c^3 - 2018 * c + 2018 = 0

theorem find_p_q (a b c : ℝ) (p q : ℕ) 
  (h1 : roots_of_polynomial a b c)
  (h2 : 0 < p ∧ p ≤ q) 
  (h3 : (a^(p+q) + b^(p+q) + c^(p+q))/(p+q) = (a^p + b^p + c^p)/p * (a^q + b^q + c^q)/q) : 
  p^2 + q^2 = 20 := 
sorry

end find_p_q_l127_127877


namespace shaded_area_eq_l127_127168

theorem shaded_area_eq : 
  let side := 8 
  let radius := 3 
  let square_area := side * side
  let sector_area := (1 / 4) * Real.pi * (radius * radius)
  let four_sectors_area := 4 * sector_area
  let triangle_area := (1 / 2) * radius * radius
  let four_triangles_area := 4 * triangle_area
  let shaded_area := square_area - four_sectors_area - four_triangles_area
  shaded_area = 64 - 9 * Real.pi - 18 :=
by
  sorry

end shaded_area_eq_l127_127168


namespace number_of_intersections_l127_127523

theorem number_of_intersections : ∃ (a_values : Finset ℚ), 
  ∀ a ∈ a_values, ∀ x y, y = 2 * x + a ∧ y = x^2 + 3 * a^2 ∧ x = 0 → 
  2 = a_values.card :=
by 
  sorry

end number_of_intersections_l127_127523


namespace solution_set_of_floor_equation_l127_127827

theorem solution_set_of_floor_equation (x : ℝ) : 
  (⌊⌊2 * x⌋ - 1/2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
by sorry

end solution_set_of_floor_equation_l127_127827


namespace repeating_mul_l127_127700

theorem repeating_mul (x y : ℚ) (h1 : x = (12 : ℚ) / 99) (h2 : y = (34 : ℚ) / 99) : 
    x * y = (136 : ℚ) / 3267 := by
  sorry

end repeating_mul_l127_127700


namespace triangle_area_l127_127633

theorem triangle_area :
  let a := 4
  let c := 5
  let b := Real.sqrt (c^2 - a^2)
  (1 / 2) * a * b = 6 :=
by sorry

end triangle_area_l127_127633


namespace propA_propB_relation_l127_127455

variable (x y : ℤ)

theorem propA_propB_relation :
  (x + y ≠ 5 → x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → x + y ≠ 5) :=
by
  sorry

end propA_propB_relation_l127_127455


namespace number_is_125_l127_127341

/-- Let x be a real number such that the difference between x and 3/5 of x is 50. -/
def problem_statement (x : ℝ) : Prop :=
  x - (3 / 5) * x = 50

/-- Prove that the only number that satisfies the above condition is 125. -/
theorem number_is_125 (x : ℝ) (h : problem_statement x) : x = 125 :=
by
  sorry

end number_is_125_l127_127341


namespace greatest_number_of_large_chips_l127_127171

theorem greatest_number_of_large_chips (s l p : ℕ) (h1 : s + l = 60) (h2 : s = l + p) 
  (hp_prime : Nat.Prime p) (hp_div : p ∣ l) : l ≤ 29 :=
by
  sorry

end greatest_number_of_large_chips_l127_127171


namespace ladder_base_length_l127_127802

theorem ladder_base_length {a b c : ℕ} (h1 : c = 13) (h2 : b = 12) (h3 : a^2 + b^2 = c^2) :
  a = 5 := 
by 
  sorry

end ladder_base_length_l127_127802


namespace evaluate_expression_l127_127414

variables (a b c d m : ℝ)

lemma opposite_is_zero (h1 : a + b = 0) : a + b = 0 := h1

lemma reciprocals_equal_one (h2 : c * d = 1) : c * d = 1 := h2

lemma abs_value_two (h3 : |m| = 2) : |m| = 2 := h3

theorem evaluate_expression (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 2) :
  m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 :=
by
  sorry

end evaluate_expression_l127_127414


namespace john_probability_l127_127595

theorem john_probability :
  let p := (1 : ℚ) / 2,
      n := 10,
      prob := ∑ k in finset.range (n + 1), if k >= 6 then nat.choose n k * p^k * (1 - p)^(n-k) else 0
  in prob = 193 / 512 := by
  sorry

end john_probability_l127_127595


namespace selling_price_is_80000_l127_127313

-- Given the conditions of the problem
def purchasePrice : ℕ := 45000
def repairCosts : ℕ := 12000
def profitPercent : ℚ := 40.35 / 100

-- Total cost calculation
def totalCost := purchasePrice + repairCosts

-- Profit calculation
def profit := profitPercent * totalCost

-- Selling price calculation
def sellingPrice := totalCost + profit

-- Statement of the proof problem
theorem selling_price_is_80000 : round sellingPrice = 80000 := by
  sorry

end selling_price_is_80000_l127_127313


namespace mouse_jump_distance_l127_127896

theorem mouse_jump_distance
  (g f m : ℕ)
  (hg : g = 25)
  (hf : f = g + 32)
  (hm : m = f - 26) :
  m = 31 := by
  sorry

end mouse_jump_distance_l127_127896


namespace smallest_value_of_Q_l127_127618

noncomputable def Q (x : ℝ) : ℝ := x^4 - 4*x^3 + 7*x^2 - 2*x + 10

theorem smallest_value_of_Q :
  min (Q 1) (min (10 : ℝ) (min (4 : ℝ) (min (1 - 4 + 7 - 2 + 10 : ℝ) (2.5 : ℝ)))) = 2.5 :=
by sorry

end smallest_value_of_Q_l127_127618


namespace teenas_speed_l127_127615

theorem teenas_speed (T : ℝ) :
  (7.5 + 15 + 40 * 1.5 = T * 1.5) → T = 55 := 
by
  intro h
  sorry

end teenas_speed_l127_127615


namespace problem_statement_l127_127980

open Real

noncomputable def curve_polar := {p : ℝ × ℝ // p.1 * (sin p.2)^2 = 4 * cos p.2}

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
(-2 + (sqrt 2)/2 * t, -4 + (sqrt 2)/2 * t)

def P := (-2 : ℝ, -4 : ℝ)

theorem problem_statement :
  (∃ (ρ θ : ℝ), (ρ * (sin θ)^2 = 4 * cos θ) ∧ (∀ t : ℝ, ∃ (x y : ℝ),
  x = -2 + (sqrt 2)/2 * t ∧ y = -4 + (sqrt 2)/2 * t ∧ (y^2 = 4 * x) ∧ (x - y - 2 = 0))) →
  |(-2 * sqrt 2) + (10 * sqrt 2)| = 12 * sqrt 2 :=
by
  intros h
  sorry

end problem_statement_l127_127980


namespace banker_l127_127319

theorem banker's_discount (BD TD FV : ℝ) (hBD : BD = 18) (hTD : TD = 15) 
(h : BD = TD + (TD^2 / FV)) : FV = 75 := by
  sorry

end banker_l127_127319


namespace part_one_solution_part_two_solution_l127_127847

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - a| + |x - 2|

-- Part (1): "When a = 1, find the solution set of the inequality f(x) ≥ 3"
theorem part_one_solution (x : ℝ) : f x 1 ≥ 3 ↔ x ≤ 0 ∨ x ≥ 3 :=
by sorry

-- Part (2): "If f(x) ≥ 2a - 1, find the range of values for a"
theorem part_two_solution (a : ℝ) : (∀ x : ℝ, f x a ≥ 2 * a - 1) ↔ a ≤ 1 :=
by sorry

end part_one_solution_part_two_solution_l127_127847


namespace age_is_nine_l127_127038

-- Define the conditions
def current_age (X : ℕ) :=
  X = 3 * (X - 6)

-- The theorem: Prove that the age X is equal to 9 under the conditions given
theorem age_is_nine (X : ℕ) (h : current_age X) : X = 9 :=
by
  -- The proof is omitted
  sorry

end age_is_nine_l127_127038


namespace merchant_profit_percentage_l127_127643

theorem merchant_profit_percentage (C S : ℝ) (h : 24 * C = 16 * S) : ((S - C) / C) * 100 = 50 := by
  -- Adding "by" to denote beginning of proof section
  sorry  -- Proof is skipped

end merchant_profit_percentage_l127_127643


namespace part1_solution_l127_127516

theorem part1_solution : ∀ n : ℕ, ∃ k : ℤ, 2^n + 3 = k^2 ↔ n = 0 :=
by sorry

end part1_solution_l127_127516


namespace lateral_surface_area_of_cone_l127_127029

-- Definitions of the given conditions
def base_radius_cm : ℝ := 3
def slant_height_cm : ℝ := 5

-- The theorem to prove
theorem lateral_surface_area_of_cone :
  let r := base_radius_cm
  let l := slant_height_cm
  π * r * l = 15 * π := 
by
  sorry

end lateral_surface_area_of_cone_l127_127029


namespace sum_non_prime_between_50_and_60_eq_383_l127_127343

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def non_primes_between_50_and_60 : List ℕ :=
  [51, 52, 54, 55, 56, 57, 58]

def sum_non_primes_between_50_and_60 : ℕ :=
  non_primes_between_50_and_60.sum

theorem sum_non_prime_between_50_and_60_eq_383 :
  sum_non_primes_between_50_and_60 = 383 :=
by
  sorry

end sum_non_prime_between_50_and_60_eq_383_l127_127343


namespace volume_of_cylinder_l127_127260

theorem volume_of_cylinder (r h : ℝ) (hr : r = 1) (hh : h = 2) (A : r * h = 4) : (π * r^2 * h = 2 * π) :=
by
  sorry

end volume_of_cylinder_l127_127260


namespace find_x_plus_inv_x_l127_127255

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + (1/x)^3 = 110) : x + (1/x) = 5 :=
sorry

end find_x_plus_inv_x_l127_127255


namespace sum_digits_10_pow_85_minus_85_l127_127387

-- Define the function that computes the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

-- Define the specific problem for n = 10^85 - 85
theorem sum_digits_10_pow_85_minus_85 : 
  sum_of_digits (10^85 - 85) = 753 :=
by
  sorry

end sum_digits_10_pow_85_minus_85_l127_127387


namespace choose_socks_l127_127822

open Nat

theorem choose_socks :
  (Nat.choose 8 4) = 70 :=
by 
  sorry

end choose_socks_l127_127822


namespace total_balls_l127_127299

theorem total_balls (jungkook_balls : ℕ) (yoongi_balls : ℕ) (h1 : jungkook_balls = 3) (h2 : yoongi_balls = 4) : 
  jungkook_balls + yoongi_balls = 7 :=
by
  -- This is a placeholder for the proof
  sorry

end total_balls_l127_127299


namespace y_intercept_l127_127630

theorem y_intercept (x y : ℝ) (h : 2 * x - 3 * y = 6) : x = 0 → y = -2 :=
by
  intro h₁
  sorry

end y_intercept_l127_127630


namespace older_brother_is_14_l127_127036

theorem older_brother_is_14 {Y O : ℕ} (h1 : Y + O = 26) (h2 : O = Y + 2) : O = 14 :=
by
  sorry

end older_brother_is_14_l127_127036


namespace inequality_proof_l127_127276

-- Conditions: a > b and c > d
variables {a b c d : ℝ}

-- The main statement to prove: d - a < c - b with given conditions
theorem inequality_proof (h1 : a > b) (h2 : c > d) : d - a < c - b := 
sorry

end inequality_proof_l127_127276


namespace sequence_value_a_l127_127410

theorem sequence_value_a (a : ℚ) (a_n : ℕ → ℚ)
  (h1 : a_n 1 = a) (h2 : a_n 2 = a)
  (h3 : ∀ n ≥ 3, a_n n = a_n (n - 1) + a_n (n - 2))
  (h4 : a_n 8 = 34) :
  a = 34 / 21 :=
by sorry

end sequence_value_a_l127_127410


namespace ratio_of_work_completed_by_a_l127_127640

theorem ratio_of_work_completed_by_a (A B W : ℝ) (ha : (A + B) * 6 = W) :
  (A * 3) / W = 1 / 2 :=
by 
  sorry

end ratio_of_work_completed_by_a_l127_127640


namespace least_pos_int_div_by_3_5_7_l127_127911

/-
  Prove that the least positive integer divisible by the primes 3, 5, and 7 is 105.
-/

theorem least_pos_int_div_by_3_5_7 : ∃ (n : ℕ), n > 0 ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ n = 105 :=
by 
  sorry

end least_pos_int_div_by_3_5_7_l127_127911


namespace units_digit_of_quotient_l127_127943

theorem units_digit_of_quotient : 
  (4^1985 + 7^1985) % 7 = 0 → (4^1985 + 7^1985) / 7 % 10 = 2 := 
  by 
    intro h
    sorry

end units_digit_of_quotient_l127_127943


namespace prob_tile_in_MACHINE_l127_127944

theorem prob_tile_in_MACHINE :
  let tiles := "MATHEMATICS".toList
  let machine_chars := "MACHINE".toList
  let common_chars := tiles.filter (fun c => c ∈ machine_chars)
  (common_chars.length : ℚ) / tiles.length = 7 / 11 :=
by
  -- Define the tiles and machine_chars for clarity
  let tiles := "MATHEMATICS".toList
  let machine_chars := "MACHINE".toList
  -- Filter tiles to get common characters with machine_chars
  let common_chars := tiles.filter (fun c => c ∈ machine_chars)
  -- Simplify to required fraction
  have h_len_tiles : tiles.length = 11 := by sorry
  have h_len_common : common_chars.length = 7 := by sorry
  rw [h_len_tiles, h_len_common]
  norm_num
  rfl

end prob_tile_in_MACHINE_l127_127944


namespace shaded_area_l127_127779

theorem shaded_area (r₁ r₂ r₃ : ℝ) (h₁ : 0 < r₁) (h₂ : 0 < r₂) (h₃ : 0 < r₃) (h₁₂ : r₁ < r₂) (h₂₃ : r₂ < r₃)
    (area_shaded_div_area_unshaded : (r₁^2 * π) + (r₂^2 * π) + (r₃^2 * π) = 77 * π)
    (shaded_by_unshaded_ratio : ∀ S U : ℝ, S = (3 / 7) * U) :
    ∃ S : ℝ, S = (1617 * π) / 70 :=
by
  sorry

end shaded_area_l127_127779


namespace educated_employees_count_l127_127645

def daily_wages_decrease (illiterate_avg_before illiterate_avg_after illiterate_count : ℕ) : ℕ :=
  (illiterate_avg_before - illiterate_avg_after) * illiterate_count

def total_employees (total_decreased total_avg_decreased : ℕ) : ℕ :=
  total_decreased / total_avg_decreased

theorem educated_employees_count :
  ∀ (illiterate_avg_before illiterate_avg_after illiterate_count total_avg_decreased : ℕ),
    illiterate_avg_before = 25 →
    illiterate_avg_after = 10 →
    illiterate_count = 20 →
    total_avg_decreased = 10 →
    total_employees (daily_wages_decrease illiterate_avg_before illiterate_avg_after illiterate_count) total_avg_decreased - illiterate_count = 10 :=
by
  intros
  sorry

end educated_employees_count_l127_127645


namespace solve_equation_l127_127981

theorem solve_equation :
  ∀ (x : ℝ), (1 / 7 + 7 / x = 15 / x + 1 / 15) → x = 8 :=
by
  intros x h
  sorry

end solve_equation_l127_127981


namespace carol_rectangle_width_l127_127510

theorem carol_rectangle_width 
  (area_jordan : ℕ) (length_jordan width_jordan : ℕ) (width_carol length_carol : ℕ)
  (h1 : length_jordan = 12)
  (h2 : width_jordan = 10)
  (h3 : width_carol = 24)
  (h4 : area_jordan = length_jordan * width_jordan)
  (h5 : area_jordan = length_carol * width_carol) :
  length_carol = 5 :=
by
  sorry

end carol_rectangle_width_l127_127510


namespace Moscow1964_27th_MMO_l127_127259

theorem Moscow1964_27th_MMO {a : ℤ} (h : ∀ k : ℤ, k ≠ 27 → ∃ m : ℤ, a - k^1964 = m * (27 - k)) : 
  a = 27^1964 :=
sorry

end Moscow1964_27th_MMO_l127_127259


namespace volume_uncovered_is_correct_l127_127614

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

end volume_uncovered_is_correct_l127_127614


namespace perfect_squares_divide_l127_127725

-- Define the problem and the conditions as Lean definitions
def numFactors (base exponent : ℕ) := (exponent / 2) + 1

def countPerfectSquareFactors : ℕ := 
  let choices2 := numFactors 2 3
  let choices3 := numFactors 3 5
  let choices5 := numFactors 5 7
  let choices7 := numFactors 7 9
  choices2 * choices3 * choices5 * choices7

theorem perfect_squares_divide (numFactors : (ℕ → ℕ → ℕ)) 
(countPerfectSquareFactors : ℕ) : countPerfectSquareFactors = 120 :=
by
  -- We skip the proof here
  -- Proof steps would go here if needed
  sorry

end perfect_squares_divide_l127_127725


namespace common_root_rational_l127_127219

variable (a b c d e f g : ℚ) -- coefficient variables

def poly1 (x : ℚ) : ℚ := 90 * x^4 + a * x^3 + b * x^2 + c * x + 18

def poly2 (x : ℚ) : ℚ := 18 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 90

theorem common_root_rational (k : ℚ) (h1 : poly1 a b c k = 0) (h2 : poly2 d e f g k = 0) 
  (hn : k < 0) (hi : ∀ (m n : ℤ), k ≠ m / n) : k = -1/3 := sorry

end common_root_rational_l127_127219


namespace correct_equation_l127_127507

theorem correct_equation :
  ¬ (7^3 * 7^3 = 7^9) ∧ 
  (-3^7 / 3^2 = -3^5) ∧ 
  ¬ (2^6 + (-2)^6 = 0) ∧ 
  ¬ ((-3)^5 / (-3)^3 = -3^2) :=
by 
  sorry

end correct_equation_l127_127507


namespace compare_abc_l127_127145

noncomputable def a : ℝ := Real.exp 0.25
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := -4 * Real.log 0.75

theorem compare_abc : b < c ∧ c < a := by
  -- Additional proof steps would follow here
  sorry

end compare_abc_l127_127145


namespace find_value_of_D_l127_127420

theorem find_value_of_D (C : ℕ) (D : ℕ) (k : ℕ) (h : C = (10^D) * k) (hD : k % 10 ≠ 0) : D = 69 := by
  sorry

end find_value_of_D_l127_127420


namespace difference_quotient_correct_l127_127149

theorem difference_quotient_correct (a b : ℝ) :
  abs (3 * a - b) / abs (a + 2 * b) = abs (3 * a - b) / abs (a + 2 * b) :=
by
  sorry

end difference_quotient_correct_l127_127149


namespace digit_150_after_decimal_point_l127_127785

theorem digit_150_after_decimal_point : 
  (150 : ℕ) % 3 = 0 → 
  (0.\overline{135} : ℝ) == (((5 : ℕ) : ℝ) / 37) → 
  (0.\overline{135} : list ℕ).nth 149 = some 5 :=
by 
  sorry

end digit_150_after_decimal_point_l127_127785


namespace problem_statement_l127_127865

-- Definitions for conditions
def cond_A : Prop := ∃ B : ℝ, B = 45 ∨ B = 135
def cond_B : Prop := ∃ C : ℝ, C = 90
def cond_C : Prop := false
def cond_D : Prop := ∃ B : ℝ, 0 < B ∧ B < 60

-- Prove that only cond_A has two possibilities
theorem problem_statement : cond_A ∧ ¬cond_B ∧ ¬cond_C ∧ ¬cond_D :=
by 
  -- Lean proof goes here
  sorry

end problem_statement_l127_127865


namespace exists_two_points_same_color_l127_127034

theorem exists_two_points_same_color :
  ∀ (x : ℝ), ∀ (color : ℝ × ℝ → Prop),
  (∀ (p : ℝ × ℝ), color p = red ∨ color p = blue) →
  (∃ (p1 p2 : ℝ × ℝ), dist p1 p2 = x ∧ color p1 = color p2) :=
by
  intro x color color_prop
  sorry

end exists_two_points_same_color_l127_127034


namespace length_of_uncovered_side_l127_127055

-- Define the conditions of the problem
def area_condition (L W : ℝ) : Prop := L * W = 210
def fencing_condition (L W : ℝ) : Prop := L + 2 * W = 41

-- Define the proof statement
theorem length_of_uncovered_side (L W : ℝ) (h_area : area_condition L W) (h_fence : fencing_condition L W) : 
  L = 21 :=
  sorry

end length_of_uncovered_side_l127_127055


namespace mildred_oranges_l127_127882

theorem mildred_oranges (original after given : ℕ) (h1 : original = 77) (h2 : after = 79) (h3 : given = after - original) : given = 2 :=
by
  sorry

end mildred_oranges_l127_127882


namespace abc_cube_geq_abc_sum_l127_127609

theorem abc_cube_geq_abc_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a ^ a * b ^ b * c ^ c) ^ 3 ≥ (a * b * c) ^ (a + b + c) :=
by
  sorry

end abc_cube_geq_abc_sum_l127_127609


namespace floor_sqrt_17_squared_eq_16_l127_127699

theorem floor_sqrt_17_squared_eq_16 :
  (⌊Real.sqrt 17⌋ : Real)^2 = 16 := by
  sorry

end floor_sqrt_17_squared_eq_16_l127_127699


namespace probability_of_drawing_4_black_cards_l127_127837

-- Definitions matching given conditions
def num_black_cards := 26
def num_total_cards := 52

-- Function to compute binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Definitions for specific binomial coefficients used in the problem
def choose_black := binom num_black_cards 4
def choose_total := binom num_total_cards 4

-- The probability computed as a fraction
noncomputable def probability := (choose_black : ℚ) / choose_total

-- The expected outcome of the probability
def expected_probability := (276 : ℚ) / 4998

-- Statement to be proved
theorem probability_of_drawing_4_black_cards : probability = expected_probability := 
by 
  sorry

end probability_of_drawing_4_black_cards_l127_127837


namespace total_balloons_l127_127985

-- Define the number of balloons each person has
def joan_balloons : ℕ := 40
def melanie_balloons : ℕ := 41

-- State the theorem about the total number of balloons
theorem total_balloons : joan_balloons + melanie_balloons = 81 :=
by
  sorry

end total_balloons_l127_127985


namespace fourth_year_students_without_glasses_l127_127903

theorem fourth_year_students_without_glasses (total_students: ℕ) (x: ℕ) (y: ℕ) 
  (h1: total_students = 1152) 
  (h2: total_students = 8 * x - 32) 
  (h3: x = 148) 
  (h4: 2 * y + 10 = x) 
  : y = 69 :=
by {
sorry
}

end fourth_year_students_without_glasses_l127_127903


namespace no_common_solution_l127_127951

theorem no_common_solution 
  (x : ℝ) 
  (h1 : 8 * x^2 + 6 * x = 5) 
  (h2 : 3 * x + 2 = 0) : 
  False := 
by
  sorry

end no_common_solution_l127_127951


namespace matrix_power_sub_l127_127001

section 
variable (A : Matrix (Fin 2) (Fin 2) ℝ)
variable (hA : A = ![![2, 3], ![0, 1]])

theorem matrix_power_sub (A : Matrix (Fin 2) (Fin 2) ℝ)
  (h: A = ![![2, 3], ![0, 1]]) :
  A ^ 20 - 2 * A ^ 19 = ![![0, 3], ![0, -1]] :=
by
  sorry
end

end matrix_power_sub_l127_127001


namespace basketball_players_l127_127424

theorem basketball_players {total : ℕ} (total_boys : total = 22) 
                           (football_boys : ℕ) (football_boys_count : football_boys = 15) 
                           (neither_boys : ℕ) (neither_boys_count : neither_boys = 3) 
                           (both_boys : ℕ) (both_boys_count : both_boys = 18) : 
                           (total - neither_boys = 19) := 
by
  sorry

end basketball_players_l127_127424


namespace conveyor_belt_efficiencies_and_min_cost_l127_127203

theorem conveyor_belt_efficiencies_and_min_cost :
  ∃ (efficiency_B efficiency_A : ℝ),
    efficiency_A = 1.5 * efficiency_B ∧
    18000 / efficiency_B - 18000 / efficiency_A = 10 ∧
    efficiency_B = 600 ∧
    efficiency_A = 900 ∧
    ∃ (cost_A cost_B : ℝ),
      cost_A = 8 * 20 ∧
      cost_B = 6 * 30 ∧
      cost_A = 160 ∧
      cost_B = 180 ∧
      cost_A < cost_B :=
by
  sorry

end conveyor_belt_efficiencies_and_min_cost_l127_127203


namespace solve_floor_equation_l127_127829

theorem solve_floor_equation (x : ℝ) :
  (⌊⌊2 * x⌋ - 1 / 2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
sorry

end solve_floor_equation_l127_127829


namespace y_works_in_40_days_l127_127187

theorem y_works_in_40_days :
  ∃ d, (d > 0) ∧ 
  (1/20 + 1/d = 3/40) ∧ 
  d = 40 :=
by
  use 40
  sorry

end y_works_in_40_days_l127_127187


namespace find_angle_C_l127_127735

variable (A B C : ℝ)
variable (a b c : ℝ)

theorem find_angle_C (hA : A = 39) 
                     (h_condition : (a^2 - b^2)*(a^2 + a*c - b^2) = b^2 * c^2) : 
                     C = 115 :=
sorry

end find_angle_C_l127_127735


namespace quadratic_roots_real_distinct_l127_127125

theorem quadratic_roots_real_distinct (k : ℝ) :
  (k > (1/2)) ∧ (k ≠ 1) ↔
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k-1) * x1^2 + 2 * x1 - 2 = 0) ∧ ((k-1) * x2^2 + 2 * x2 - 2 = 0)) :=
by
  sorry

end quadratic_roots_real_distinct_l127_127125


namespace determine_n_l127_127625

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem determine_n :
  (∃ n : ℕ, digit_sum (9 * (10^n - 1)) = 999 ∧ n = 111) :=
sorry

end determine_n_l127_127625


namespace smallest_b_for_45_b_square_l127_127337

theorem smallest_b_for_45_b_square :
  ∃ b : ℕ, b > 5 ∧ ∃ n : ℕ, 4 * b + 5 = n^2 ∧ b = 11 :=
by
  sorry

end smallest_b_for_45_b_square_l127_127337


namespace train_crossing_time_l127_127673

noncomputable def length_train : ℝ := 250
noncomputable def length_bridge : ℝ := 150
noncomputable def speed_train_kmh : ℝ := 57.6
noncomputable def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600)

theorem train_crossing_time : 
  let total_length := length_train + length_bridge 
  let time := total_length / speed_train_ms 
  time = 25 := 
by 
  -- Convert all necessary units and parameters
  let length_train := (250 : ℝ)
  let length_bridge := (150 : ℝ)
  let speed_train_ms := (57.6 * (1000 / 3600) : ℝ)
  
  -- Compute the total length and time
  let total_length := length_train + length_bridge
  let time := total_length / speed_train_ms
  
  -- State the proof
  show time = 25
  { sorry }

end train_crossing_time_l127_127673


namespace sebastian_older_than_jeremy_by_4_l127_127627

def J : ℕ := 40
def So : ℕ := 60 - 3
def sum_ages_in_3_years (S : ℕ) : Prop := (J + 3) + (S + 3) + (So + 3) = 150

theorem sebastian_older_than_jeremy_by_4 (S : ℕ) (h : sum_ages_in_3_years S) : S - J = 4 := by
  -- proof will be filled in
  sorry

end sebastian_older_than_jeremy_by_4_l127_127627


namespace length_of_AB_l127_127430

/-- A triangle ABC lies between two parallel lines where AC = 5 cm. Prove that AB = 10 cm. -/
noncomputable def triangle_is_between_two_parallel_lines : Prop := sorry

noncomputable def segmentAC : ℝ := 5

theorem length_of_AB :
  ∃ (AB : ℝ), triangle_is_between_two_parallel_lines ∧ segmentAC = 5 ∧ AB = 10 :=
sorry

end length_of_AB_l127_127430


namespace eugene_total_pencils_l127_127945

def initial_pencils : ℕ := 51
def additional_pencils : ℕ := 6
def total_pencils : ℕ := initial_pencils + additional_pencils

theorem eugene_total_pencils : total_pencils = 57 := by
  sorry

end eugene_total_pencils_l127_127945


namespace largest_of_given_numbers_l127_127727

theorem largest_of_given_numbers :
  ∀ (a b c d e : ℝ), a = 0.998 → b = 0.9899 → c = 0.99 → d = 0.981 → e = 0.995 →
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by
  intros a b c d e Ha Hb Hc Hd He
  rw [Ha, Hb, Hc, Hd, He]
  exact ⟨ by norm_num, by norm_num, by norm_num, by norm_num ⟩

end largest_of_given_numbers_l127_127727


namespace tetrahedron_sum_of_faces_l127_127095

theorem tetrahedron_sum_of_faces (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
(h_sum_vertices : b * c * d + a * c * d + a * b * d + a * b * c = 770) :
  a + b + c + d = 57 :=
sorry

end tetrahedron_sum_of_faces_l127_127095


namespace train_additional_time_l127_127674

theorem train_additional_time
  (t : ℝ)  -- time the car takes to reach station B
  (x : ℝ)  -- additional time the train takes compared to the car
  (h₁ : t = 4.5)  -- car takes 4.5 hours to reach station B
  (h₂ : t + (t + x) = 11)  -- combined time for both the car and the train to reach station B
  : x = 2 :=
sorry

end train_additional_time_l127_127674


namespace specific_values_exist_l127_127068

def expr_equal_for_specific_values (a b c : ℝ) : Prop :=
  a + b^2 * c = (a^2 + b) * (a + c)

theorem specific_values_exist :
  ∃ a b c : ℝ, expr_equal_for_specific_values a b c :=
sorry

end specific_values_exist_l127_127068


namespace product_of_two_numbers_l127_127466

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by {
  sorry
}

end product_of_two_numbers_l127_127466


namespace base_nine_to_base_ten_conversion_l127_127474

theorem base_nine_to_base_ten_conversion : 
  (2 * 9^3 + 8 * 9^2 + 4 * 9^1 + 7 * 9^0 = 2149) := 
by 
  sorry

end base_nine_to_base_ten_conversion_l127_127474


namespace amelia_distance_l127_127619

theorem amelia_distance (total_distance amelia_monday_distance amelia_tuesday_distance : ℕ) 
  (h1 : total_distance = 8205) 
  (h2 : amelia_monday_distance = 907) 
  (h3 : amelia_tuesday_distance = 582) : 
  total_distance - (amelia_monday_distance + amelia_tuesday_distance) = 6716 := 
by 
  sorry

end amelia_distance_l127_127619


namespace quadratic_equation_statements_l127_127952

theorem quadratic_equation_statements (a b c : ℝ) (h₀ : a ≠ 0) :
  (if -4 * a * c > 0 then (b^2 - 4 * a * c) > 0 else false) ∧
  ¬((b^2 - 4 * a * c > 0) → (b^2 - 4 * c * a > 0)) ∧
  ¬((c^2 * a + c * b + c = 0) → (a * c + b + 1 = 0)) ∧
  ¬(∀ (x₀ : ℝ), (a * x₀^2 + b * x₀ + c = 0) → (b^2 - 4 * a * c = (2 * a * x₀ - b)^2)) :=
by
    sorry

end quadratic_equation_statements_l127_127952


namespace number_line_problem_l127_127608

theorem number_line_problem (A B C : ℤ) (hA : A = -1) (hB : B = A - 5 + 6) (hC : abs (C - B) = 5) :
  C = 5 ∨ C = -5 :=
by sorry

end number_line_problem_l127_127608


namespace total_tickets_l127_127063

theorem total_tickets (n_friends : ℕ) (tickets_per_friend : ℕ) (h1 : n_friends = 6) (h2 : tickets_per_friend = 39) : n_friends * tickets_per_friend = 234 :=
by
  -- Place for proof, to be constructed
  sorry

end total_tickets_l127_127063


namespace no_perfect_squares_in_seq_l127_127839

def seq (x : ℕ → ℤ) : Prop :=
  x 0 = 1 ∧ x 1 = 3 ∧ ∀ n : ℕ, 0 < n → x (n + 1) = 6 * x n - x (n - 1)

theorem no_perfect_squares_in_seq (x : ℕ → ℤ) (n : ℕ) (h_seq : seq x) :
  ¬ ∃ k : ℤ, k * k = x (n + 1) :=
by
  sorry

end no_perfect_squares_in_seq_l127_127839


namespace agatha_initial_money_l127_127061

/-
Agatha has some money to spend on a new bike. She spends $15 on the frame, and $25 on the front wheel.
If she has $20 left to spend on a seat and handlebar tape, prove that she had $60 initially.
-/

theorem agatha_initial_money (frame_cost wheel_cost remaining_money initial_money: ℕ) 
  (h1 : frame_cost = 15) 
  (h2 : wheel_cost = 25) 
  (h3 : remaining_money = 20) 
  (h4 : initial_money = frame_cost + wheel_cost + remaining_money) : 
  initial_money = 60 :=
by {
  -- We state explicitly that initial_money should be 60
  sorry
}

end agatha_initial_money_l127_127061


namespace number_of_bikes_l127_127141

theorem number_of_bikes (total_wheels : ℕ) (car_wheels : ℕ) (tricycle_wheels : ℕ) (roller_skate_wheels : ℕ) (trash_can_wheels : ℕ) (bike_wheels : ℕ) (num_bikes : ℕ) :
  total_wheels = 25 →
  car_wheels = 2 * 4 →
  tricycle_wheels = 3 →
  roller_skate_wheels = 4 →
  trash_can_wheels = 2 →
  bike_wheels = 2 →
  (total_wheels - (car_wheels + tricycle_wheels + roller_skate_wheels + trash_can_wheels)) = bike_wheels * num_bikes →
  num_bikes = 4 := 
by
  intros total_wheels_eq total_car_wheels_eq tricycle_wheels_eq roller_skate_wheels_eq trash_can_wheels_eq bike_wheels_eq remaining_wheels_eq
  sorry

end number_of_bikes_l127_127141


namespace not_perfect_square_7_pow_2025_all_others_perfect_squares_l127_127181

theorem not_perfect_square_7_pow_2025 :
  ¬ (∃ x : ℕ, x^2 = 7^2025) :=
sorry

theorem all_others_perfect_squares :
  (∃ x : ℕ, x^2 = 6^2024) ∧
  (∃ x : ℕ, x^2 = 8^2026) ∧
  (∃ x : ℕ, x^2 = 9^2027) ∧
  (∃ x : ℕ, x^2 = 10^2028) :=
sorry

end not_perfect_square_7_pow_2025_all_others_perfect_squares_l127_127181


namespace digit_150_of_5_div_37_is_5_l127_127788

theorem digit_150_of_5_div_37_is_5 : 
  ( ( 0.135135135...) ) .( ( 150 ) ) = 5 :=
sorry

end digit_150_of_5_div_37_is_5_l127_127788


namespace green_face_probability_l127_127460

def probability_of_green_face (total_faces green_faces : Nat) : ℚ :=
  green_faces / total_faces

theorem green_face_probability :
  let total_faces := 10
  let green_faces := 3
  let blue_faces := 5
  let red_faces := 2
  probability_of_green_face total_faces green_faces = 3/10 :=
by
  sorry

end green_face_probability_l127_127460


namespace coefficients_balance_l127_127146

noncomputable def num_positive_coeffs (n : ℕ) : ℕ :=
  n + 1

noncomputable def num_negative_coeffs (n : ℕ) : ℕ :=
  n + 1

theorem coefficients_balance (n : ℕ) (h_odd: Odd n) (x : ℝ) :
  num_positive_coeffs n = num_negative_coeffs n :=
by
  sorry

end coefficients_balance_l127_127146


namespace product_of_two_numbers_l127_127467

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by {
  sorry
}

end product_of_two_numbers_l127_127467


namespace arjun_starting_amount_l127_127202

theorem arjun_starting_amount (X : ℝ) (h1 : Anoop_investment = 4000) (h2 : Anoop_months = 6) (h3 : Arjun_months = 12) (h4 : (X * 12) = (4000 * 6)) :
  X = 2000 :=
sorry

end arjun_starting_amount_l127_127202


namespace smallest_divisible_four_digit_number_l127_127090

theorem smallest_divisible_four_digit_number :
  let n := 2310
  in ∃ k : ℕ, n * k >= 1000 ∧ n = 2 * 3 * 5 * 7 * 11 :=
begin
  let n := 2310,
  use 1,
  split,
  { norm_num },
  { norm_num },
  sorry
end

end smallest_divisible_four_digit_number_l127_127090


namespace vector_addition_parallel_l127_127965

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem vector_addition_parallel:
  ∀ x : ℝ, parallel (2, 1) (x, -2) → a + b x = ((-2 : ℝ), -1) :=
by
  intros x h
  sorry

end vector_addition_parallel_l127_127965


namespace repeating_mul_l127_127701

theorem repeating_mul (x y : ℚ) (h1 : x = (12 : ℚ) / 99) (h2 : y = (34 : ℚ) / 99) : 
    x * y = (136 : ℚ) / 3267 := by
  sorry

end repeating_mul_l127_127701


namespace cashier_five_dollar_bills_l127_127492

-- Define the conditions as a structure
structure CashierBills (x y : ℕ) : Prop :=
(total_bills : x + y = 126)
(total_value : 5 * x + 10 * y = 840)

-- State the theorem that we need to prove
theorem cashier_five_dollar_bills (x y : ℕ) (h : CashierBills x y) : x = 84 :=
sorry

end cashier_five_dollar_bills_l127_127492


namespace primes_sq_not_divisible_by_p_l127_127694

theorem primes_sq_not_divisible_by_p (p : ℕ) [Fact p.Prime] :
  (∀ (a : ℤ), ¬(p ∣ a) → (a^2 % p = 1 % p)) ↔ (p = 2 ∨ p = 3) :=
by
  sorry

end primes_sq_not_divisible_by_p_l127_127694


namespace power_function_solution_l127_127851

theorem power_function_solution (m : ℤ)
  (h1 : ∃ (f : ℝ → ℝ), ∀ x : ℝ, f x = x^(-m^2 + 2 * m + 3) ∧ ∀ x, f x = f (-x))
  (h2 : ∀ x : ℝ, x > 0 → (x^(-m^2 + 2 * m + 3)) < x^(-m^2 + 2 * m + 3 + x)) :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f x = x^4 :=
by
  sorry

end power_function_solution_l127_127851


namespace sum_of_solutions_l127_127913

  theorem sum_of_solutions :
    (∃ x : ℝ, x = abs (2 * x - abs (50 - 2 * x)) ∧ ∃ y : ℝ, y = abs (2 * y - abs (50 - 2 * y)) ∧ ∃ z : ℝ, z = abs (2 * z - abs (50 - 2 * z)) ∧ (x + y + z = 170 / 3)) :=
  sorry
  
end sum_of_solutions_l127_127913


namespace probability_B_receives_once_after_three_passes_l127_127018

-- Definitions
def ball_passing_sequence : Type := List (List Char)
def total_sequences (n : Nat) (choices : Nat) := choices^n
def valid_sequences (sequences : ball_passing_sequence) (n : Nat) : ball_passing_sequence :=
  (sequences.filter (λ seq => seq.count ('B') = 1))

-- The main theorem to prove
theorem probability_B_receives_once_after_three_passes :
  let sequences := (['A', 'B', 'C', 'D']).bind (λ x => ['A', 'B', 'C', 'D']) ++ (['A', 'B', 'C', 'D'])
  let relevant_sequences := valid_sequences sequences 3
  (relevant_sequences.length : ℚ) / (total_sequences 3 3 : ℚ) = 16 / 27 :=
sorry

end probability_B_receives_once_after_three_passes_l127_127018


namespace circle_area_from_equation_l127_127333

theorem circle_area_from_equation :
  (∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = -9) →
  ∃ (r : ℝ), (r = 2) ∧
    (∃ (A : ℝ), A = π * r^2 ∧ A = 4 * π) :=
by {
  -- Conditions included as hypothesis
  sorry -- Proof to be provided here
}

end circle_area_from_equation_l127_127333


namespace expensive_time_8_l127_127755

variable (x : ℝ) -- x represents the time to pick an expensive handcuff lock

-- Conditions
def cheap_time := 6
def total_time := 42
def cheap_pairs := 3
def expensive_pairs := 3

-- Total time for cheap handcuffs
def total_cheap_time := cheap_pairs * cheap_time

-- Total time for expensive handcuffs
def total_expensive_time := total_time - total_cheap_time

-- Equation relating x to total_expensive_time
def expensive_equation := expensive_pairs * x = total_expensive_time

-- Proof goal
theorem expensive_time_8 : expensive_equation x -> x = 8 := by
  sorry

end expensive_time_8_l127_127755


namespace even_function_solution_l127_127423

theorem even_function_solution :
  ∀ (m : ℝ), (∀ x : ℝ, (m+1) * x^2 + (m-2) * x = (m+1) * x^2 - (m-2) * x) → (m = 2 ∧ ∀ x : ℝ, (2+1) * x^2 + (2-2) * x = 3 * x^2) :=
by
  sorry

end even_function_solution_l127_127423


namespace sum_lent_correct_l127_127190

noncomputable section

-- Define the principal amount (sum lent)
def P : ℝ := 4464.29

-- Define the interest rate per annum
def R : ℝ := 12.0

-- Define the time period in years
def T : ℝ := 12.0

-- Define the interest after 12 years (using the initial conditions and results)
def I : ℝ := 1.44 * P

-- Define the interest given as "2500 less than double the sum lent" condition
def I_condition : ℝ := 2 * P - 2500

-- Theorem stating the sum lent is the given value P
theorem sum_lent_correct : P = 4464.29 :=
by
  -- Placeholder for the proof
  sorry

end sum_lent_correct_l127_127190


namespace jacob_has_more_money_l127_127009

def exchange_rate : ℝ := 1.11
def Mrs_Hilt_total_in_dollars : ℝ := 
  3 * 0.01 + 2 * 0.10 + 2 * 0.05 + 5 * 0.25 + 1 * 1.00

def Jacob_total_in_euros : ℝ := 
  4 * 0.01 + 1 * 0.05 + 1 * 0.10 + 3 * 0.20 + 2 * 0.50 + 2 * 1.00

def Jacob_total_in_dollars : ℝ := Jacob_total_in_euros * exchange_rate

def difference : ℝ := Jacob_total_in_dollars - Mrs_Hilt_total_in_dollars

theorem jacob_has_more_money : difference = 1.63 :=
by sorry

end jacob_has_more_money_l127_127009


namespace center_of_circle_is_at_10_3_neg5_l127_127921

noncomputable def center_of_tangent_circle (x y : ℝ) : Prop :=
  (6 * x - 5 * y = 50 ∨ 6 * x - 5 * y = -20) ∧ (3 * x + 2 * y = 0)

theorem center_of_circle_is_at_10_3_neg5 :
  ∃ x y : ℝ, center_of_tangent_circle x y ∧ x = 10 / 3 ∧ y = -5 :=
by
  sorry

end center_of_circle_is_at_10_3_neg5_l127_127921


namespace gcd_of_4410_and_10800_l127_127819

theorem gcd_of_4410_and_10800 : Nat.gcd 4410 10800 = 90 := 
by 
  sorry

end gcd_of_4410_and_10800_l127_127819


namespace trapezoid_LM_sqrt2_l127_127294

theorem trapezoid_LM_sqrt2 (K L M N P Q : Point) : 
  ∀ (h_trapezoid : is_trapezoid K L M N) 
     (diag_eq_height : distance K M = 1 ∧ height_trapezoid K L M N = 1) 
     (perp_KP_MQ : is_perpendicular(K P MN) ∧ is_perpendicular(M Q KL)) 
     (KN_MQ_eq : distance K N = distance M Q) 
     (LM_MP_eq : distance L M = distance M P), 
  distance L M = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l127_127294


namespace log35_28_l127_127524

variable (a b : ℝ)
variable (log : ℝ → ℝ → ℝ)

-- Conditions
axiom log14_7_eq_a : log 14 7 = a
axiom log14_5_eq_b : log 14 5 = b

-- Theorem to prove
theorem log35_28 (h1 : log 14 7 = a) (h2 : log 14 5 = b) : log 35 28 = (2 - a) / (a + b) :=
sorry

end log35_28_l127_127524


namespace trapezoid_LM_value_l127_127291

theorem trapezoid_LM_value (K L M N P Q : Type) 
  (d1 d2 : ℝ)
  (h1 : d1 = 1)
  (h2 : d2 = 1)
  (height_eq : KM = 1)
  (KN_eq_MQ : KN = MQ)
  (LM_eq_MP : LM = MP) :
  LM = 1 / real.sqrt (real.sqrt 2) :=
by 
  sorry

end trapezoid_LM_value_l127_127291


namespace number_of_paths_l127_127689

/-
We need to define the conditions and the main theorem
-/

def grid_width : ℕ := 5
def grid_height : ℕ := 4
def total_steps : ℕ := 8
def steps_right : ℕ := 5
def steps_up : ℕ := 3

theorem number_of_paths : (Nat.choose total_steps steps_up) = 56 := by
  sorry

end number_of_paths_l127_127689


namespace binomial_20_10_l127_127540

open Nat

theorem binomial_20_10 :
  (binomial 18 8 = 43758) →
  (binomial 18 9 = 48620) →
  (binomial 18 10 = 43758) →
  binomial 20 10 = 184756 :=
by
  intros h1 h2 h3
  sorry

end binomial_20_10_l127_127540


namespace binom_20_10_l127_127536

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end binom_20_10_l127_127536


namespace second_customer_payment_l127_127661

def price_of_headphones : ℕ := 30
def total_cost_first_customer (P H : ℕ) : ℕ := 5 * P + 8 * H
def total_cost_second_customer (P H : ℕ) : ℕ := 3 * P + 4 * H

theorem second_customer_payment
  (P : ℕ)
  (H_eq : H = price_of_headphones)
  (first_customer_eq : total_cost_first_customer P H = 840) :
  total_cost_second_customer P H = 480 :=
by
  -- Proof to be filled in later
  sorry

end second_customer_payment_l127_127661


namespace number_of_pairs_l127_127241

theorem number_of_pairs (H : ∀ x y : ℕ , 0 < x → 0 < y → x < y → 2 * x * y / (x + y) = 4 ^ 15) :
  ∃ n : ℕ, n = 29 :=
by
  sorry

end number_of_pairs_l127_127241


namespace simplify_expression_l127_127686

theorem simplify_expression : 4 * Real.sqrt (1 / 2) + 3 * Real.sqrt (1 / 3) - Real.sqrt 8 = Real.sqrt 3 := 
by 
  sorry

end simplify_expression_l127_127686


namespace distinct_four_digit_numbers_product_18_l127_127569

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end distinct_four_digit_numbers_product_18_l127_127569


namespace log_inequality_l127_127439

theorem log_inequality
  (a : ℝ := Real.log 4 / Real.log 5)
  (b : ℝ := (Real.log 3 / Real.log 5)^2)
  (c : ℝ := Real.log 5 / Real.log 4) :
  b < a ∧ a < c :=
by
  sorry

end log_inequality_l127_127439


namespace problem_correct_l127_127216

theorem problem_correct (x : ℝ) : 
  14 * ((150 / 3) + (35 / 7) + (16 / 32) + x) = 777 + 14 * x := 
by
  sorry

end problem_correct_l127_127216


namespace total_crayons_l127_127120

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
      (h1 : crayons_per_child = 18) (h2 : num_children = 36) : 
        crayons_per_child * num_children = 648 := by
  sorry

end total_crayons_l127_127120


namespace problem1_problem2_l127_127850

-- Definitions related to the given problem
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 9 / (Real.cos θ^2 + 9 * Real.sin θ^2)

def standard_curve (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Proving the standard equation of the curve
theorem problem1 (ρ θ : ℝ) (h : polar_curve ρ θ) : ∃ x y, standard_curve x y :=
  sorry

-- Proving the perpendicular condition and its consequence
theorem problem2 (ρ1 ρ2 α : ℝ)
  (hA : polar_curve ρ1 α)
  (hB : polar_curve ρ2 (α + π/2))
  (perpendicular : ∀ (A B : (ℝ × ℝ)), A ≠ B → A.1 * B.1 + A.2 * B.2 = 0) :
  (1 / ρ1^2) + (1 / ρ2^2) = 10 / 9 :=
  sorry

end problem1_problem2_l127_127850


namespace common_divisors_60_90_l127_127855

theorem common_divisors_60_90 :
  ∃ (count : ℕ), 
  (∀ d, d ∣ 60 ∧ d ∣ 90 ↔ d ∈ {1, 2, 3, 5, 6, 10, 15, 30}) ∧ 
  count = 8 :=
by
  sorry

end common_divisors_60_90_l127_127855


namespace line_eq_l127_127893

theorem line_eq (m b : ℝ) 
  (h_slope : m = (4 + 2) / (3 - 1)) 
  (h_point : -2 = m * 1 + b) :
  m + b = -2 :=
by
  sorry

end line_eq_l127_127893


namespace sqrt_subtraction_result_l127_127914

theorem sqrt_subtraction_result : 
  (Real.sqrt (49 + 36) - Real.sqrt (36 - 0)) = 4 :=
by
  sorry

end sqrt_subtraction_result_l127_127914


namespace sequence_bound_l127_127250

/-- This definition states that given the initial conditions and recurrence relation
for a sequence of positive integers, the 2021st term is greater than 2^2019. -/
theorem sequence_bound (a : ℕ → ℕ) (h_initial : a 2 > a 1)
  (h_recurrence : ∀ n, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 2021 > 2 ^ 2019 :=
sorry

end sequence_bound_l127_127250


namespace prob_contestant_A_makes_it_to_final_expected_value_of_xi_l127_127503

/-- Problem conditions -/

def prob_correct : ℚ := 2 / 3
def prob_incorrect : ℚ := 1 / 3

def make_it_to_final_3_correct : ℚ :=
(2 / 3) ^ 3

def make_it_to_final_4_questions : ℚ :=
(binomial 3 2) * (2 / 3)^2 * (1 / 3) * (2 / 3)

def make_it_to_final_5_questions : ℚ :=
(binomial 4 2) * (2 / 3) ^ 3 * (1 / 3) ^ 2

def prob_make_it_to_final : ℚ :=
make_it_to_final_3_correct + make_it_to_final_4_questions + make_it_to_final_5_questions

/-- Proof statement (1) -/
theorem prob_contestant_A_makes_it_to_final
  (p_correct : prob_correct = 2 / 3)
  (p_incorrect : prob_incorrect = 1 / 3)
  : prob_make_it_to_final = 64 / 81 := by
  sorry

def xi_dist_3 : ℚ := ((2 / 3) ^ 3) + (1 / 3) ^ 3
def xi_dist_4 : ℚ := (binomial 3 2) * (2 / 3)^2 * (1 / 3) * (2 / 3) + (binomial 3 2) * (1 / 3)^2 * (2 / 3) * (1 / 3)
def xi_dist_5 : ℚ := (binomial 4 2) * (2 / 3) ^ 3 * (1 / 3) ^ 2 + (binomial 4 2) * (2 / 3) ^ 2 * (1 / 3) ^ 3

def expected_value_xi : ℚ :=
3 * xi_dist_3 + 4 * xi_dist_4 + 5 * xi_dist_5

/-- Proof statement (2) -/
theorem expected_value_of_xi
  (dist_3 : xi_dist_3 = 1 / 3)
  (dist_4 : xi_dist_4 = 10 / 27)
  (dist_5 : xi_dist_5 = 8 / 27)
  : expected_value_xi = 107 / 27 := by
  sorry

end prob_contestant_A_makes_it_to_final_expected_value_of_xi_l127_127503


namespace tom_days_to_finish_l127_127906

noncomputable def days_to_finish_show
  (episodes : Nat) 
  (minutes_per_episode : Nat) 
  (hours_per_day : Nat) : Nat :=
  let total_minutes := episodes * minutes_per_episode
  let total_hours := total_minutes / 60
  total_hours / hours_per_day

theorem tom_days_to_finish :
  days_to_finish_show 90 20 2 = 15 :=
by
  -- the proof steps go here
  sorry

end tom_days_to_finish_l127_127906


namespace polynomial_division_properties_l127_127999

open Polynomial

noncomputable def g : Polynomial ℝ := 3 * X^4 + 9 * X^3 - 7 * X^2 + 2 * X + 5
noncomputable def e : Polynomial ℝ := X^2 + 2 * X - 3

theorem polynomial_division_properties (s t : Polynomial ℝ) (h : g = s * e + t) (h_deg : t.degree < e.degree) :
  s.eval 1 + t.eval (-1) = -22 :=
sorry

end polynomial_division_properties_l127_127999


namespace dewei_less_than_daliah_l127_127691

theorem dewei_less_than_daliah
  (daliah_amount : ℝ := 17.5)
  (zane_amount : ℝ := 62)
  (zane_multiple_dewei : zane_amount = 4 * (zane_amount / 4)) :
  (daliah_amount - (zane_amount / 4)) = 2 :=
by
  sorry

end dewei_less_than_daliah_l127_127691


namespace odd_function_neg_value_l127_127115

theorem odd_function_neg_value
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 ≤ x → f x = x^2 + 2 * x) :
  ∀ x, x < 0 → f x = -x^2 + 2 * x :=
by
  intros x hx
  -- The proof would go here
  sorry

end odd_function_neg_value_l127_127115


namespace boys_speed_l127_127490

-- Define the conditions
def sideLength : ℕ := 50
def timeTaken : ℕ := 72

-- Define the goal
theorem boys_speed (sideLength timeTaken : ℕ) (D T : ℝ) :
  D = (4 * sideLength : ℕ) / 1000 ∧
  T = timeTaken / 3600 →
  (D / T = 10) := by
  sorry

end boys_speed_l127_127490


namespace c_investment_ratio_l127_127364

-- Conditions as definitions
variables (x : ℕ) (m : ℕ) (total_profit a_share : ℕ)
variables (h_total_profit : total_profit = 19200)
variables (h_a_share : a_share = 6400)

-- Definition of total investment (investments weighted by time)
def total_investment (x m : ℕ) : ℕ :=
  (12 * x) + (6 * 2 * x) + (4 * m * x)

-- Definition of A's share in terms of total investment
def a_share_in_terms_of_total_investment (x : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (12 * x * total_profit) / total_investment

-- The theorem stating the ratio of C's investment to A's investment
theorem c_investment_ratio (x m total_profit a_share : ℕ) (h_total_profit : total_profit = 19200)
  (h_a_share : a_share = 6400) (h_a_share_eq : a_share_in_terms_of_total_investment x (total_investment x m) total_profit = a_share) :
  m = 3 :=
by sorry

end c_investment_ratio_l127_127364


namespace f_zero_unique_l127_127895

theorem f_zero_unique (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f x + f (xy)) : f 0 = 0 :=
by {
  -- proof goes here
  sorry
}

end f_zero_unique_l127_127895


namespace sum_and_count_evens_20_30_l127_127644

def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_evens_20_30 :
  let x := sum_integers 20 30
  let y := count_even_integers 20 30
  x + y = 281 :=
by
  sorry

end sum_and_count_evens_20_30_l127_127644


namespace probability_sunglasses_and_cap_l127_127011

-- We denote the number of people wearing sunglasses, caps, and the probability condition.
def numSunglasses : ℕ := 80
def numCaps : ℕ := 60
def probCapThenSunglasses : ℚ := 1 / 3

-- Definition of the number of people wearing both sunglasses and caps.
def numBothSunglassesAndCaps : ℕ := numCaps * probCapThenSunglasses

-- The main theorem: Finding the probability of wearing sunglasses and also a cap.
theorem probability_sunglasses_and_cap :
  numBothSunglassesAndCaps / numSunglasses = 1 / 4 :=
by
  sorry

end probability_sunglasses_and_cap_l127_127011


namespace quadratic_distinct_roots_k_range_l127_127123

theorem quadratic_distinct_roots_k_range (k : ℝ) :
  (k - 1) * x^2 + 2 * x - 2 = 0 ∧ 
  ∀ Δ, Δ = 2^2 - 4*(k-1)*(-2) ∧ Δ > 0 ∧ (k ≠ 1) ↔ k > 1/2 ∧ k ≠ 1 :=
by
  sorry

end quadratic_distinct_roots_k_range_l127_127123


namespace find_n_l127_127317

noncomputable def cube_probability_solid_color (num_cubes edge_length num_corner num_edge num_face_center num_center : ℕ)
  (corner_prob edge_prob face_center_prob center_prob : ℚ) : ℚ :=
  have total_corner_prob := corner_prob ^ num_corner
  have total_edge_prob := edge_prob ^ num_edge
  have total_face_center_prob := face_center_prob ^ num_face_center
  have total_center_prob := center_prob ^ num_center
  2 * (total_corner_prob * total_edge_prob * total_face_center_prob * total_center_prob)

theorem find_n : ∃ n : ℕ, cube_probability_solid_color 27 3 8 12 6 1
  (1/8) (1/4) (1/2) 1 = (1 / (2 : ℚ) ^ n) ∧ n = 53 := by
  use 53
  simp only [cube_probability_solid_color]
  sorry

end find_n_l127_127317


namespace num_games_last_year_l127_127612

-- Definitions from conditions
def num_games_this_year : ℕ := 14
def total_num_games : ℕ := 43

-- Theorem to prove
theorem num_games_last_year (num_games_last_year : ℕ) : 
  total_num_games - num_games_this_year = num_games_last_year ↔ num_games_last_year = 29 :=
by
  sorry

end num_games_last_year_l127_127612


namespace calculate_A_minus_B_l127_127363

variable (A B : ℝ)
variable (h1 : A + B + B = 814.8)
variable (h2 : 10 * B = A)

theorem calculate_A_minus_B : A - B = 611.1 :=
by
  sorry

end calculate_A_minus_B_l127_127363


namespace simplify_fractions_l127_127014

theorem simplify_fractions :
  (270 / 18) * (7 / 210) * (9 / 4) = 9 / 8 :=
by sorry

end simplify_fractions_l127_127014


namespace polynomial_comparison_l127_127552

theorem polynomial_comparison {x : ℝ} :
  let A := (x - 3) * (x - 2)
  let B := (x + 1) * (x - 6)
  A > B :=
by 
  sorry -- Proof is omitted.

end polynomial_comparison_l127_127552


namespace sum_of_interior_angles_increases_l127_127176

theorem sum_of_interior_angles_increases (n : ℕ) (h : n ≥ 3) : (n-2) * 180 > (n-3) * 180 :=
by
  sorry

end sum_of_interior_angles_increases_l127_127176


namespace domain_f1_correct_f2_correct_f2_at_3_l127_127347

noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt (4 - 2 * x) + 1 + 1 / (x + 1)

noncomputable def domain_f1 : Set ℝ := {x | 4 - 2 * x ≥ 0} \ (insert 1 (insert (-1) {}))

theorem domain_f1_correct : domain_f1 = { x | x ≤ 2 ∧ x ≠ 1 ∧ x ≠ -1 } :=
by
  sorry

noncomputable def f2 (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem f2_correct : ∀ x, f2 (x + 1) = x^2 - 2 * x :=
by
  sorry

theorem f2_at_3 : f2 3 = 0 :=
by
  sorry

end domain_f1_correct_f2_correct_f2_at_3_l127_127347


namespace find_x_value_l127_127800

theorem find_x_value (x y z k: ℚ)
  (h1 : x = k * (z^3) / (y^2))
  (h2 : y = 2) (h3 : z = 3)
  (h4 : x = 1)
  : x = (4 / 27) * (4^3) / (6^2) := by
  sorry

end find_x_value_l127_127800


namespace binom_20_10_eq_184756_l127_127545

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end binom_20_10_eq_184756_l127_127545


namespace sin_double_angle_l127_127253

variable {α : Real}

theorem sin_double_angle (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
sorry

end sin_double_angle_l127_127253


namespace mixed_solution_concentration_l127_127976

def salt_amount_solution1 (weight1 : ℕ) (concentration1 : ℕ) : ℕ := (concentration1 * weight1) / 100
def salt_amount_solution2 (salt2 : ℕ) : ℕ := salt2
def total_salt (salt1 salt2 : ℕ) : ℕ := salt1 + salt2
def total_weight (weight1 weight2 : ℕ) : ℕ := weight1 + weight2
def concentration (total_salt : ℕ) (total_weight : ℕ) : ℚ := (total_salt : ℚ) / (total_weight : ℚ) * 100

theorem mixed_solution_concentration 
  (weight1 weight2 salt2 : ℕ) (concentration1 : ℕ)
  (h_weight1 : weight1 = 200)
  (h_weight2 : weight2 = 300)
  (h_concentration1 : concentration1 = 25)
  (h_salt2 : salt2 = 60) :
  concentration (total_salt (salt_amount_solution1 weight1 concentration1) (salt_amount_solution2 salt2)) (total_weight weight1 weight2) = 22 := 
sorry

end mixed_solution_concentration_l127_127976


namespace students_with_two_skills_l127_127224

theorem students_with_two_skills :
  ∀ (n_students n_chess n_puzzles n_code : ℕ),
  n_students = 120 →
  n_chess = n_students - 50 →
  n_puzzles = n_students - 75 →
  n_code = n_students - 40 →
  (n_chess + n_puzzles + n_code - n_students) = 75 :=
by 
  sorry

end students_with_two_skills_l127_127224


namespace smallest_four_digit_divisible_five_smallest_primes_l127_127077

theorem smallest_four_digit_divisible_five_smallest_primes :
  let lcm_5_min_primes := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 (Nat.lcm 7 11))
  ∀ m : ℕ, (m ≥ 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ m)) → m = 2310 :=
by
  let lcm_5_min_primes := 2310
  intro m h
  have h1 : m = 2310 := sorry
  exact h1

end smallest_four_digit_divisible_five_smallest_primes_l127_127077


namespace chloe_profit_l127_127377

def cost_per_dozen : ℕ := 50
def sell_per_half_dozen : ℕ := 30
def total_dozens_sold : ℕ := 50

def total_cost (n: ℕ) : ℕ := n * cost_per_dozen
def total_revenue (n: ℕ) : ℕ := n * (sell_per_half_dozen * 2)
def profit (cost revenue : ℕ) : ℕ := revenue - cost

theorem chloe_profit : 
  profit (total_cost total_dozens_sold) (total_revenue total_dozens_sold) = 500 := 
by
  sorry

end chloe_profit_l127_127377


namespace smallest_four_digit_number_divisible_by_smallest_primes_l127_127087

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end smallest_four_digit_number_divisible_by_smallest_primes_l127_127087


namespace discountIs50Percent_l127_127979

noncomputable def promotionalPrice (originalPrice : ℝ) : ℝ :=
  (2/3) * originalPrice

noncomputable def finalPrice (originalPrice : ℝ) : ℝ :=
  0.75 * promotionalPrice originalPrice

theorem discountIs50Percent (originalPrice : ℝ) (h₁ : originalPrice > 0) :
  finalPrice originalPrice = 0.5 * originalPrice := by
  sorry

end discountIs50Percent_l127_127979


namespace length_of_symmedian_l127_127515

theorem length_of_symmedian (a b c : ℝ) (AS : ℝ) :
  AS = (2 * b * c^2) / (b^2 + c^2) := sorry

end length_of_symmedian_l127_127515


namespace subset_bound_l127_127752

theorem subset_bound {m n k : ℕ} (h1 : m ≥ n) (h2 : n > 1) 
  (F : Fin k → Finset (Fin m)) 
  (hF : ∀ i j, i < j → (F i ∩ F j).card ≤ 1) 
  (hcard : ∀ i, (F i).card = n) : 
  k ≤ (m * (m - 1)) / (n * (n - 1)) :=
sorry

end subset_bound_l127_127752


namespace geom_S4_eq_2S2_iff_abs_q_eq_1_l127_127954

variable {α : Type*} [LinearOrderedField α]

-- defining the sum of first n terms of a geometric sequence
def geom_series_sum (a q : α) (n : ℕ) :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

noncomputable def S (a q : α) (n : ℕ) := geom_series_sum a q n

theorem geom_S4_eq_2S2_iff_abs_q_eq_1 
  (a q : α) : 
  S a q 4 = 2 * S a q 2 ↔ |q| = 1 :=
sorry

end geom_S4_eq_2S2_iff_abs_q_eq_1_l127_127954


namespace polynomial_factorization_l127_127995

theorem polynomial_factorization (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + ab + ac + b^2 + bc + c^2) :=
sorry

end polynomial_factorization_l127_127995


namespace division_of_fractions_l127_127854

theorem division_of_fractions : (1 / 10) / (1 / 5) = 1 / 2 :=
by
  sorry

end division_of_fractions_l127_127854


namespace divide_plane_into_regions_l127_127528

theorem divide_plane_into_regions (n : ℕ) (h₁ : n < 199) (h₂ : ∃ (k : ℕ), k = 99):
  n = 100 ∨ n = 198 :=
sorry

end divide_plane_into_regions_l127_127528


namespace total_messages_l127_127680

theorem total_messages (l1 l2 l3 a1 a2 a3 : ℕ)
  (h1 : l1 = 120)
  (h2 : a1 = l1 - 20)
  (h3 : l2 = l1 / 3)
  (h4 : a2 = 2 * a1)
  (h5 : l3 = l1)
  (h6 : a3 = a1) :
  l1 + l2 + l3 + a1 + a2 + a3 = 680 :=
by
  -- Proof steps would go here. Adding 'sorry' to skip proof.
  sorry

end total_messages_l127_127680


namespace set_intersection_M_N_l127_127101

theorem set_intersection_M_N (x : ℝ) :
  let M := {x | -4 < x ∧ x < -2}
  let N := {x | x^2 + 5*x + 6 < 0}
  M ∩ N = {x | -3 < x ∧ x < -2} :=
by
  sorry

end set_intersection_M_N_l127_127101


namespace cylinder_radius_range_l127_127130

theorem cylinder_radius_range :
  (V : ℝ) → (h : ℝ) → (r : ℝ) →
  V = 20 * Real.pi →
  h = 2 →
  (V = Real.pi * r^2 * h) →
  3 < r ∧ r < 4 :=
by
  -- Placeholder for the proof
  intro V h r hV hh hV_eq
  sorry

end cylinder_radius_range_l127_127130


namespace total_samples_correct_l127_127064

-- Define the conditions as constants
def samples_per_shelf : ℕ := 65
def number_of_shelves : ℕ := 7

-- Define the total number of samples and the expected result
def total_samples : ℕ := samples_per_shelf * number_of_shelves
def expected_samples : ℕ := 455

-- State the theorem to be proved
theorem total_samples_correct : total_samples = expected_samples := by
  -- Proof to be filled in
  sorry

end total_samples_correct_l127_127064


namespace ratio_of_sums_l127_127415

theorem ratio_of_sums (a b c u v w : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_pos_u : 0 < u) (h_pos_v : 0 < v) (h_pos_w : 0 < w)
    (h1 : a^2 + b^2 + c^2 = 9) (h2 : u^2 + v^2 + w^2 = 49) (h3 : a * u + b * v + c * w = 21) : 
    (a + b + c) / (u + v + w) = 3 / 7 := 
by
  sorry

end ratio_of_sums_l127_127415


namespace sobhas_parents_age_difference_l127_127157

def difference_in_ages (F M : ℕ) : ℕ := F - M

theorem sobhas_parents_age_difference
  (S F M : ℕ)
  (h1 : F = S + 38)
  (h2 : M = S + 32) :
  difference_in_ages F M = 6 := by
  sorry

end sobhas_parents_age_difference_l127_127157


namespace fruit_display_total_l127_127473

-- Define the number of bananas
def bananas : ℕ := 5

-- Define the number of oranges, twice the number of bananas
def oranges : ℕ := 2 * bananas

-- Define the number of apples, twice the number of oranges
def apples : ℕ := 2 * oranges

-- Define the total number of fruits
def total_fruits : ℕ := bananas + oranges + apples

-- Theorem statement: Prove that the total number of fruits is 35
theorem fruit_display_total : total_fruits = 35 := 
by
  -- proof will be skipped, but all necessary conditions and definitions are included to ensure the statement is valid
  sorry

end fruit_display_total_l127_127473


namespace geometric_sequence_fifth_term_l127_127040

theorem geometric_sequence_fifth_term (a₁ r : ℤ) (n : ℕ) (h_a₁ : a₁ = 5) (h_r : r = -2) (h_n : n = 5) :
  (a₁ * r^(n-1) = 80) :=
by
  rw [h_a₁, h_r, h_n]
  sorry

end geometric_sequence_fifth_term_l127_127040


namespace difference_representations_l127_127969

open Finset

def my_set : Finset ℕ := range 22 \ {0}

theorem difference_representations : (card {d ∈ (my_set.product my_set) | ∃ a b, a ≠ b ∧ d = abs (a - b)}).card = 20 :=
by {
  sorry
}

end difference_representations_l127_127969


namespace possible_values_of_a_l127_127721

def A (a : ℤ) : Set ℤ := {2, 4, a^3 - 2 * a^2 - a + 7}
def B (a : ℤ) : Set ℤ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem possible_values_of_a (a : ℤ) :
  A a ∩ B a = {2, 5} ↔ a = -1 ∨ a = 2 :=
by
  sorry

end possible_values_of_a_l127_127721


namespace number_of_pencil_cartons_l127_127153

theorem number_of_pencil_cartons
  (P E : ℕ) 
  (h1 : P + E = 100)
  (h2 : 6 * P + 3 * E = 360) : 
  P = 20 := 
by
  sorry

end number_of_pencil_cartons_l127_127153


namespace problem_I_problem_II_l127_127958

-- Definitions of sets A and B
def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
def B (a b : ℝ) : Set ℝ := { x : ℝ | x^2 - a * x + b < 0 }

-- Problem (I)
theorem problem_I (a b : ℝ) (h : A = B a b) : a = 2 ∧ b = -3 := 
sorry

-- Problem (II)
theorem problem_II (a : ℝ) (h₁ : ∀ x, (x ∈ A ∧ x ∈ B a 3) → x ∈ B a 3) : -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 := 
sorry


end problem_I_problem_II_l127_127958


namespace caleb_spent_more_on_ice_cream_l127_127372

theorem caleb_spent_more_on_ice_cream :
  let num_ic_cream := 10
  let cost_ic_cream := 4
  let num_frozen_yog := 4
  let cost_frozen_yog := 1
  (num_ic_cream * cost_ic_cream - num_frozen_yog * cost_frozen_yog) = 36 := 
by
  sorry

end caleb_spent_more_on_ice_cream_l127_127372


namespace engineer_walk_duration_l127_127370

variables (D : ℕ) (S : ℕ) (v : ℕ) (t : ℕ) (t1 : ℕ)

-- Stating the conditions
-- The time car normally takes to travel distance D
-- Speed (S) times the time (t) equals distance (D)
axiom speed_distance_relation : S * t = D

-- Engineer arrives at station at 7:00 AM and walks towards the car
-- They meet at t1 minutes past 7:00 AM, and the car covers part of the distance
-- Engineer reaches factory 20 minutes earlier than usual
-- Therefore, the car now meets the engineer covering less distance and time
axiom car_meets_engineer : S * t1 + v * t1 = D

-- The total travel time to the factory is reduced by 20 minutes
axiom travel_time_reduction : t - t1 = (t - 20 / 60)

-- Mathematically equivalent proof problem
theorem engineer_walk_duration : t1 = 50 := by
  sorry

end engineer_walk_duration_l127_127370


namespace area_union_square_circle_l127_127502

noncomputable def side_length_square : ℝ := 12
noncomputable def radius_circle : ℝ := 15
noncomputable def area_union : ℝ := 144 + 168.75 * Real.pi

theorem area_union_square_circle : 
  let area_square := side_length_square ^ 2
  let area_circle := Real.pi * radius_circle ^ 2
  let area_quarter_circle := area_circle / 4
  area_union = area_square + area_circle - area_quarter_circle :=
by
  -- The actual proof is omitted
  sorry

end area_union_square_circle_l127_127502


namespace num_red_light_runners_l127_127780

-- Definitions based on problem conditions
def surveys : ℕ := 800
def total_yes_answers : ℕ := 240

-- Assumptions
axiom coin_toss_equally_likely : ∀ {X : Type}, (X → Prop) → μ[X | _ = X] = 1 / 2

-- Proposition statement
theorem num_red_light_runners (students surveyed yes_answers : ℕ) 
(h1 : surveyed = 800) (h2 : yes_answers = 240) :
    ∃ red_light_runners, red_light_runners = 80 :=
by
  sorry

end num_red_light_runners_l127_127780


namespace minute_hand_length_l127_127898

noncomputable def length_minute_hand (A : ℝ) (θ : ℝ) : ℝ :=
  real.sqrt (2 * A / θ)

theorem minute_hand_length :
  length_minute_hand 15.274285714285716 (π / 3) ≈ 5.4 :=
by {
  sorry
}

end minute_hand_length_l127_127898


namespace fraction_by_foot_l127_127138

theorem fraction_by_foot (D distance_by_bus distance_by_car distance_by_foot : ℕ) (h1 : D = 24) 
  (h2 : distance_by_bus = D / 4) (h3 : distance_by_car = 6) 
  (h4 : distance_by_foot = D - (distance_by_bus + distance_by_car)) : 
  (distance_by_foot : ℚ) / D = 1 / 2 :=
by
  sorry

end fraction_by_foot_l127_127138


namespace distinct_four_digit_positive_integers_product_18_l127_127558

theorem distinct_four_digit_positive_integers_product_18 :
  Finset.card {n | ∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
                             (1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9) ∧
                             a * b * c * d = 18} = 24 :=
by
  sorry

end distinct_four_digit_positive_integers_product_18_l127_127558


namespace sufficient_condition_for_min_value_not_necessary_condition_for_min_value_l127_127264

noncomputable def f (x b : ℝ) : ℝ := x^2 + b*x

theorem sufficient_condition_for_min_value (b : ℝ) : b < 0 → ∀ x, min (f (f x b) b) = min (f x b) :=
sorry

theorem not_necessary_condition_for_min_value (b : ℝ) : (b < 0) ∧ (∀ x, min (f (f x b) b) = min (f x b)) → b ≤ 0 ∨ b ≥ 2 := 
sorry

end sufficient_condition_for_min_value_not_necessary_condition_for_min_value_l127_127264


namespace A_B_days_together_l127_127803

variable (W : ℝ) -- total work
variable (x : ℝ) -- days A and B worked together
variable (A_B_rate : ℝ) -- combined work rate of A and B
variable (A_rate : ℝ) -- work rate of A
variable (B_days : ℝ) -- days A worked alone after B left

-- Conditions:
axiom condition1 : A_B_rate = W / 40
axiom condition2 : A_rate = W / 80
axiom condition3 : B_days = 6
axiom condition4 : (x * A_B_rate + B_days * A_rate = W)

-- We want to prove that x = 37:
theorem A_B_days_together : x = 37 :=
by
  sorry

end A_B_days_together_l127_127803


namespace second_printer_cost_l127_127050

theorem second_printer_cost (p1_cost : ℕ) (num_units : ℕ) (total_spent : ℕ) (x : ℕ) 
  (h1 : p1_cost = 375) 
  (h2 : num_units = 7) 
  (h3 : total_spent = p1_cost * num_units) 
  (h4 : total_spent = x * num_units) : 
  x = 375 := 
sorry

end second_printer_cost_l127_127050


namespace var_power_l127_127117

theorem var_power {a b c x y z : ℝ} (h1 : x = a * y^4) (h2 : y = b * z^(1/3)) :
  ∃ n : ℝ, x = c * z^n ∧ n = 4/3 := by
  sorry

end var_power_l127_127117


namespace always_negative_l127_127550

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (x ^ 2 + 1) - x) - Real.sin x

theorem always_negative (a b : ℝ) (ha : a ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
                     (hb : b ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
                     (hab : a + b ≠ 0) : 
  (f a + f b) / (a + b) < 0 := 
sorry

end always_negative_l127_127550


namespace max_value_is_one_eighth_l127_127094

noncomputable def find_max_value (a b c : ℝ) : ℝ :=
  a^2 * b^2 * c^2 * (a + b + c) / ((a + b)^3 * (b + c)^3)

theorem max_value_is_one_eighth (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  find_max_value a b c ≤ 1 / 8 :=
by
  sorry

end max_value_is_one_eighth_l127_127094


namespace sum_of_remainders_mod_53_l127_127177

theorem sum_of_remainders_mod_53 (x y z : ℕ) (h1 : x % 53 = 31) (h2 : y % 53 = 17) (h3 : z % 53 = 8) : 
  (x + y + z) % 53 = 3 :=
by {
  sorry
}

end sum_of_remainders_mod_53_l127_127177


namespace math_proof_problem_l127_127074

open Nat

noncomputable def number_of_pairs := 
  let N := 20^19
  let num_divisors := (38 + 1) * (19 + 1)
  let total_pairs := num_divisors * num_divisors
  let ab_dividing_pairs := 780 * 210
  total_pairs - ab_dividing_pairs

theorem math_proof_problem : number_of_pairs = 444600 := 
  by exact sorry

end math_proof_problem_l127_127074


namespace greatest_value_a2_b2_c2_d2_l127_127989

theorem greatest_value_a2_b2_c2_d2 :
  ∃ (a b c d : ℝ), a + b = 12 ∧ ab + c + d = 54 ∧ ad + bc = 105 ∧ cd = 50 ∧ a^2 + b^2 + c^2 + d^2 = 124 := by
  sorry

end greatest_value_a2_b2_c2_d2_l127_127989


namespace locus_of_centroid_l127_127956

noncomputable def distances_from_plane (A B C : Point) (ε : Plane) : ℝ :=
  let a := distance A ε
  let b := distance B ε
  let c := distance C ε
  (a + b + c) / 6

theorem locus_of_centroid {A B C : Point} (ε : Plane) (a b c : ℝ) :
  (¬ Collinear {A, B, C}) →
  ∃ A' B' C' : Point, (A' ∈ ε ∧ B' ∈ ε ∧ C' ∈ ε) →
  let L := midpoint A A'
  let M := midpoint B B'
  let N := midpoint C C'
  let G := centroid L M N
  (distance G ε = distances_from_plane A B C ε) :=
by {
  sorry -- Proof steps are not required
}

end locus_of_centroid_l127_127956


namespace petya_board_problem_l127_127452

variable (A B Z : ℕ)

theorem petya_board_problem (h1 : A + B + Z = 10) (h2 : A * B = 15) : Z = 2 := sorry

end petya_board_problem_l127_127452


namespace new_length_maintains_area_l127_127054

noncomputable def new_length_for_doubled_width (A W : ℝ) : ℝ := A / (2 * W)

theorem new_length_maintains_area (A W : ℝ) (hA : A = 35.7) (hW : W = 3.8) :
  new_length_for_doubled_width A W = 4.69736842 :=
by
  rw [new_length_for_doubled_width, hA, hW]
  norm_num
  sorry

end new_length_maintains_area_l127_127054


namespace defective_probability_l127_127368

variable (total_products defective_products qualified_products : ℕ)
variable (first_draw_defective second_draw_defective : Prop)

-- Definitions of the problem
def total_prods := 10
def def_prods := 4
def qual_prods := 6
def p_A := def_prods / total_prods
def p_AB := (def_prods / total_prods) * ((def_prods - 1) / (total_prods - 1))
def p_B_given_A := p_AB / p_A

-- Theorem: The probability of drawing a defective product on the second draw given the first was defective is 1/3.
theorem defective_probability 
  (hp1 : total_products = total_prods)
  (hp2 : defective_products = def_prods)
  (hp3 : qualified_products = qual_prods)
  (pA_eq : p_A = 2 / 5)
  (pAB_eq : p_AB = 2 / 15) : 
  p_B_given_A = 1 / 3 := sorry

end defective_probability_l127_127368


namespace work_completion_days_l127_127118

theorem work_completion_days (A B C : ℝ) (h1 : A + B + C = 1/4) (h2 : B = 1/18) (h3 : C = 1/6) : A = 1/36 :=
by
  sorry

end work_completion_days_l127_127118


namespace half_difference_donation_l127_127305

def margoDonation : ℝ := 4300
def julieDonation : ℝ := 4700

theorem half_difference_donation : (julieDonation - margoDonation) / 2 = 200 := by
  sorry

end half_difference_donation_l127_127305


namespace philip_school_trip_days_l127_127629

-- Define the distances for the trips
def school_trip_one_way_miles : ℝ := 2.5
def market_trip_one_way_miles : ℝ := 2

-- Define the number of times he makes the trips in a day and in a week
def school_round_trips_per_day : ℕ := 2
def market_round_trips_per_week : ℕ := 1

-- Define the total mileage in a week
def weekly_mileage : ℕ := 44

-- Define the equation based on the given conditions
def weekly_school_trip_distance (d : ℕ) : ℝ :=
  (school_trip_one_way_miles * 2 * school_round_trips_per_day) * d

def weekly_market_trip_distance : ℝ :=
  (market_trip_one_way_miles * 2) * market_round_trips_per_week

-- Define the main theorem to be proved
theorem philip_school_trip_days :
  ∃ d : ℕ, weekly_school_trip_distance d + weekly_market_trip_distance = weekly_mileage ∧ d = 4 :=
by
  sorry

end philip_school_trip_days_l127_127629


namespace maximum_value_of_a_squared_b_l127_127100

theorem maximum_value_of_a_squared_b {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a * (a + b) = 27) : 
  a^2 * b ≤ 54 :=
sorry

end maximum_value_of_a_squared_b_l127_127100


namespace probability_of_event_l127_127647

theorem probability_of_event (favorable unfavorable : ℕ) (h : favorable = 3) (h2 : unfavorable = 5) :
  (favorable / (favorable + unfavorable) : ℚ) = 3 / 8 :=
by
  sorry

end probability_of_event_l127_127647


namespace second_friend_shells_l127_127143

theorem second_friend_shells (initial_shells : ℕ) (first_friend_shells : ℕ) (total_shells : ℕ) (second_friend_shells : ℕ) :
  initial_shells = 5 → first_friend_shells = 15 → total_shells = 37 → initial_shells + first_friend_shells + second_friend_shells = total_shells → second_friend_shells = 17 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end second_friend_shells_l127_127143


namespace smallest_four_digit_divisible_by_2_3_5_7_11_l127_127082

theorem smallest_four_digit_divisible_by_2_3_5_7_11 :
  ∃ n : ℕ, n = 2310 ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ (n % 11 = 0) ∧ 1000 ≤ n ∧ n < 10000 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_2_3_5_7_11_l127_127082


namespace f_odd_and_periodic_l127_127754

open Function

-- Define the function f : ℝ → ℝ satisfying the given conditions
variables (f : ℝ → ℝ)

-- Conditions
axiom f_condition1 : ∀ x : ℝ, f (10 + x) = f (10 - x)
axiom f_condition2 : ∀ x : ℝ, f (20 - x) = -f (20 + x)

-- Theorem statement
theorem f_odd_and_periodic : Odd f ∧ Periodic f 40 :=
by
  -- Proof will be filled here
  sorry

end f_odd_and_periodic_l127_127754


namespace squared_sum_l127_127131

theorem squared_sum (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 :=
by
  sorry

end squared_sum_l127_127131


namespace sqrt_expression_value_l127_127637

theorem sqrt_expression_value :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 * Real.sqrt 3 :=
by
  sorry

end sqrt_expression_value_l127_127637


namespace correct_polynomials_are_l127_127518

noncomputable def polynomial_solution (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, p.eval (x^2) = (p.eval x) * (p.eval (x - 1))

theorem correct_polynomials_are (p : Polynomial ℝ) :
  polynomial_solution p ↔ ∃ n : ℕ, p = (Polynomial.C (1 : ℝ) * Polynomial.X ^ 2 + Polynomial.C (1 : ℝ) * Polynomial.X + Polynomial.C (1 : ℝ)) ^ n :=
by
  sorry

end correct_polynomials_are_l127_127518


namespace total_messages_l127_127679

theorem total_messages (l1 l2 l3 a1 a2 a3 : ℕ)
  (h1 : l1 = 120)
  (h2 : a1 = l1 - 20)
  (h3 : l2 = l1 / 3)
  (h4 : a2 = 2 * a1)
  (h5 : l3 = l1)
  (h6 : a3 = a1) :
  l1 + l2 + l3 + a1 + a2 + a3 = 680 :=
by
  -- Proof steps would go here. Adding 'sorry' to skip proof.
  sorry

end total_messages_l127_127679


namespace binom_20_10_eq_184756_l127_127546

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end binom_20_10_eq_184756_l127_127546


namespace thief_distance_l127_127504

variable (d : ℝ := 250)   -- initial distance in meters
variable (v_thief : ℝ := 12 * 1000 / 3600)  -- thief's speed in m/s (converted from km/hr)
variable (v_policeman : ℝ := 15 * 1000 / 3600)  -- policeman's speed in m/s (converted from km/hr)

noncomputable def distance_thief_runs : ℝ :=
  v_thief * (d / (v_policeman - v_thief))

theorem thief_distance :
  distance_thief_runs d v_thief v_policeman = 990.47 := sorry

end thief_distance_l127_127504


namespace sum_of_elements_in_T_l127_127993

def T : finset ℕ := (finset.range (2 ^ 5)).filter (λ x, x ≥ 16)

theorem sum_of_elements_in_T :
  T.sum id = 0b111110100 :=
sorry

end sum_of_elements_in_T_l127_127993


namespace sally_students_are_30_l127_127314

-- Define the conditions given in the problem
def school_money : ℕ := 320
def book_cost : ℕ := 12
def sally_money : ℕ := 40
def total_students : ℕ := 30

-- Define the total amount Sally can spend on books
def total_amount_available : ℕ := school_money + sally_money

-- The total cost of books for S students
def total_cost (S : ℕ) : ℕ := book_cost * S

-- The main theorem stating that S students will cost the same as the amount Sally can spend
theorem sally_students_are_30 : total_cost 30 = total_amount_available :=
by
  sorry

end sally_students_are_30_l127_127314


namespace part_I_part_II_l127_127263

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem part_I (a : ℝ) : ∀ x : ℝ, (0 < (2^x * Real.log 2) / (2^x + 1)^2) :=
by
  sorry

theorem part_II (h : ∀ x : ℝ, f a x = -f a (-x)) : 
  a = (1:ℝ)/2 ∧ ∀ x : ℝ, -((1:ℝ)/2) < f (1/2) x ∧ f (1/2) x < (1:ℝ)/2 :=
by
  sorry

end part_I_part_II_l127_127263


namespace triangle_ineq_l127_127998

theorem triangle_ineq (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  (a / (b + c) + b / (a + c) + c / (a + b)) < 5/2 := 
by
  sorry

end triangle_ineq_l127_127998


namespace find_a_l127_127983

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 152) : a = 50 := 
by 
  sorry

end find_a_l127_127983


namespace problem_2011_Mentougou_l127_127046

theorem problem_2011_Mentougou 
  (f : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (H2 : ∀ x : ℝ, 0 < x → 0 < f x) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :=
sorry

end problem_2011_Mentougou_l127_127046


namespace ethan_presents_l127_127698

theorem ethan_presents (ethan alissa : ℕ) 
  (h1 : alissa = ethan + 22) 
  (h2 : alissa = 53) : 
  ethan = 31 := 
by
  sorry

end ethan_presents_l127_127698


namespace sunny_lead_l127_127283

-- Define the given conditions as hypotheses
variables (h d : ℝ) (s w : ℝ)
    (H1 : ∀ t, t = 2 * h → (s * t) = 2 * h ∧ (w * t) = 2 * h - 2 * d)
    (H2 : ∀ t, (s * t) = 2 * h + 2 * d → (w * t) = 2 * h)

-- State the theorem we want to prove
theorem sunny_lead (h d : ℝ) (s w : ℝ) 
    (H1 : ∀ t, t = 2 * h → (s * t) = 2 * h ∧ (w * t) = 2 * h - 2 * d)
    (H2 : ∀ t, (s * t) = 2 * h + 2 * d → (w * t) = 2 * h) :
    ∃ distance_ahead_Sunny : ℝ, distance_ahead_Sunny = (2 * d^2) / h :=
sorry

end sunny_lead_l127_127283


namespace count_four_digit_integers_with_product_18_l127_127562

def valid_digits (n : ℕ) : Prop := 
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def digit_product_18 (a b c d : ℕ) : Prop := 
  a * b * c * d = 18

def four_digit_integer (a b c d : ℕ) : Prop := 
  valid_digits a ∧ valid_digits b ∧ valid_digits c ∧ valid_digits d

theorem count_four_digit_integers_with_product_18 : 
  (∑ a b c d in {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    ite (four_digit_integer a b c d ∧ digit_product_18 a b c d) 1 0) = 48 := 
sorry

end count_four_digit_integers_with_product_18_l127_127562


namespace find_LM_l127_127293

variables (K L M N P Q : Type)
variables (KL MN LM KN MQ MP KP KM : ℝ) 

-- Conditions
def trapezoid (K L M N : Type) : Prop := 
  KM = 1 ∧ 
  KP = 1 ∧
  MQ = 1 ∧
  KN = MQ ∧
  LM = MP

-- To Prove
theorem find_LM (h : trapezoid K L M N) : LM = sqrt 2 :=
by sorry

end find_LM_l127_127293


namespace smallest_consecutive_sum_l127_127628

theorem smallest_consecutive_sum (x : ℤ) (h : x + (x + 1) + (x + 2) = 90) : x = 29 :=
by 
  sorry

end smallest_consecutive_sum_l127_127628


namespace area_of_circumcircle_of_isosceles_triangle_l127_127191

theorem area_of_circumcircle_of_isosceles_triangle :
  ∀ (r : ℝ) (π : ℝ), (∀ (a b c : ℝ)
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 4),
  r = sqrt (a * b * (a + b + c) * (a + b - c)) / c →
  ∀ (area : ℝ), area = π * r ^ 2 →
  area = 13125 / 1764 * π) :=
  λ r π a b c h1 h2 h3 h_radius area h_area, sorry

end area_of_circumcircle_of_isosceles_triangle_l127_127191


namespace remainder_is_four_l127_127950

def least_number : Nat := 174

theorem remainder_is_four (n : Nat) (m₁ m₂ : Nat) (h₁ : n = least_number / m₁ * m₁ + 4) 
(h₂ : n = least_number / m₂ * m₂ + 4) (h₃ : m₁ = 34) (h₄ : m₂ = 5) : 
  n % m₁ = 4 ∧ n % m₂ = 4 := 
by
  sorry

end remainder_is_four_l127_127950


namespace trapezium_area_l127_127590

variables {A B C D O : Type}
variables (P Q : ℕ)

-- Conditions
def trapezium (ABCD : Type) : Prop := true
def parallel_lines (AB DC : Type) : Prop := true
def intersection (AC BD O : Type) : Prop := true
def area_AOB (P : ℕ) : Prop := P = 16
def area_COD : ℕ := 25

theorem trapezium_area (ABCD AC BD AB DC O : Type) (P Q : ℕ)
  (h1 : trapezium ABCD)
  (h2 : parallel_lines AB DC)
  (h3 : intersection AC BD O)
  (h4 : area_AOB P) 
  (h5 : area_COD = 25) :
  Q = 81 :=
sorry

end trapezium_area_l127_127590


namespace problem1_problem2_l127_127972

theorem problem1 (x : ℕ) : 
  2 / 8^x * 16^x = 2^5 → x = 4 := 
by
  sorry

theorem problem2 (x : ℕ) : 
  2^(x+2) + 2^(x+1) = 24 → x = 2 := 
by
  sorry

end problem1_problem2_l127_127972


namespace luke_total_points_l127_127446

theorem luke_total_points (rounds : ℕ) (points_per_round : ℕ) (total_points : ℕ) 
  (h1 : rounds = 177) (h2 : points_per_round = 46) : 
  total_points = 8142 := by
  have h : total_points = rounds * points_per_round := by sorry
  rw [h1, h2] at h
  exact h

end luke_total_points_l127_127446


namespace max_area_of_triangle_ABC_l127_127529

noncomputable def parabola (x : ℝ) : ℝ := (6 * x)^(1/2)

def point_on_parabola (x1 x2 y1 y2 : ℝ) : Prop :=
  (y1^2 = 6 * x1) ∧ (y2^2 = 6 * x2)

def parabola_conditions (x1 x2 : ℝ) : Prop :=
  (x1 ≠ x2) ∧ (x1 + x2 = 4)

def perp_bisector_point (x1 x2 y1 y2 : ℝ) : Prop :=
  (- (y1^2 - y2^2)/(4 * (x2 - x1)) + 2)

def max_area (x1 x2 y1 y2 : ℝ) : ℝ :=
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  let m_c := (0 - ym) / (5 - 2)
  let area := 1 / 2 * abs (t * ym + 3 * m_c) in
  max area

theorem max_area_of_triangle_ABC :
  ∀ x1 x2 y1 y2 : ℝ,
    point_on_parabola x1 x2 y1 y2 →
    parabola_conditions x1 x2 →
    max_area x1 x2 y1 y2 = 7*sqrt(7)/6 :=
by
  sorry

end max_area_of_triangle_ABC_l127_127529


namespace shaded_area_T_shape_l127_127826

theorem shaded_area_T_shape (a b c d e: ℕ) (square_side_length rect_length rect_width: ℕ)
  (h_side_lengths: ∀ x, x = 2 ∨ x = 4) (h_square: square_side_length = 6) 
  (h_rect_dim: rect_length = 4 ∧ rect_width = 2)
  (h_areas: [a, b, c, d, e] = [4, 4, 4, 8, 4]) :
  a + b + d + e = 20 :=
by
  sorry

end shaded_area_T_shape_l127_127826


namespace trapezoid_LM_sqrt2_l127_127292

theorem trapezoid_LM_sqrt2 (K L M N P Q : Type*)
  (KM : ℝ)
  (KN MQ LM MP : ℝ)
  (h_KM : KM = 1)
  (h_KN_MQ : KN = MQ)
  (h_LM_MP : LM = MP) 
  (h_KP_1 : KN = 1) 
  (h_MQ_1 : MQ = 1) :
  LM = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l127_127292


namespace points_opposite_sides_l127_127478

theorem points_opposite_sides (m : ℝ) : (-2 < m ∧ m < -1) ↔ ((2 - 3 * 1 - m) * (1 - 3 * 1 - m) < 0) := by
  sorry

end points_opposite_sides_l127_127478


namespace probability_of_different_colors_l127_127170

noncomputable def total_chips := 6 + 5 + 4

noncomputable def prob_diff_color : ℚ :=
  let pr_blue := 6 / total_chips
  let pr_red := 5 / total_chips
  let pr_yellow := 4 / total_chips

  let pr_not_blue := (5 + 4) / total_chips
  let pr_not_red := (6 + 4) / total_chips
  let pr_not_yellow := (6 + 5) / total_chips

  pr_blue * pr_not_blue + pr_red * pr_not_red + pr_yellow * pr_not_yellow

theorem probability_of_different_colors :
  prob_diff_color = 148 / 225 :=
sorry

end probability_of_different_colors_l127_127170


namespace no_polynomial_exists_l127_127110

open Polynomial

theorem no_polynomial_exists (a b c : ℤ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  ¬ ∃ (P : ℤ[X]), P.eval a = b ∧ P.eval b = c ∧ P.eval c = a :=
sorry

end no_polynomial_exists_l127_127110


namespace determine_y_l127_127385

theorem determine_y (y : ℝ) (h1 : 0 < y) (h2 : y * (⌊y⌋ : ℝ) = 90) : y = 10 :=
sorry

end determine_y_l127_127385


namespace proof_P_otimes_Q_l127_127878

-- Define the sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def Q : Set ℝ := { x | 1 < x }

-- Define the operation ⊗ between sets
def otimes (P Q : Set ℝ) : Set ℝ := { x | x ∈ P ∪ Q ∧ x ∉ P ∩ Q }

-- Prove that P ⊗ Q = [0,1] ∪ (2, +∞)
theorem proof_P_otimes_Q :
  otimes P Q = {x | (0 ≤ x ∧ x ≤ 1) ∨ (2 < x)} :=
by
 sorry

end proof_P_otimes_Q_l127_127878


namespace contrapositive_l127_127409

theorem contrapositive (p q : Prop) (h : p → q) : ¬q → ¬p :=
by
  sorry

end contrapositive_l127_127409


namespace chimps_seen_l127_127684

-- Given conditions
def lions := 8
def lion_legs := 4
def lizards := 5
def lizard_legs := 4
def tarantulas := 125
def tarantula_legs := 8
def goal_legs := 1100

-- Required to be proved
def chimp_legs := 4

theorem chimps_seen : (goal_legs - ((lions * lion_legs) + (lizards * lizard_legs) + (tarantulas * tarantula_legs))) / chimp_legs = 25 :=
by
  -- placeholder for the proof
  sorry

end chimps_seen_l127_127684


namespace exists_multiple_representations_l127_127603

def V (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

def indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V n ∧ ¬∃ (p q : ℕ), p ∈ V n ∧ q ∈ V n ∧ p * q = m

theorem exists_multiple_representations (n : ℕ) (h : 2 < n) :
  ∃ r ∈ V n, ∃ s t u v : ℕ, 
    indecomposable n s ∧ indecomposable n t ∧ indecomposable n u ∧ indecomposable n v ∧ 
    r = s * t ∧ r = u * v ∧ (s ≠ u ∨ t ≠ v) :=
sorry

end exists_multiple_representations_l127_127603


namespace not_lucky_1994_l127_127497

def is_valid_month (m : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 12

def is_valid_day (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 31

def is_lucky_year (y : ℕ) : Prop :=
  ∃ (m d : ℕ), is_valid_month m ∧ is_valid_day d ∧ m * d = y

theorem not_lucky_1994 : ¬ is_lucky_year 94 := 
by
  sorry

end not_lucky_1994_l127_127497


namespace relationship_a_plus_b_greater_c_relationship_a_squared_plus_b_squared_equals_c_squared_relationship_a_n_plus_b_n_less_than_c_n_l127_127003

-- Let a, b, and c be the sides of a right triangle with c as the hypotenuse.
variables (a b c : ℝ) (n : ℕ)

-- Assume the triangle is a right triangle
-- and assume n is a positive integer.
axiom right_triangle : a^2 + b^2 = c^2
axiom positive_integer : n > 0 

-- The relationships we need to prove:
theorem relationship_a_plus_b_greater_c : n = 1 → a + b > c := sorry
theorem relationship_a_squared_plus_b_squared_equals_c_squared : n = 2 → a^2 + b^2 = c^2 := sorry
theorem relationship_a_n_plus_b_n_less_than_c_n : n ≥ 3 → a^n + b^n < c^n := sorry

end relationship_a_plus_b_greater_c_relationship_a_squared_plus_b_squared_equals_c_squared_relationship_a_n_plus_b_n_less_than_c_n_l127_127003


namespace number_of_students_l127_127186

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N)
  (h2 : 95 * (N - 5) = T - 100) : N = 25 :=
by
  sorry

end number_of_students_l127_127186


namespace john_trip_time_l127_127437

theorem john_trip_time (x : ℝ) (h : x + 2 * x + 2 * x = 10) : x = 2 :=
by
  sorry

end john_trip_time_l127_127437


namespace infinite_set_P_l127_127198

-- Define the condition as given in the problem
def has_property_P (P : Set ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → (∀ p : ℕ, p.Prime → p ∣ k^3 + 6 → p ∈ P)

-- State the proof problem
theorem infinite_set_P (P : Set ℕ) (h : has_property_P P) : ∃ p : ℕ, p ∉ P → false :=
by
  -- The statement asserts that the set P described by has_property_P is infinite.
  sorry

end infinite_set_P_l127_127198


namespace scientific_calculators_ordered_l127_127946

variables (x y : ℕ)

theorem scientific_calculators_ordered :
  (10 * x + 57 * y = 1625) ∧ (x + y = 45) → x = 20 :=
by
  -- proof goes here
  sorry

end scientific_calculators_ordered_l127_127946


namespace monic_polynomial_roots_l127_127303

theorem monic_polynomial_roots (r1 r2 r3 : ℝ) (h : ∀ x : ℝ, x^3 - 4*x^2 + 5 = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) :
  ∀ x : ℝ, x^3 - 12*x^2 + 135 = 0 ↔ x = 3*r1 ∨ x = 3*r2 ∨ x = 3*r3 :=
by
  sorry

end monic_polynomial_roots_l127_127303


namespace next_volunteer_day_l127_127932

-- Definitions based on conditions.
def Alison_schedule := 5
def Ben_schedule := 3
def Carla_schedule := 9
def Dave_schedule := 8

-- Main theorem
theorem next_volunteer_day : Nat.lcm Alison_schedule (Nat.lcm Ben_schedule (Nat.lcm Carla_schedule Dave_schedule)) = 360 := by
  sorry

end next_volunteer_day_l127_127932


namespace tagged_fish_in_second_catch_l127_127977

theorem tagged_fish_in_second_catch (N : ℕ) (initially_tagged second_catch : ℕ)
  (h1 : N = 1250)
  (h2 : initially_tagged = 50)
  (h3 : second_catch = 50) :
  initially_tagged / N * second_catch = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l127_127977


namespace quadratic_roots_l127_127577

noncomputable def roots_quadratic : Prop :=
  ∀ (a b : ℝ), (a + b = 7) ∧ (a * b = 7) → (a^2 + b^2 = 35)

theorem quadratic_roots (a b : ℝ) (h : a + b = 7 ∧ a * b = 7) : a^2 + b^2 = 35 :=
by
  sorry

end quadratic_roots_l127_127577


namespace probability_sum_15_l127_127734

/-- If three standard 6-faced dice are rolled, the probability that the sum of the face-up integers is 15 is 5/72. -/
theorem probability_sum_15 : (1 / 6 : ℚ) ^ 3 * 3 + (1 / 6 : ℚ) ^ 3 * 6 = 5 / 72 := by 
  sorry

end probability_sum_15_l127_127734


namespace div_100_by_a8_3a4_minus_4_l127_127763

theorem div_100_by_a8_3a4_minus_4 (a : ℕ) (h : ¬ (5 ∣ a)) : 100 ∣ (a^8 + 3 * a^4 - 4) :=
sorry

end div_100_by_a8_3a4_minus_4_l127_127763


namespace trigonometric_comparison_l127_127144

noncomputable def a : ℝ := Real.sin (3 * Real.pi / 5)
noncomputable def b : ℝ := Real.cos (2 * Real.pi / 5)
noncomputable def c : ℝ := Real.tan (2 * Real.pi / 5)

theorem trigonometric_comparison :
  b < a ∧ a < c :=
by {
  -- Use necessary steps to demonstrate b < a and a < c
  sorry
}

end trigonometric_comparison_l127_127144


namespace sample_size_student_congress_l127_127049

-- Definitions based on the conditions provided in the problem
def num_classes := 40
def students_per_class := 3

-- Theorem statement for the mathematically equivalent proof problem
theorem sample_size_student_congress : 
  (num_classes * students_per_class) = 120 := 
by 
  sorry

end sample_size_student_congress_l127_127049


namespace sector_angle_l127_127103

theorem sector_angle (r l : ℝ) (h1 : l + 2 * r = 6) (h2 : 1/2 * l * r = 2) : 
  l / r = 1 ∨ l / r = 4 := 
sorry

end sector_angle_l127_127103


namespace sin_cos_sum_l127_127246

theorem sin_cos_sum (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π)
  (h : Real.tan (θ + Real.pi / 4) = 1 / 7) : Real.sin θ + Real.cos θ = -1 / 5 := 
by
  sorry

end sin_cos_sum_l127_127246


namespace rest_stop_location_l127_127491

theorem rest_stop_location (km_A km_B : ℕ) (fraction : ℚ) (difference := km_B - km_A) 
  (rest_stop_distance := fraction * difference) : 
  km_A = 30 → km_B = 210 → fraction = 4 / 5 → rest_stop_distance + km_A = 174 :=
by 
  intros h1 h2 h3
  sorry

end rest_stop_location_l127_127491


namespace percentage_increase_l127_127348

theorem percentage_increase (initial final : ℝ) (h_initial : initial = 200) (h_final : final = 250) :
  ((final - initial) / initial) * 100 = 25 := 
sorry

end percentage_increase_l127_127348


namespace sequence_thirtieth_term_l127_127298

theorem sequence_thirtieth_term :
  ∀ (a : ℕ) (d : ℕ), a = 2 ∧ d = 2 → a + 29 * d = 60 :=
by
  intro a d
  rintro ⟨ha, hd⟩
  simp [ha, hd]
  sorry

end sequence_thirtieth_term_l127_127298


namespace fill_tank_time_l127_127453

theorem fill_tank_time :
  ∀ (capacity rateA rateB rateC timeA timeB timeC : ℕ),
  capacity = 1000 →
  rateA = 200 →
  rateB = 50 →
  rateC = 25 →
  timeA = 1 →
  timeB = 2 →
  timeC = 2 →
  let net_fill := rateA * timeA + rateB * timeB - rateC * timeC in
  let total_cycles := capacity / net_fill in
  let cycle_time := timeA + timeB + timeC in
  let total_time := total_cycles * cycle_time in
  total_time = 20 := sorry

end fill_tank_time_l127_127453


namespace difference_between_roots_l127_127073

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := -7
noncomputable def c : ℝ := 11

noncomputable def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b ^ 2 - 4 * a * c
  ((-b + Real.sqrt discriminant) / (2 * a), (-b - Real.sqrt discriminant) / (2 * a))

-- Extract the roots from the equation
noncomputable def r1_r2 := quadratic_roots a b c

noncomputable def r1 : ℝ := r1_r2.1
noncomputable def r2 : ℝ := r1_r2.2

-- Theorem statement: the difference between the roots is sqrt(5)
theorem difference_between_roots :
  |r1 - r2| = Real.sqrt 5 :=
  sorry

end difference_between_roots_l127_127073


namespace geometric_sequence_mean_l127_127432

theorem geometric_sequence_mean (a : ℕ → ℝ) (q : ℝ) (h_q : q = -2) 
  (h_condition : a 3 * a 7 = 4 * a 4) : 
  ((a 8 + a 11) / 2 = -56) 
:= sorry

end geometric_sequence_mean_l127_127432


namespace probability_sum_of_10_l127_127864

theorem probability_sum_of_10 (total_outcomes : ℕ) 
  (h1 : total_outcomes = 6^4) : 
  (46 / total_outcomes) = 23 / 648 := by
  sorry

end probability_sum_of_10_l127_127864


namespace total_amount_paid_l127_127919

-- Definitions from the conditions
def quantity_grapes : ℕ := 8
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 60

-- Main statement to prove
theorem total_amount_paid :
  (quantity_grapes * rate_grapes) + (quantity_mangoes * rate_mangoes) = 1100 :=
by
  sorry

end total_amount_paid_l127_127919


namespace running_problem_l127_127451

variables (x y : ℝ)

theorem running_problem :
  (5 * x = 5 * y + 10) ∧ (4 * x = 4 * y + 2 * y) :=
by
  sorry

end running_problem_l127_127451


namespace find_range_of_r_l127_127589

noncomputable def range_of_r : Set ℝ :=
  {r : ℝ | 3 * Real.sqrt 5 - 3 * Real.sqrt 2 ≤ r ∧ r ≤ 3 * Real.sqrt 5 + 3 * Real.sqrt 2}

theorem find_range_of_r 
  (O : ℝ × ℝ) (A : ℝ × ℝ) (r : ℝ) (h : r > 0)
  (hA : A = (0, 3))
  (C : Set (ℝ × ℝ)) (hC : C = {M : ℝ × ℝ | (M.1 - 3)^2 + (M.2 - 3)^2 = r^2})
  (M : ℝ × ℝ) (hM : M ∈ C)
  (h_cond : (M.1 - 0)^2 + (M.2 - 3)^2 = 2 * ((M.1 - 0)^2 + (M.2 - 0)^2)) :
  r ∈ range_of_r :=
sorry

end find_range_of_r_l127_127589


namespace alexis_pants_l127_127594

theorem alexis_pants (P D : ℕ) (A_p : ℕ)
  (h1 : P + D = 13)
  (h2 : 3 * D = 18)
  (h3 : A_p = 3 * P) : A_p = 21 :=
  sorry

end alexis_pants_l127_127594


namespace solve_for_x_l127_127513

theorem solve_for_x : ∀ (x : ℝ), (-3 * x - 8 = 5 * x + 4) → (x = -3 / 2) := by
  intro x
  intro h
  sorry

end solve_for_x_l127_127513


namespace quadratic_function_count_l127_127096

open Finset

def is_even (n : ℕ) := ∃ k, n = 2 * k

theorem quadratic_function_count :
  let S := (range 10).filter (λ x, x > 0) in
  (card (S.powerset.filter (λ t, t.card = 3 ∧ is_even (t.sum)))) = 264 :=
by
  sorry

end quadratic_function_count_l127_127096


namespace change_digit_correct_sum_l127_127019

theorem change_digit_correct_sum :
  ∃ d e, 
  d = 2 ∧ e = 8 ∧ 
  653479 + 938521 ≠ 1616200 ∧
  (658479 + 938581 = 1616200) ∧ 
  d + e = 10 := 
by {
  -- our proof goes here
  sorry
}

end change_digit_correct_sum_l127_127019


namespace weight_lift_equality_l127_127771

-- Definitions based on conditions
def total_weight_25_pounds_lifted_times := 750
def total_weight_20_pounds_lifted_per_time (n : ℝ) := 60 * n

-- Statement of the proof problem
theorem weight_lift_equality : ∃ n, total_weight_20_pounds_lifted_per_time n = total_weight_25_pounds_lifted_times :=
  sorry

end weight_lift_equality_l127_127771


namespace paul_peaches_l127_127935

theorem paul_peaches (P : ℕ) (h1 : 26 - P = 22) : P = 4 :=
by {
  sorry
}

end paul_peaches_l127_127935


namespace sum_first_10_terms_abs_a_n_l127_127399

noncomputable def a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else 3 * n - 7

def abs_a_n (n : ℕ) : ℤ :=
  if n = 1 ∨ n = 2 then -3 * n + 7 else 3 * n - 7

def sum_abs_a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else List.sum (List.map abs_a_n (List.range n))

theorem sum_first_10_terms_abs_a_n : sum_abs_a_n 10 = 105 := 
  sorry

end sum_first_10_terms_abs_a_n_l127_127399


namespace distinct_four_digit_numbers_product_18_l127_127570

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end distinct_four_digit_numbers_product_18_l127_127570


namespace final_tree_count_l127_127778

def current_trees : ℕ := 7
def monday_trees : ℕ := 3
def tuesday_trees : ℕ := 2
def wednesday_trees : ℕ := 5
def thursday_trees : ℕ := 1
def friday_trees : ℕ := 6
def saturday_trees : ℕ := 4
def sunday_trees : ℕ := 3

def total_trees_planted : ℕ := monday_trees + tuesday_trees + wednesday_trees + thursday_trees + friday_trees + saturday_trees + sunday_trees

theorem final_tree_count :
  current_trees + total_trees_planted = 31 :=
by
  sorry

end final_tree_count_l127_127778


namespace find_smallest_z_l127_127418

theorem find_smallest_z (x y z : ℤ) (h1 : 7 < x) (h2 : x < 9) (h3 : x < y) (h4 : y < z) 
  (h5 : y - x = 7) : z = 16 :=
by
  sorry

end find_smallest_z_l127_127418


namespace blueBirdChessTeam72_l127_127462

def blueBirdChessTeamArrangements : Nat :=
  let boys_girls_ends := 3 * 3 + 3 * 3
  let alternate_arrangements := 2 * 2
  boys_girls_ends * alternate_arrangements

theorem blueBirdChessTeam72 : blueBirdChessTeamArrangements = 72 := by
  unfold blueBirdChessTeamArrangements
  sorry

end blueBirdChessTeam72_l127_127462


namespace factorize_expression_l127_127226

variable {R : Type} [CommRing R] (a x y : R)

theorem factorize_expression :
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l127_127226


namespace decimal_150th_digit_of_5_over_37_l127_127789

theorem decimal_150th_digit_of_5_over_37 :
  let r := (5 : ℚ) / 37 in r.to_decimal 150 = 3 := by
  sorry

end decimal_150th_digit_of_5_over_37_l127_127789


namespace negation_statement_l127_127477

open Set

variable {S : Set ℝ}

theorem negation_statement (h : ∀ x ∈ S, 3 * x - 5 > 0) : ∃ x ∈ S, 3 * x - 5 ≤ 0 :=
sorry

end negation_statement_l127_127477


namespace probability_each_player_has_five_coins_l127_127060

noncomputable def probability_each_has_5_coins : ℝ :=
  1 / 207360000

theorem probability_each_player_has_five_coins :
  ∀ (players : Fin 4 → ℕ) (rounds : ℕ) (urn : List String) (draws : Fin 5 → (Fin 4 × String)),
  (∀ i, players i = 5) →
  rounds = 5 →
  urn = ["green", "red", "blue", "white", "white"] →
  (∀ r, 
    let ⟨p1, b1⟩ := draws r,
        ⟨p2, b2⟩ := draws (r + 1 mod 5)
    in 
    (if b1 = "green" then players p1 := players p1 - 1 else players p1) =
    (if b2 = "red" then players p2 := players p2 + 1 else players p2) ∧
    (if b1 = "blue" then players p1 := players p1 - 2 else players p1) =
    (if b2 = "green" ∧ b2 = "red" then players p2 := players p2 + 1 else players p2)
  ) →
  real.ext_iff.mp (pmf.mass_of players = pmf.mass_of draws) = probability_each_has_5_coins :=
sorry

end probability_each_player_has_five_coins_l127_127060


namespace min_value_expr_l127_127440

theorem min_value_expr (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + ((y / x) - 1)^2 + ((z / y) - 1)^2 + ((5 / z) - 1)^2 = 9 :=
sorry

end min_value_expr_l127_127440
