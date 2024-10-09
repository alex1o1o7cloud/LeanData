import Mathlib

namespace relationship_between_A_B_C_l1816_181674

-- Definitions based on the problem conditions
def A : Set ℝ := {θ | ∃ k : ℤ, 2 * k * Real.pi < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2}
def B : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def C : Set ℝ := {θ | θ < Real.pi / 2}

-- Proof statement: Prove the specified relationship
theorem relationship_between_A_B_C : B ∪ C = C := by
  sorry

end relationship_between_A_B_C_l1816_181674


namespace increasing_interval_of_f_on_0_pi_l1816_181687

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 4)

theorem increasing_interval_of_f_on_0_pi {ω : ℝ} (hω : ω > 0)
  (h_symmetry : ∀ x, f ω x = g x) :
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi ∧ ∀ x1 x2, (0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ Real.pi) → f ω x1 < f ω x2} = 
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi / 8} :=
sorry

end increasing_interval_of_f_on_0_pi_l1816_181687


namespace Roe_saved_15_per_month_aug_nov_l1816_181660

-- Step 1: Define the given conditions
def savings_per_month_jan_jul : ℕ := 10
def months_jan_jul : ℕ := 7
def savings_dec : ℕ := 20
def total_savings_needed : ℕ := 150
def months_aug_nov : ℕ := 4

-- Step 2: Define the intermediary calculations based on the conditions
def total_saved_jan_jul := savings_per_month_jan_jul * months_jan_jul
def total_savings_aug_nov := total_savings_needed - total_saved_jan_jul - savings_dec

-- Step 3: Define what we need to prove
def savings_per_month_aug_nov : ℕ := total_savings_aug_nov / months_aug_nov

-- Step 4: State the proof goal
theorem Roe_saved_15_per_month_aug_nov :
  savings_per_month_aug_nov = 15 :=
by
  sorry

end Roe_saved_15_per_month_aug_nov_l1816_181660


namespace lab_tech_ratio_l1816_181686

theorem lab_tech_ratio (U T C : ℕ) (hU : U = 12) (hC : C = 6 * U) (hT : T = (C + U) / 14) :
  (T : ℚ) / U = 1 / 2 :=
by
  sorry

end lab_tech_ratio_l1816_181686


namespace animal_sighting_ratio_l1816_181684

theorem animal_sighting_ratio
  (jan_sightings : ℕ)
  (feb_sightings : ℕ)
  (march_sightings : ℕ)
  (total_sightings : ℕ)
  (h1 : jan_sightings = 26)
  (h2 : feb_sightings = 3 * jan_sightings)
  (h3 : total_sightings = jan_sightings + feb_sightings + march_sightings)
  (h4 : total_sightings = 143) :
  (march_sightings : ℚ) / (feb_sightings : ℚ) = 1 / 2 :=
by
  sorry

end animal_sighting_ratio_l1816_181684


namespace tangent_ellipse_hyperbola_l1816_181607

theorem tangent_ellipse_hyperbola (n : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 ∧ x^2 - n * (y - 1)^2 = 1) → n = 2 :=
by
  intro h
  sorry

end tangent_ellipse_hyperbola_l1816_181607


namespace circle_line_no_intersection_l1816_181659

theorem circle_line_no_intersection (b : ℝ) :
  (∀ x y : ℝ, ¬ (x^2 + y^2 = 2 ∧ y = x + b)) ↔ (b > 2 ∨ b < -2) :=
by sorry

end circle_line_no_intersection_l1816_181659


namespace solution_of_abs_eq_l1816_181698

theorem solution_of_abs_eq (x : ℝ) : |x - 5| = 3 * x + 6 → x = -1 / 4 :=
by
  sorry

end solution_of_abs_eq_l1816_181698


namespace probability_of_9_heads_in_12_flips_l1816_181601

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l1816_181601


namespace maximize_volume_l1816_181639

-- Define the problem-specific constants
def bar_length : ℝ := 0.18
def length_to_width_ratio : ℝ := 2

-- Function to define volume of the rectangle frame
def volume (length width height : ℝ) : ℝ := length * width * height

theorem maximize_volume :
  ∃ (length width height : ℝ), 
  (length / width = length_to_width_ratio) ∧ 
  (2 * (length + width) = bar_length) ∧ 
  ((length = 2) ∧ (height = 1.5)) :=
sorry

end maximize_volume_l1816_181639


namespace laptop_selection_l1816_181663

open Nat

theorem laptop_selection :
  ∃ (n : ℕ), n = (choose 4 2) * (choose 5 1) + (choose 4 1) * (choose 5 2) := 
sorry

end laptop_selection_l1816_181663


namespace cost_of_socks_l1816_181648

theorem cost_of_socks (S : ℝ) (players : ℕ) (jersey : ℝ) (shorts : ℝ) 
                      (total_cost : ℝ) 
                      (h1 : players = 16) 
                      (h2 : jersey = 25) 
                      (h3 : shorts = 15.20) 
                      (h4 : total_cost = 752) 
                      (h5 : total_cost = players * (jersey + shorts + S)) 
                      : S = 6.80 := 
by
  sorry

end cost_of_socks_l1816_181648


namespace base_of_hill_depth_l1816_181653

theorem base_of_hill_depth : 
  ∀ (H : ℕ), 
  (H = 900) → 
  (1 / 4 * H = 225) :=
by
  intros H h
  sorry

end base_of_hill_depth_l1816_181653


namespace sum_of_three_numbers_is_520_l1816_181654

noncomputable def sum_of_three_numbers (x y z : ℝ) : ℝ :=
  x + y + z

theorem sum_of_three_numbers_is_520 (x y z : ℝ) (h1 : z = (1848 / 1540) * x) (h2 : z = 0.4 * y) (h3 : x + y = 400) :
  sum_of_three_numbers x y z = 520 :=
sorry

end sum_of_three_numbers_is_520_l1816_181654


namespace intersection_setA_setB_l1816_181688

noncomputable def setA : Set ℝ := { x : ℝ | abs (x - 1) < 2 }
noncomputable def setB : Set ℝ := { x : ℝ | (x - 2) / (x + 4) < 0 }

theorem intersection_setA_setB : 
  (setA ∩ setB) = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end intersection_setA_setB_l1816_181688


namespace min_living_allowance_inequality_l1816_181677

variable (x : ℝ)

-- The regulation stipulates that the minimum living allowance should not be less than 300 yuan.
def min_living_allowance_regulation (x : ℝ) : Prop := x >= 300

theorem min_living_allowance_inequality (x : ℝ) :
  min_living_allowance_regulation x ↔ x ≥ 300 := by
  sorry

end min_living_allowance_inequality_l1816_181677


namespace radius_of_sphere_in_truncated_cone_l1816_181625

-- Definitions based on conditions
def radius_top_base := 5
def radius_bottom_base := 24

-- Theorem statement (without proof)
theorem radius_of_sphere_in_truncated_cone :
    (∃ (R_s : ℝ),
      (R_s = Real.sqrt 180.5) ∧
      ∀ (h : ℝ),
      (h^2 + (radius_bottom_base - radius_top_base)^2 = (h + R_s)^2 - R_s^2)) :=
sorry

end radius_of_sphere_in_truncated_cone_l1816_181625


namespace parallelepiped_volume_l1816_181665

open Real

noncomputable def volume_parallelepiped
  (a b : ℝ) (angle : ℝ) (S : ℝ) (sin_30 : angle = π / 6) : ℝ :=
  let h := S / (2 * (a + b))
  let base_area := (a * b * sin (π / 6)) / 2
  base_area * h

theorem parallelepiped_volume 
  (a b : ℝ) (S : ℝ) (h : S ≠ 0 ∧ a > 0 ∧ b > 0) :
  volume_parallelepiped a b (π / 6) S (rfl) = (a * b * S) / (4 * (a + b)) :=
by
  sorry

end parallelepiped_volume_l1816_181665


namespace probability_sum_less_than_product_l1816_181635

def set_of_even_integers : Set ℕ := {2, 4, 6, 8, 10}

def sum_less_than_product (a b : ℕ) : Prop :=
  a + b < a * b

theorem probability_sum_less_than_product :
  let total_combinations := 25
  let valid_combinations := 16
  (valid_combinations / total_combinations : ℚ) = 16 / 25 :=
by
  sorry

end probability_sum_less_than_product_l1816_181635


namespace find_integer_solutions_l1816_181610

noncomputable def integer_solutions (x y z w : ℤ) : Prop :=
  x * y * z / w + y * z * w / x + z * w * x / y + w * x * y / z = 4

theorem find_integer_solutions :
  { (x, y, z, w) : ℤ × ℤ × ℤ × ℤ |
    integer_solutions x y z w } =
  {(1,1,1,1), (-1,-1,-1,-1), (-1,-1,1,1), (-1,1,-1,1),
   (-1,1,1,-1), (1,-1,-1,1), (1,-1,1,-1), (1,1,-1,-1)} :=
by
  sorry

end find_integer_solutions_l1816_181610


namespace base_prime_representation_450_l1816_181627

-- Define prime factorization property for number 450
def prime_factorization_450 := (450 = 2^1 * 3^2 * 5^2)

-- Define base prime representation concept
def base_prime_representation (n : ℕ) : ℕ := 
  if n = 450 then 122 else 0

-- Prove that the base prime representation of 450 is 122
theorem base_prime_representation_450 : 
  prime_factorization_450 →
  base_prime_representation 450 = 122 :=
by
  intros
  sorry

end base_prime_representation_450_l1816_181627


namespace find_k_l1816_181671

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + k^3 * x

theorem find_k (k : ℝ) (h : deriv (f k) 0 = 27) : k = 3 :=
by
  sorry

end find_k_l1816_181671


namespace distance_from_Martins_house_to_Lawrences_house_l1816_181644

def speed : ℝ := 2 -- Martin's speed is 2 miles per hour
def time : ℝ := 6  -- Time taken is 6 hours
def distance : ℝ := speed * time -- Distance formula

theorem distance_from_Martins_house_to_Lawrences_house : distance = 12 := by
  sorry

end distance_from_Martins_house_to_Lawrences_house_l1816_181644


namespace athletes_camp_duration_l1816_181690

theorem athletes_camp_duration
  (h : ℕ)
  (initial_athletes : ℕ := 300)
  (rate_leaving : ℕ := 28)
  (rate_entering : ℕ := 15)
  (hours_entering : ℕ := 7)
  (difference : ℕ := 7) :
  300 - 28 * h + 15 * 7 = 300 + 7 → h = 4 :=
by
  sorry

end athletes_camp_duration_l1816_181690


namespace product_of_remainders_one_is_one_l1816_181605

theorem product_of_remainders_one_is_one (a b : ℕ) (h1 : a % 3 = 1) (h2 : b % 3 = 1) : (a * b) % 3 = 1 :=
sorry

end product_of_remainders_one_is_one_l1816_181605


namespace domain_real_iff_l1816_181652

noncomputable def is_domain_ℝ (m : ℝ) : Prop :=
  ∀ x : ℝ, (m * x^2 + 4 * m * x + 3 ≠ 0)

theorem domain_real_iff (m : ℝ) :
  is_domain_ℝ m ↔ (0 ≤ m ∧ m < 3 / 4) :=
sorry

end domain_real_iff_l1816_181652


namespace shaded_area_of_square_with_quarter_circles_l1816_181689

theorem shaded_area_of_square_with_quarter_circles :
  let side_len : ℝ := 12
  let square_area := side_len * side_len
  let radius := side_len / 2
  let total_circle_area := 4 * (π * radius^2 / 4)
  let shaded_area := square_area - total_circle_area
  shaded_area = 144 - 36 * π := 
by
  sorry

end shaded_area_of_square_with_quarter_circles_l1816_181689


namespace rival_awards_eq_24_l1816_181608

-- Definitions
def Scott_awards : ℕ := 4
def Jessie_awards (scott: ℕ) : ℕ := 3 * scott
def Rival_awards (jessie: ℕ) : ℕ := 2 * jessie

-- Theorem to prove
theorem rival_awards_eq_24 : Rival_awards (Jessie_awards Scott_awards) = 24 := by
  sorry

end rival_awards_eq_24_l1816_181608


namespace quadratic_intersects_x_axis_l1816_181664

theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersects_x_axis_l1816_181664


namespace sticky_strips_used_l1816_181681

theorem sticky_strips_used 
  (total_decorations : ℕ) 
  (nails_used : ℕ) 
  (decorations_hung_with_nails_fraction : ℚ) 
  (decorations_hung_with_thumbtacks_fraction : ℚ) 
  (nails_used_eq : nails_used = 50)
  (decorations_hung_with_nails_fraction_eq : decorations_hung_with_nails_fraction = 2/3)
  (decorations_hung_with_thumbtacks_fraction_eq : decorations_hung_with_thumbtacks_fraction = 2/5)
  (total_decorations_eq : total_decorations = nails_used / decorations_hung_with_nails_fraction)
  : (total_decorations - nails_used - decorations_hung_with_thumbtacks_fraction * (total_decorations - nails_used)) = 15 := 
by {
  sorry
}

end sticky_strips_used_l1816_181681


namespace tan_sum_l1816_181628

theorem tan_sum (A B : ℝ) (h₁ : A = 17) (h₂ : B = 28) :
  Real.tan (A) + Real.tan (B) + Real.tan (A) * Real.tan (B) = 1 := 
by
  sorry

end tan_sum_l1816_181628


namespace new_average_marks_l1816_181695

theorem new_average_marks
  (orig_avg : ℕ) (num_papers : ℕ)
  (add_geography : ℕ) (add_history : ℕ)
  (H_orig_avg : orig_avg = 63)
  (H_num_papers : num_papers = 11)
  (H_add_geography : add_geography = 20)
  (H_add_history : add_history = 2) :
  (orig_avg * num_ppapers + add_geography + add_history) / num_papers = 65 :=
by
  -- Here would be the proof steps
  sorry

end new_average_marks_l1816_181695


namespace sqrt_equality_l1816_181675

theorem sqrt_equality :
  Real.sqrt ((18: ℝ) * (17: ℝ) * (16: ℝ) * (15: ℝ) + 1) = 271 :=
by
  sorry

end sqrt_equality_l1816_181675


namespace rectangle_width_is_14_l1816_181670

noncomputable def rectangleWidth (areaOfCircle : ℝ) (length : ℝ) : ℝ :=
  let r := Real.sqrt (areaOfCircle / Real.pi)
  2 * r

theorem rectangle_width_is_14 :
  rectangleWidth 153.93804002589985 18 = 14 :=
by 
  sorry

end rectangle_width_is_14_l1816_181670


namespace size_of_each_bottle_l1816_181603

-- Defining given conditions
def petals_per_ounce : ℕ := 320
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes : ℕ := 800
def bottles : ℕ := 20

-- Proving the size of each bottle in ounces
theorem size_of_each_bottle : (petals_per_rose * roses_per_bush * bushes / petals_per_ounce) / bottles = 12 := by
  sorry

end size_of_each_bottle_l1816_181603


namespace max_height_of_table_l1816_181656

theorem max_height_of_table (BC CA AB : ℕ) (h : ℝ) :
  BC = 24 →
  CA = 28 →
  AB = 32 →
  h ≤ (49 * Real.sqrt 60) / 19 :=
by
  intros
  sorry

end max_height_of_table_l1816_181656


namespace M_gt_N_l1816_181614

variable (a b : ℝ)

def M := 10 * a^2 + 2 * b^2 - 7 * a + 6
def N := a^2 + 2 * b^2 + 5 * a + 1

theorem M_gt_N : M a b > N a b := by
  sorry

end M_gt_N_l1816_181614


namespace friends_courses_l1816_181655

-- Define the notions of students and their properties
structure Student :=
  (first_name : String)
  (last_name : String)
  (year : ℕ)

-- Define the specific conditions from the problem
def students : List Student := [
  ⟨"Peter", "Krylov", 1⟩,
  ⟨"Nikolay", "Ivanov", 2⟩,
  ⟨"Boris", "Karpov", 3⟩,
  ⟨"Vasily", "Orlov", 4⟩
]

-- The main statement of the problem
theorem friends_courses :
  ∀ (s : Student), s ∈ students →
    (s.first_name = "Peter" → s.last_name = "Krylov" ∧ s.year = 1) ∧
    (s.first_name = "Nikolay" → s.last_name = "Ivanov" ∧ s.year = 2) ∧
    (s.first_name = "Boris" → s.last_name = "Karpov" ∧ s.year = 3) ∧
    (s.first_name = "Vasily" → s.last_name = "Orlov" ∧ s.year = 4) :=
by
  sorry

end friends_courses_l1816_181655


namespace initial_black_beads_l1816_181669

theorem initial_black_beads (B : ℕ) : 
  let white_beads := 51
  let black_beads_removed := 1 / 6 * B
  let white_beads_removed := 1 / 3 * white_beads
  let total_beads_removed := 32
  white_beads_removed + black_beads_removed = total_beads_removed →
  B = 90 :=
by
  sorry

end initial_black_beads_l1816_181669


namespace proof_of_problem_l1816_181624

theorem proof_of_problem (a b : ℝ) (h1 : a > b) (h2 : a * b = a / b) : b = 1 ∧ 0 < a :=
by
  sorry

end proof_of_problem_l1816_181624


namespace find_angle_A_find_perimeter_l1816_181678

-- Given problem conditions as Lean definitions
def triangle_sides (a b c : ℝ) : Prop :=
  ∃ B : ℝ, c = a * (Real.cos B + Real.sqrt 3 * Real.sin B)

def triangle_area (S a : ℝ) : Prop :=
  S = Real.sqrt 3 / 4 ∧ a = 1

-- Prove angle A
theorem find_angle_A (a b c S : ℝ) (hc : triangle_sides a b c) (ha : triangle_area S a) :
  ∃ A : ℝ, A = Real.pi / 6 := 
sorry

-- Prove perimeter
theorem find_perimeter (a b c S : ℝ) (hc : triangle_sides a b c) (ha : triangle_area S a) :
  ∃ P : ℝ, P = Real.sqrt 3 + 2 := 
sorry

end find_angle_A_find_perimeter_l1816_181678


namespace sqrt_fraction_evaluation_l1816_181642

theorem sqrt_fraction_evaluation :
  (Real.sqrt ((2 / 25) + (1 / 49) - (1 / 100)) = 3 / 10) :=
by sorry

end sqrt_fraction_evaluation_l1816_181642


namespace c_less_than_a_l1816_181611

variable (a b c : ℝ)

-- Conditions definitions
def are_negative : Prop := a < 0 ∧ b < 0 ∧ c < 0
def eq1 : Prop := c = 2 * (a + b)
def eq2 : Prop := c = 3 * (b - a)

-- Theorem statement
theorem c_less_than_a (h_neg : are_negative a b c) (h_eq1 : eq1 a b c) (h_eq2 : eq2 a b c) : c < a :=
  sorry

end c_less_than_a_l1816_181611


namespace house_assignment_l1816_181691

theorem house_assignment (n : ℕ) (assign : Fin n → Fin n) (pref : Fin n → Fin n → Fin n → Prop) :
  (∀ (p : Fin n), ∃ (better_assign : Fin n → Fin n),
    (∃ q, pref p (assign p) (better_assign p) ∧ pref q (assign q) (better_assign p) ∧ better_assign q ≠ assign q)
  ) → (∃ p, pref p (assign p) (assign p))
:= sorry

end house_assignment_l1816_181691


namespace bill_spots_39_l1816_181651

theorem bill_spots_39 (P : ℕ) (h1 : P + (2 * P - 1) = 59) : 2 * P - 1 = 39 :=
by sorry

end bill_spots_39_l1816_181651


namespace systematic_sampling_interval_l1816_181619

-- Define the total number of students and sample size
def N : ℕ := 1200
def n : ℕ := 40

-- Define the interval calculation for systematic sampling
def k : ℕ := N / n

-- Prove that the interval k is 30
theorem systematic_sampling_interval : k = 30 := by
sorry

end systematic_sampling_interval_l1816_181619


namespace bijection_if_injective_or_surjective_l1816_181609

variables {X Y : Type} [Fintype X] [Fintype Y] (f : X → Y)

theorem bijection_if_injective_or_surjective (hX : Fintype.card X = Fintype.card Y)
  (hf : Function.Injective f ∨ Function.Surjective f) : Function.Bijective f :=
by
  sorry

end bijection_if_injective_or_surjective_l1816_181609


namespace boxes_needed_l1816_181612

noncomputable def living_room_length : ℝ := 16
noncomputable def living_room_width : ℝ := 20
noncomputable def sq_ft_per_box : ℝ := 10
noncomputable def already_floored : ℝ := 250

theorem boxes_needed : 
  (living_room_length * living_room_width - already_floored) / sq_ft_per_box = 7 :=
by 
  sorry

end boxes_needed_l1816_181612


namespace smores_cost_calculation_l1816_181673

variable (people : ℕ) (s'mores_per_person : ℕ) (s'mores_per_set : ℕ) (cost_per_set : ℕ)

theorem smores_cost_calculation
  (h1 : s'mores_per_person = 3)
  (h2 : people = 8)
  (h3 : s'mores_per_set = 4)
  (h4 : cost_per_set = 3):
  (people * s'mores_per_person / s'mores_per_set) * cost_per_set = 18 := 
by
  sorry

end smores_cost_calculation_l1816_181673


namespace boiling_point_of_water_l1816_181638

theorem boiling_point_of_water :
  (boiling_point_F : ℝ) = 212 →
  (boiling_point_C : ℝ) = (5 / 9) * (boiling_point_F - 32) →
  boiling_point_C = 100 :=
by
  intro h1 h2
  sorry

end boiling_point_of_water_l1816_181638


namespace baby_polar_bear_playing_hours_l1816_181621

-- Define the conditions
def total_hours_in_a_day : ℕ := 24
def total_central_angle : ℕ := 360
def angle_sleeping : ℕ := 130
def angle_eating : ℕ := 110

-- Main theorem statement
theorem baby_polar_bear_playing_hours :
  let angle_playing := total_central_angle - angle_sleeping - angle_eating
  let fraction_playing := angle_playing / total_central_angle
  let hours_playing := fraction_playing * total_hours_in_a_day
  hours_playing = 8 := by
  sorry

end baby_polar_bear_playing_hours_l1816_181621


namespace largest_square_l1816_181661

def sticks_side1 : List ℕ := [4, 4, 2, 3]
def sticks_side2 : List ℕ := [4, 4, 3, 1, 1]
def sticks_side3 : List ℕ := [4, 3, 3, 2, 1]
def sticks_side4 : List ℕ := [3, 3, 3, 2, 2]

def sum_of_sticks (sticks : List ℕ) : ℕ := sticks.foldl (· + ·) 0

theorem largest_square (h1 : sum_of_sticks sticks_side1 = 13)
                      (h2 : sum_of_sticks sticks_side2 = 13)
                      (h3 : sum_of_sticks sticks_side3 = 13)
                      (h4 : sum_of_sticks sticks_side4 = 13) :
  13 = 13 := by
  sorry

end largest_square_l1816_181661


namespace card_subsets_l1816_181645

theorem card_subsets (A : Finset ℕ) (hA_card : A.card = 3) : (A.powerset.card = 8) :=
sorry

end card_subsets_l1816_181645


namespace subtract_from_40_squared_l1816_181626

theorem subtract_from_40_squared : 39 * 39 = 40 * 40 - 79 := by
  sorry

end subtract_from_40_squared_l1816_181626


namespace sum_of_squares_l1816_181647

theorem sum_of_squares (n : Nat) (h : n = 2005^2) : 
  ∃ a1 b1 a2 b2 a3 b3 a4 b4 : Int, 
    (n = a1^2 + b1^2 ∧ a1 ≠ 0 ∧ b1 ≠ 0) ∧ 
    (n = a2^2 + b2^2 ∧ a2 ≠ 0 ∧ b2 ≠ 0) ∧ 
    (n = a3^2 + b3^2 ∧ a3 ≠ 0 ∧ b3 ≠ 0) ∧ 
    (n = a4^2 + b4^2 ∧ a4 ≠ 0 ∧ b4 ≠ 0) ∧ 
    (a1, b1) ≠ (a2, b2) ∧ 
    (a1, b1) ≠ (a3, b3) ∧ 
    (a1, b1) ≠ (a4, b4) ∧ 
    (a2, b2) ≠ (a3, b3) ∧ 
    (a2, b2) ≠ (a4, b4) ∧ 
    (a3, b3) ≠ (a4, b4) :=
by
  sorry

end sum_of_squares_l1816_181647


namespace divisible_bc_ad_l1816_181634

open Int

theorem divisible_bc_ad (a b c d m : ℤ) (hm : 0 < m)
  (h1 : m ∣ a * c)
  (h2 : m ∣ b * d)
  (h3 : m ∣ (b * c + a * d)) :
  m ∣ b * c ∧ m ∣ a * d :=
by
  sorry

end divisible_bc_ad_l1816_181634


namespace find_angles_l1816_181613

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  2 * y = x + z

theorem find_angles (a : ℝ) (h1 : 0 < a) (h2 : a < 360)
  (h3 : is_arithmetic_sequence (Real.sin a) (Real.sin (2 * a)) (Real.sin (3 * a))) :
  a = 90 ∨ a = 270 := by
  sorry

end find_angles_l1816_181613


namespace g_half_eq_neg_one_l1816_181637

noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem g_half_eq_neg_one : g (1/2) = -1 := by 
  sorry

end g_half_eq_neg_one_l1816_181637


namespace weight_of_lightest_dwarf_l1816_181657

noncomputable def weight_of_dwarf (n : ℕ) (x : ℝ) : ℝ := 5 - (n - 1) * x

theorem weight_of_lightest_dwarf :
  ∃ x : ℝ, 
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 101 → weight_of_dwarf 1 x = 5) ∧
    (weight_of_dwarf 76 x + weight_of_dwarf 77 x + weight_of_dwarf 78 x + weight_of_dwarf 79 x + weight_of_dwarf 80 x =
     weight_of_dwarf 96 x + weight_of_dwarf 97 x + weight_of_dwarf 98 x + weight_of_dwarf 99 x + weight_of_dwarf 100 x + weight_of_dwarf 101 x) →
    weight_of_dwarf 101 x = 2.5 :=
by
  sorry

end weight_of_lightest_dwarf_l1816_181657


namespace team_a_wins_3_2_prob_l1816_181649

-- Definitions for the conditions in the problem
def prob_win_first_four : ℚ := 2 / 3
def prob_win_fifth : ℚ := 1 / 2

-- Definitions related to combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Main statement: Proving the probability of winning 3:2
theorem team_a_wins_3_2_prob :
  (C 4 2 * (prob_win_first_four ^ 2) * ((1 - prob_win_first_four) ^ 2) * prob_win_fifth) = 4 / 27 := 
sorry

end team_a_wins_3_2_prob_l1816_181649


namespace teamA_worked_days_l1816_181620

def teamA_days_to_complete := 10
def teamB_days_to_complete := 15
def teamC_days_to_complete := 20
def total_days := 6
def teamA_halfway_withdrew := true

theorem teamA_worked_days : 
  ∀ (T_A T_B T_C total: ℕ) (halfway_withdrawal: Bool),
    T_A = teamA_days_to_complete ->
    T_B = teamB_days_to_complete ->
    T_C = teamC_days_to_complete ->
    total = total_days ->
    halfway_withdrawal = teamA_halfway_withdrew ->
    (total / 2) * (1 / T_A + 1 / T_B + 1 / T_C) = 3 := 
by 
  sorry

end teamA_worked_days_l1816_181620


namespace sequence_and_sum_l1816_181616

-- Given conditions as definitions
def a₁ : ℕ := 1

def recurrence (a_n a_n1 : ℕ) (n : ℕ) : Prop := (a_n1 = 3 * a_n * (1 + (1 / n : ℝ)))

-- Stating the theorem
theorem sequence_and_sum (a : ℕ → ℕ) (S : ℕ → ℝ) :
  (a 1 = a₁) →
  (∀ n, recurrence (a n) (a (n + 1)) n) →
  (∀ n, a n = n * 3 ^ (n - 1)) ∧
  (∀ n, S n = (2 * n - 1) * 3 ^ n / 4 + 1 / 4) :=
by
  sorry

end sequence_and_sum_l1816_181616


namespace ratio_of_areas_l1816_181629

noncomputable def circumferences_equal_arcs (C1 C2 : ℝ) (k1 k2 : ℕ) : Prop :=
  (k1 : ℝ) / 360 * C1 = (k2 : ℝ) / 360 * C2

theorem ratio_of_areas (C1 C2 : ℝ) (h : circumferences_equal_arcs C1 C2 60 30) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_l1816_181629


namespace binom_identity_l1816_181641

-- Definition: Combinatorial coefficient (binomial coefficient)
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_identity (n k : ℕ) (h : k ≤ n) :
  binom (n + 1) k = binom n k + binom n (k - 1) := by
  sorry

end binom_identity_l1816_181641


namespace sin_beta_value_l1816_181685

theorem sin_beta_value (alpha beta : ℝ) (h1 : 0 < alpha) (h2 : alpha < beta) (h3 : beta < π / 2)
  (h4 : Real.sin alpha = 3 / 5) (h5 : Real.cos (alpha - beta) = 12 / 13) : Real.sin beta = 56 / 65 := by
  sorry

end sin_beta_value_l1816_181685


namespace negation_of_tan_one_l1816_181646

theorem negation_of_tan_one :
  (∃ x : ℝ, Real.tan x = 1) ↔ ¬ (∀ x : ℝ, Real.tan x ≠ 1) :=
by
  sorry

end negation_of_tan_one_l1816_181646


namespace car_X_travel_distance_l1816_181680

def car_distance_problem (speed_X speed_Y : ℝ) (delay : ℝ) : ℝ :=
  let t := 7 -- duration in hours computed in the provided solution
  speed_X * t

theorem car_X_travel_distance
  (speed_X speed_Y : ℝ) (delay : ℝ)
  (h_speed_X : speed_X = 35) (h_speed_Y : speed_Y = 39) (h_delay : delay = 48 / 60) :
  car_distance_problem speed_X speed_Y delay = 245 :=
by
  rw [h_speed_X, h_speed_Y, h_delay]
  -- compute the given car distance problem using the values provided
  sorry

end car_X_travel_distance_l1816_181680


namespace ratio_area_triangle_circle_l1816_181643

open Real

theorem ratio_area_triangle_circle
  (l r : ℝ)
  (h : ℝ := sqrt 2 * l)
  (h_eq_perimeter : 2 * l + h = 2 * π * r) :
  (1 / 2 * l^2) / (π * r^2) = (π * (3 - 2 * sqrt 2)) / 2 :=
by
  sorry

end ratio_area_triangle_circle_l1816_181643


namespace angle_between_diagonals_l1816_181666

variables (α β : ℝ) 

theorem angle_between_diagonals (α β : ℝ) :
  ∃ γ : ℝ, γ = Real.arccos (Real.sin α * Real.sin β) :=
by
  sorry

end angle_between_diagonals_l1816_181666


namespace find_abcde_l1816_181662

noncomputable def find_five_digit_number (a b c d e : ℕ) : ℕ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem find_abcde
  (a b c d e : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 0 ≤ e ∧ e ≤ 9)
  (h6 : a ≠ 0)
  (h7 : (10 * a + b + 10 * b + c) * (10 * b + c + 10 * c + d) * (10 * c + d + 10 * d + e) = 157605) :
  find_five_digit_number a b c d e = 12345 ∨ find_five_digit_number a b c d e = 21436 :=
sorry

end find_abcde_l1816_181662


namespace trajectory_eq_of_moving_point_Q_l1816_181697

-- Define the conditions and the correct answer
theorem trajectory_eq_of_moving_point_Q 
(a b : ℝ) (h : a > b) (h_pos : b > 0)
(P Q : ℝ × ℝ)
(h_ellipse : (P.1^2) / (a^2) + (P.2^2) / (b^2) = 1)
(h_Q : Q = (P.1 * 2, P.2 * 2)) :
  (Q.1^2) / (4 * a^2) + (Q.2^2) / (4 * b^2) = 1 :=
by 
  sorry

end trajectory_eq_of_moving_point_Q_l1816_181697


namespace staff_member_pays_l1816_181667

noncomputable def calculate_final_price (d : ℝ) : ℝ :=
  let discounted_price := 0.55 * d
  let staff_discounted_price := 0.33 * d
  let final_price := staff_discounted_price + 0.08 * staff_discounted_price
  final_price

theorem staff_member_pays (d : ℝ) : calculate_final_price d = 0.3564 * d :=
by
  unfold calculate_final_price
  sorry

end staff_member_pays_l1816_181667


namespace homothety_transformation_l1816_181633

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V]

/-- Definition of a homothety transformation -/
def homothety (S A A' : V) (k : ℝ) : Prop :=
  A' = k • A + (1 - k) • S

theorem homothety_transformation (S A A' : V) (k : ℝ) :
  homothety S A A' k ↔ A' = k • A + (1 - k) • S := 
by
  sorry

end homothety_transformation_l1816_181633


namespace largest_divisor_of_square_divisible_by_24_l1816_181650

theorem largest_divisor_of_square_divisible_by_24 (n : ℕ) (h₁ : n > 0) (h₂ : 24 ∣ n^2) (h₃ : ∀ k : ℕ, k ∣ n → k ≤ 8) : n = 24 := 
sorry

end largest_divisor_of_square_divisible_by_24_l1816_181650


namespace solve_n_m_equation_l1816_181699

theorem solve_n_m_equation : 
  ∃ (n m : ℤ), n^4 - 2*n^2 = m^2 + 38 ∧ ((n, m) = (3, 5) ∨ (n, m) = (3, -5) ∨ (n, m) = (-3, 5) ∨ (n, m) = (-3, -5)) :=
by { sorry }

end solve_n_m_equation_l1816_181699


namespace kittens_count_l1816_181682

def initial_kittens : ℕ := 8
def additional_kittens : ℕ := 2
def total_kittens : ℕ := 10

theorem kittens_count : initial_kittens + additional_kittens = total_kittens := by
  -- Proof will go here
  sorry

end kittens_count_l1816_181682


namespace no_integer_polynomial_exists_l1816_181600

theorem no_integer_polynomial_exists 
    (a b c d : ℤ) (h : a ≠ 0) (P : ℤ → ℤ) 
    (h1 : ∀ x, P x = a * x ^ 3 + b * x ^ 2 + c * x + d)
    (h2 : P 4 = 1) (h3 : P 7 = 2) : 
    false := 
by
    sorry

end no_integer_polynomial_exists_l1816_181600


namespace ratio_of_areas_l1816_181693

variables (s : ℝ)

def side_length_square := s
def longer_side_rect := 1.2 * s
def shorter_side_rect := 0.8 * s

noncomputable def area_rectangle := longer_side_rect s * shorter_side_rect s
noncomputable def area_triangle := (1 / 2) * (longer_side_rect s * shorter_side_rect s)

theorem ratio_of_areas :
  (area_triangle s) / (area_rectangle s) = 1 / 2 :=
by
  sorry

end ratio_of_areas_l1816_181693


namespace condition_I_condition_II_l1816_181679

noncomputable def f (x a : ℝ) : ℝ := |x - a|

-- Condition (I) proof problem
theorem condition_I (x : ℝ) (a : ℝ) (h : a = 1) :
  f x a ≥ 4 - |x - 1| ↔ (x ≤ -1 ∨ x ≥ 3) :=
by sorry

-- Condition (II) proof problem
theorem condition_II (a : ℝ) (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_f : ∀ x, f x a ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2)
    (h_eq : 1/m + 1/(2*n) = a) : mn ≥ 2 :=
by sorry

end condition_I_condition_II_l1816_181679


namespace chocolates_bought_l1816_181618

theorem chocolates_bought (C S : ℝ) (h1 : N * C = 45 * S) (h2 : 80 = ((S - C) / C) * 100) : 
  N = 81 :=
by
  sorry

end chocolates_bought_l1816_181618


namespace find_angle_A_find_area_l1816_181668

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def law_c1 (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + c * Real.cos A = -2 * b * Real.cos A

def law_c2 (a : ℝ) : Prop := a = 2 * Real.sqrt 3
def law_c3 (b c : ℝ) : Prop := b + c = 4

-- Questions
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c) : 
  A = 2 * Real.pi / 3 :=
sorry

theorem find_area (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c)
  (hA : A = 2 * Real.pi / 3) : 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
sorry

end find_angle_A_find_area_l1816_181668


namespace probability_all_same_color_l1816_181623

def total_marbles := 15
def red_marbles := 4
def white_marbles := 5
def blue_marbles := 6

def prob_all_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
def prob_all_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
def prob_all_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2))

def prob_all_same_color := prob_all_red + prob_all_white + prob_all_blue

theorem probability_all_same_color :
  prob_all_same_color = (34/455) :=
by sorry

end probability_all_same_color_l1816_181623


namespace original_polynomial_l1816_181683

theorem original_polynomial {x y : ℝ} (P : ℝ) :
  P - (-x^2 * y) = 3 * x^2 * y - 2 * x * y - 1 → P = 2 * x^2 * y - 2 * x * y - 1 :=
sorry

end original_polynomial_l1816_181683


namespace range_of_a_l1816_181672

def is_in_third_quadrant (A : ℝ × ℝ) : Prop :=
  A.1 < 0 ∧ A.2 < 0

theorem range_of_a (a : ℝ) (h : is_in_third_quadrant (a, a - 1)) : a < 0 :=
by
  sorry

end range_of_a_l1816_181672


namespace prove_ratio_l1816_181630

noncomputable def box_dimensions : ℝ × ℝ × ℝ := (2, 3, 5)
noncomputable def d := (2 * 3 * 5 : ℝ)
noncomputable def a := ((4 * Real.pi) / 3 : ℝ)
noncomputable def b := (10 * Real.pi : ℝ)
noncomputable def c := (62 : ℝ)

theorem prove_ratio :
  (b * c) / (a * d) = (15.5 : ℝ) :=
by
  unfold a b c d
  sorry

end prove_ratio_l1816_181630


namespace complement_event_l1816_181640

def total_students : ℕ := 4
def males : ℕ := 2
def females : ℕ := 2
def choose2 (n : ℕ) := n * (n - 1) / 2

noncomputable def eventA : ℕ := males * females
noncomputable def eventB : ℕ := choose2 males
noncomputable def eventC : ℕ := choose2 females

theorem complement_event {total_students males females : ℕ}
  (h_total : total_students = 4)
  (h_males : males = 2)
  (h_females : females = 2) :
  (total_students.choose 2 - (eventB + eventC)) / total_students.choose 2 = 1 / 3 :=
by
  sorry

end complement_event_l1816_181640


namespace range_of_m_l1816_181692

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := x^2 - x + 1

-- Define the interval
def interval (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 2

-- Prove the range of m
theorem range_of_m (m : ℝ) : (∀ x : ℝ, interval x → f x > 2 * x + m) ↔ m < - 5 / 4 :=
by
  -- This is the theorem statement, hence the proof starts here
  sorry

end range_of_m_l1816_181692


namespace cost_to_fill_pool_l1816_181617

-- Definitions based on the conditions
def cubic_foot_to_liters: ℕ := 25
def pool_depth: ℕ := 10
def pool_width: ℕ := 6
def pool_length: ℕ := 20
def cost_per_liter: ℕ := 3

-- Statement to be proved
theorem cost_to_fill_pool : 
  (pool_depth * pool_width * pool_length * cubic_foot_to_liters * cost_per_liter) = 90000 := 
by 
  sorry

end cost_to_fill_pool_l1816_181617


namespace cevian_concurrency_l1816_181604

-- Definitions for the acute triangle and the angles
structure AcuteTriangle (α β γ : ℝ) :=
  (A B C : ℝ)
  (acute_α : α > 0 ∧ α < π / 2)
  (acute_β : β > 0 ∧ β < π / 2)
  (acute_γ : γ > 0 ∧ γ < π / 2)
  (triangle_sum : α + β + γ = π)

-- Definition for the concurrency of cevians
def cevians_concurrent (α β γ : ℝ) (t : AcuteTriangle α β γ) :=
  ∀ (A₁ B₁ C₁ : ℝ), sorry -- placeholder

-- The main theorem with the proof of concurrency
theorem cevian_concurrency (α β γ : ℝ) (t : AcuteTriangle α β γ) :
  ∃ (A₁ B₁ C₁ : ℝ), cevians_concurrent α β γ t :=
  sorry -- proof to be provided


end cevian_concurrency_l1816_181604


namespace divisor_of_1025_l1816_181606

theorem divisor_of_1025 : ∃ k : ℕ, 41 * k = 1025 :=
  sorry

end divisor_of_1025_l1816_181606


namespace remainder_division_l1816_181694

theorem remainder_division (L S R : ℕ) (h1 : L - S = 1325) (h2 : L = 1650) (h3 : L = 5 * S + R) : 
  R = 25 :=
sorry

end remainder_division_l1816_181694


namespace segment_length_l1816_181676

theorem segment_length (AB BC AC : ℝ) (hAB : AB = 4) (hBC : BC = 3) :
  AC = 7 ∨ AC = 1 :=
sorry

end segment_length_l1816_181676


namespace windows_per_floor_is_3_l1816_181631

-- Given conditions
variables (W : ℕ)
def windows_each_floor (W : ℕ) : Prop :=
  (3 * 2 * W) - 2 = 16

-- Correct answer
theorem windows_per_floor_is_3 : windows_each_floor 3 :=
by 
  sorry

end windows_per_floor_is_3_l1816_181631


namespace chosen_number_l1816_181622

theorem chosen_number (x : ℕ) (h : (x / 12) - 240 = 8) : x = 2976 :=
sorry

end chosen_number_l1816_181622


namespace Clea_escalator_time_l1816_181658

variable {s : ℝ} -- speed of the escalator at its normal operating speed
variable {c : ℝ} -- speed of Clea walking down the escalator
variable {d : ℝ} -- distance of the escalator

theorem Clea_escalator_time :
  (30 * (c + s) = 72 * c) →
  (s = (7 * c) / 5) →
  (t = (72 * c) / ((3 / 2) * s)) →
  t = 240 / 7 :=
by
  sorry

end Clea_escalator_time_l1816_181658


namespace point_in_second_quadrant_l1816_181602

structure Point where
  x : Int
  y : Int

-- Define point P
def P : Point := { x := -1, y := 2 }

-- Define the second quadrant condition
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- The mathematical statement to prove
theorem point_in_second_quadrant : second_quadrant P := by
  sorry

end point_in_second_quadrant_l1816_181602


namespace scientific_notation_819000_l1816_181632

theorem scientific_notation_819000 :
  ∃ (a : ℝ) (n : ℤ), 819000 = a * 10 ^ n ∧ a = 8.19 ∧ n = 5 :=
by
  -- Proof goes here
  sorry

end scientific_notation_819000_l1816_181632


namespace polynomial_divisibility_by_6_l1816_181636

theorem polynomial_divisibility_by_6 (a b c : ℤ) (h : (a + b + c) % 6 = 0) : (a^5 + b^3 + c) % 6 = 0 :=
sorry

end polynomial_divisibility_by_6_l1816_181636


namespace range_of_a_l1816_181696

-- Define the problem statement in Lean 4
theorem range_of_a (a : ℝ) : (∀ x : ℝ, ((x^2 - (a-1)*x + 1) > 0)) → (-1 < a ∧ a < 3) :=
by
  intro h
  sorry -- Proof to be filled in

end range_of_a_l1816_181696


namespace middle_card_four_or_five_l1816_181615

def three_cards (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 15 ∧ a < b ∧ b < c

theorem middle_card_four_or_five (a b c : ℕ) :
  three_cards a b c → (b = 4 ∨ b = 5) :=
by
  sorry

end middle_card_four_or_five_l1816_181615
