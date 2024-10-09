import Mathlib

namespace usual_time_to_catch_bus_l79_7962

variable {S T T' D : ℝ}

theorem usual_time_to_catch_bus (h1 : D = S * T)
  (h2 : D = (4 / 5) * S * T')
  (h3 : T' = T + 4) : T = 16 := by
  sorry

end usual_time_to_catch_bus_l79_7962


namespace michael_ratio_l79_7980

-- Definitions
def Michael_initial := 42
def Brother_initial := 17

-- Conditions
def Brother_after_candy_purchase := 35
def Candy_cost := 3
def Brother_before_candy := Brother_after_candy_purchase + Candy_cost
def x := Brother_before_candy - Brother_initial

-- Prove the ratio of the money Michael gave to his brother to his initial amount is 1:2
theorem michael_ratio :
  x * 2 = Michael_initial := by
  sorry

end michael_ratio_l79_7980


namespace alexa_fractions_l79_7955

theorem alexa_fractions (alexa_days ethans_days : ℕ) 
  (h1 : alexa_days = 9) (h2 : ethans_days = 12) : 
  alexa_days / ethans_days = 3 / 4 := 
by 
  sorry

end alexa_fractions_l79_7955


namespace sum_nat_numbers_from_1_to_5_l79_7995

theorem sum_nat_numbers_from_1_to_5 : (1 + 2 + 3 + 4 + 5 = 15) :=
by
  sorry

end sum_nat_numbers_from_1_to_5_l79_7995


namespace dream_clock_time_condition_l79_7915

theorem dream_clock_time_condition (x : ℝ) (h1 : 0 < x) (h2 : x < 1)
  (h3 : (120 + 0.5 * 60 * x) = (240 - 6 * 60 * x)) :
  (4 + x) = 4 + 36 + 12 / 13 := by sorry

end dream_clock_time_condition_l79_7915


namespace vans_needed_for_trip_l79_7918

theorem vans_needed_for_trip (total_people : ℕ) (van_capacity : ℕ) (h_total_people : total_people = 24) (h_van_capacity : van_capacity = 8) : ℕ :=
  let exact_vans := total_people / van_capacity
  let vans_needed := if total_people % van_capacity = 0 then exact_vans else exact_vans + 1
  have h_exact : exact_vans = 3 := by sorry
  have h_vans_needed : vans_needed = 4 := by sorry
  vans_needed

end vans_needed_for_trip_l79_7918


namespace fewer_people_third_bus_l79_7979

noncomputable def people_first_bus : Nat := 12
noncomputable def people_second_bus : Nat := 2 * people_first_bus
noncomputable def people_fourth_bus : Nat := people_first_bus + 9
noncomputable def total_people : Nat := 75
noncomputable def people_other_buses : Nat := people_first_bus + people_second_bus + people_fourth_bus
noncomputable def people_third_bus : Nat := total_people - people_other_buses

theorem fewer_people_third_bus :
  people_second_bus - people_third_bus = 6 :=
by
  sorry

end fewer_people_third_bus_l79_7979


namespace complex_magnitude_addition_l79_7929

theorem complex_magnitude_addition :
  (Complex.abs (3 / 4 - 3 * Complex.I) + 5 / 12) = (9 * Real.sqrt 17 + 5) / 12 := 
  sorry

end complex_magnitude_addition_l79_7929


namespace sarah_probability_l79_7951

noncomputable def probability_odd_product_less_than_20 : ℚ :=
  let total_possibilities := 36
  let favorable_pairs := [(1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3)]
  let favorable_count := favorable_pairs.length
  let probability := favorable_count / total_possibilities
  probability

theorem sarah_probability : probability_odd_product_less_than_20 = 2 / 9 :=
by
  sorry

end sarah_probability_l79_7951


namespace boxes_with_neither_l79_7945

-- Definitions based on the conditions given
def total_boxes : Nat := 12
def boxes_with_markers : Nat := 8
def boxes_with_erasers : Nat := 5
def boxes_with_both : Nat := 4

-- The statement we want to prove
theorem boxes_with_neither :
  total_boxes - (boxes_with_markers + boxes_with_erasers - boxes_with_both) = 3 :=
by
  sorry

end boxes_with_neither_l79_7945


namespace find_divisor_l79_7906

theorem find_divisor (d : ℕ) (h : 127 = d * 5 + 2) : d = 25 :=
by 
  -- Given conditions
  -- 127 = d * 5 + 2
  -- We need to prove d = 25
  sorry

end find_divisor_l79_7906


namespace number_of_white_balls_l79_7901

theorem number_of_white_balls (x : ℕ) : (3 : ℕ) + x = 12 → x = 9 :=
by
  intros h
  sorry

end number_of_white_balls_l79_7901


namespace one_hundred_fiftieth_digit_l79_7964

theorem one_hundred_fiftieth_digit : (0.135 : Real) * 10^150 = 1 :=
by
  sorry

end one_hundred_fiftieth_digit_l79_7964


namespace option_d_correct_l79_7900

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)
def M : Set ℝ := {x | f x = 0}

theorem option_d_correct : ({1, 3} ∪ {2, 3} : Set ℝ) = M := by
  sorry

end option_d_correct_l79_7900


namespace fries_remaining_time_l79_7966

theorem fries_remaining_time (recommended_time_min : ℕ) (time_in_oven_sec : ℕ)
    (h1 : recommended_time_min = 5)
    (h2 : time_in_oven_sec = 45) :
    (recommended_time_min * 60 - time_in_oven_sec = 255) :=
by
  sorry

end fries_remaining_time_l79_7966


namespace units_digit_31_2020_units_digit_37_2020_l79_7994

theorem units_digit_31_2020 : ((31 ^ 2020) % 10) = 1 := by
  sorry

theorem units_digit_37_2020 : ((37 ^ 2020) % 10) = 1 := by
  sorry

end units_digit_31_2020_units_digit_37_2020_l79_7994


namespace Ramsey_number_bound_l79_7910

noncomputable def Ramsey_number (k : ℕ) : ℕ := sorry

theorem Ramsey_number_bound (k : ℕ) (h : k ≥ 3) : Ramsey_number k > 2^(k / 2) := sorry

end Ramsey_number_bound_l79_7910


namespace gcd_of_8247_13619_29826_l79_7958

theorem gcd_of_8247_13619_29826 : Nat.gcd (Nat.gcd 8247 13619) 29826 = 3 := 
sorry

end gcd_of_8247_13619_29826_l79_7958


namespace boys_count_l79_7950

theorem boys_count (B G : ℕ) (h1 : B + G = 41) (h2 : 12 * B + 8 * G = 460) : B = 33 := 
by
  sorry

end boys_count_l79_7950


namespace sufficient_not_necessary_perpendicular_l79_7911

theorem sufficient_not_necessary_perpendicular (a : ℝ) :
  (∀ x y : ℝ, (a + 2) * x + 3 * a * y + 1 = 0 ∧
              (a - 2) * x + (a + 2) * y - 3 = 0 → false) ↔ a = -2 :=
sorry

end sufficient_not_necessary_perpendicular_l79_7911


namespace find_tan_beta_l79_7924

variable (α β : ℝ)

def condition1 : Prop := Real.tan α = 3
def condition2 : Prop := Real.tan (α + β) = 2

theorem find_tan_beta (h1 : condition1 α) (h2 : condition2 α β) : Real.tan β = -1 / 7 := 
by {
  sorry
}

end find_tan_beta_l79_7924


namespace Niklaus_walked_distance_l79_7973

noncomputable def MilesToFeet (miles : ℕ) : ℕ := miles * 5280
noncomputable def YardsToFeet (yards : ℕ) : ℕ := yards * 3

theorem Niklaus_walked_distance (n_feet : ℕ) :
  MilesToFeet 4 + YardsToFeet 975 + n_feet = 25332 → n_feet = 1287 := by
  sorry

end Niklaus_walked_distance_l79_7973


namespace correct_answer_l79_7972

def M : Set ℤ := {x | |x| < 5}

theorem correct_answer : {0} ⊆ M := by
  sorry

end correct_answer_l79_7972


namespace number_of_solutions_l79_7931

theorem number_of_solutions (θ : ℝ) (h : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  2 - 4 * Real.sin (2 * θ) + 3 * Real.cos (4 * θ) = 0 → 
  ∃ s : Fin 9, s.val = 8 :=
by
  sorry

end number_of_solutions_l79_7931


namespace day_of_month_l79_7965

/--
The 25th day of a particular month is a Monday. 
We need to prove that the 1st day of that month is a Friday.
-/
theorem day_of_month (h : (25 % 7 = 1)) : (1 % 7 = 5) :=
sorry

end day_of_month_l79_7965


namespace soy_sauce_bottle_size_l79_7937

theorem soy_sauce_bottle_size 
  (ounces_per_cup : ℕ)
  (cups_recipe1 : ℕ)
  (cups_recipe2 : ℕ)
  (cups_recipe3 : ℕ)
  (number_of_bottles : ℕ)
  (total_ounces_needed : ℕ)
  (ounces_per_bottle : ℕ) :
  ounces_per_cup = 8 →
  cups_recipe1 = 2 →
  cups_recipe2 = 1 →
  cups_recipe3 = 3 →
  number_of_bottles = 3 →
  total_ounces_needed = (cups_recipe1 + cups_recipe2 + cups_recipe3) * ounces_per_cup →
  ounces_per_bottle = total_ounces_needed / number_of_bottles →
  ounces_per_bottle = 16 :=
by
  sorry

end soy_sauce_bottle_size_l79_7937


namespace bound_on_f_l79_7909

theorem bound_on_f 
  (f : ℝ → ℝ) 
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ 1) 
  (h_zeros : f 0 = 0 ∧ f 1 = 0)
  (h_condition : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ x1 ≠ x2 → |f x2 - f x1| < |x2 - x1|) 
  : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 → |f x2 - f x1| < 1/2 :=
by
  sorry

end bound_on_f_l79_7909


namespace value_of_b_l79_7984

theorem value_of_b (b : ℝ) : 
  (∃ (x : ℝ), x^2 + b * x - 45 = 0 ∧ x = -4) →
  b = -29 / 4 :=
by
  -- Introduce the condition and rewrite it properly
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  -- Proceed with assumption that we have the condition and need to prove the statement
  sorry

end value_of_b_l79_7984


namespace range_of_m_l79_7943

-- Define points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (2, -2)

-- Define the line equation as a predicate
def line_l (m : ℝ) (p : ℝ × ℝ) : Prop := p.1 + m * p.2 + m = 0

-- Define the condition for the line intersecting the segment AB
def intersects_segment_AB (m : ℝ) : Prop :=
  let P : ℝ × ℝ := (0, -1)
  let k_PA := (P.2 - A.2) / (P.1 - A.1) -- Slope of PA
  let k_PB := (P.2 - B.2) / (P.1 - B.1) -- Slope of PB
  (k_PA <= -1 / m) ∧ (-1 / m <= k_PB)

-- State the theorem
theorem range_of_m : ∀ (m : ℝ), intersects_segment_AB m → (1/2 ≤ m ∧ m ≤ 2) :=
by sorry

end range_of_m_l79_7943


namespace space_per_bush_l79_7998

theorem space_per_bush (side_length : ℝ) (num_sides : ℝ) (num_bushes : ℝ) (h1 : side_length = 16) (h2 : num_sides = 3) (h3 : num_bushes = 12) :
  (num_sides * side_length) / num_bushes = 4 :=
by
  sorry

end space_per_bush_l79_7998


namespace train_speed_l79_7914

theorem train_speed (v : ℝ) 
  (h1 : 50 * 2.5 + v * 2.5 = 285) : v = 64 := 
by
  -- h1 unfolds conditions into the mathematical equation
  -- here we would have the proof steps, adding a "sorry" to skip proof steps.
  sorry

end train_speed_l79_7914


namespace satisfies_equation_l79_7985

noncomputable def y (b x : ℝ) : ℝ := (b + x) / (1 + b * x)

theorem satisfies_equation (b x : ℝ) :
  let y_val := y b x
  let y_prime := (1 - b^2) / (1 + b * x)^2
  y_val - x * y_prime = b * (1 + x^2 * y_prime) :=
by
  sorry

end satisfies_equation_l79_7985


namespace horner_rule_v3_is_36_l79_7960

def f (x : ℤ) : ℤ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_rule_v3_is_36 :
  let v0 := 1;
  let v1 := v0 * 3 + 0;
  let v2 := v1 * 3 + 2;
  let v3 := v2 * 3 + 3;
  v3 = 36 := 
by
  sorry

end horner_rule_v3_is_36_l79_7960


namespace elaine_rent_percentage_l79_7989

theorem elaine_rent_percentage (E : ℝ) (P : ℝ) 
  (h1 : E > 0) 
  (h2 : P > 0) 
  (h3 : 0.25 * 1.15 * E = 1.4375 * (P / 100) * E) : 
  P = 20 := 
sorry

end elaine_rent_percentage_l79_7989


namespace ellipse_parabola_four_intersections_intersection_points_lie_on_circle_l79_7978

-- Part A:
-- Define intersections of a given ellipse and parabola under conditions on m and n
theorem ellipse_parabola_four_intersections (m n : ℝ) :
  (3 / n < m) ∧ (m < (4 * m^2 + 9) / (4 * m)) ∧ (m > 3 / 2) →
  ∃ x y : ℝ, (x^2 / n + y^2 / 9 = 1) ∧ (y = x^2 - m) :=
sorry

-- Part B:
-- Prove four intersection points of given ellipse and parabola lie on same circle for m = n = 4
theorem intersection_points_lie_on_circle (x y : ℝ) :
  (4 / 4 + y^2 / 9 = 1) ∧ (y = x^2 - 4) →
  ∃ k l r : ℝ, ∀ x' y', ((x' - k)^2 + (y' - l)^2 = r^2) :=
sorry

end ellipse_parabola_four_intersections_intersection_points_lie_on_circle_l79_7978


namespace toothpicks_needed_base_1001_l79_7974

-- Define the number of small triangles at the base of the larger triangle
def base_triangle_count := 1001

-- Define the total number of small triangles using the sum of the first 'n' natural numbers
def total_small_triangles (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Calculate the total number of sides for all triangles if there was no sharing
def total_sides (n : ℕ) : ℕ :=
  3 * total_small_triangles n

-- Calculate the number of shared toothpicks
def shared_toothpicks (n : ℕ) : ℕ :=
  total_sides n / 2

-- Calculate the number of unshared perimeter toothpicks
def unshared_perimeter_toothpicks (n : ℕ) : ℕ :=
  3 * n

-- Calculate the total number of toothpicks required
def total_toothpicks (n : ℕ) : ℕ :=
  shared_toothpicks n + unshared_perimeter_toothpicks n

-- Prove that the total toothpicks required for the base of 1001 small triangles is 755255
theorem toothpicks_needed_base_1001 : total_toothpicks base_triangle_count = 755255 :=
by {
  sorry
}

end toothpicks_needed_base_1001_l79_7974


namespace solve_alcohol_mixture_problem_l79_7993

theorem solve_alcohol_mixture_problem (x y : ℝ) 
(h1 : x + y = 18) 
(h2 : 0.75 * x + 0.15 * y = 9) 
: x = 10.5 ∧ y = 7.5 :=
by 
  sorry

end solve_alcohol_mixture_problem_l79_7993


namespace identity_holds_for_all_a_b_l79_7970

theorem identity_holds_for_all_a_b (a b : ℝ) :
  let x := a + 5 * b
  let y := 5 * a - b
  let z := 3 * a - 2 * b
  let t := 2 * a + 3 * b
  x^2 + y^2 = 2 * (z^2 + t^2) :=
by {
  let x := a + 5 * b
  let y := 5 * a - b
  let z := 3 * a - 2 * b
  let t := 2 * a + 3 * b
  sorry
}

end identity_holds_for_all_a_b_l79_7970


namespace complement_of_supplement_of_30_degrees_l79_7936

def supplementary_angle (x : ℕ) : ℕ := 180 - x
def complementary_angle (x : ℕ) : ℕ := if x > 90 then x - 90 else 90 - x

theorem complement_of_supplement_of_30_degrees : complementary_angle (supplementary_angle 30) = 60 := by
  sorry

end complement_of_supplement_of_30_degrees_l79_7936


namespace find_x_l79_7922

theorem find_x 
  (x : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (hP : P = (x, 6)) 
  (hcos : Real.cos θ = -4/5) 
  : x = -8 := 
sorry

end find_x_l79_7922


namespace lateral_surface_area_of_cone_l79_7959

theorem lateral_surface_area_of_cone (r h : ℝ) (r_is_4 : r = 4) (h_is_3 : h = 3) :
  ∃ A : ℝ, A = 20 * Real.pi := by
  sorry

end lateral_surface_area_of_cone_l79_7959


namespace unattainable_value_of_y_l79_7953

noncomputable def f (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

theorem unattainable_value_of_y :
  ∃ y : ℝ, y = -(1 / 3) ∧ ∀ x : ℝ, 3 * x + 4 ≠ 0 → f x ≠ y :=
by
  sorry

end unattainable_value_of_y_l79_7953


namespace first_year_after_2020_with_digit_sum_4_l79_7944

theorem first_year_after_2020_with_digit_sum_4 :
  ∃ x : ℕ, x > 2020 ∧ (Nat.digits 10 x).sum = 4 ∧ ∀ y : ℕ, y > 2020 ∧ (Nat.digits 10 y).sum = 4 → x ≤ y :=
sorry

end first_year_after_2020_with_digit_sum_4_l79_7944


namespace intersection_is_23_l79_7988

open Set

def setA : Set ℤ := {1, 2, 3, 4}
def setB : Set ℤ := {x | 2 ≤ x ∧ x ≤ 3}

theorem intersection_is_23 : setA ∩ setB = {2, 3} := 
by 
  sorry

end intersection_is_23_l79_7988


namespace find_c2013_l79_7913

theorem find_c2013 :
  ∀ (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ),
    (a 1 = 3) →
    (b 1 = 3) →
    (∀ n : ℕ, 1 ≤ n → a (n+1) - a n = 3) →
    (∀ n : ℕ, 1 ≤ n → b (n+1) = 3 * b n) →
    (∀ n : ℕ, c n = b (a n)) →
    c 2013 = 27^2013 := by
  sorry

end find_c2013_l79_7913


namespace incorrect_equation_a_neq_b_l79_7932

theorem incorrect_equation_a_neq_b (a b : ℝ) (h : a ≠ b) : a - b ≠ b - a :=
  sorry

end incorrect_equation_a_neq_b_l79_7932


namespace number_of_books_bought_l79_7967

def initial_books : ℕ := 35
def books_given_away : ℕ := 12
def final_books : ℕ := 56

theorem number_of_books_bought : initial_books - books_given_away + (final_books - (initial_books - books_given_away)) = final_books :=
by
  sorry

end number_of_books_bought_l79_7967


namespace money_distribution_l79_7963

-- Declare the variables and the conditions as hypotheses
theorem money_distribution (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 40) :
  B + C = 340 :=
by
  sorry

end money_distribution_l79_7963


namespace fewest_tiles_needed_l79_7983

def tiles_needed (tile_length tile_width region_length region_width : ℕ) : ℕ :=
  let length_tiles := (region_length + tile_length - 1) / tile_length
  let width_tiles := (region_width + tile_width - 1) / tile_width
  length_tiles * width_tiles

theorem fewest_tiles_needed :
  let tile_length := 2
  let tile_width := 5
  let region_length := 36
  let region_width := 72
  tiles_needed tile_length tile_width region_length region_width = 270 :=
by
  sorry

end fewest_tiles_needed_l79_7983


namespace hexagonal_pyramid_cross_section_distance_l79_7986

theorem hexagonal_pyramid_cross_section_distance
  (A1 A2 : ℝ) (distance_between_planes : ℝ)
  (A1_area : A1 = 125 * Real.sqrt 3)
  (A2_area : A2 = 500 * Real.sqrt 3)
  (distance_between_planes_eq : distance_between_planes = 10) :
  ∃ h : ℝ, h = 20 :=
by
  sorry

end hexagonal_pyramid_cross_section_distance_l79_7986


namespace Matias_sales_l79_7956

def books_sold (Tuesday Wednesday Thursday : Nat) : Prop :=
  Tuesday = 7 ∧ 
  Wednesday = 3 * Tuesday ∧ 
  Thursday = 3 * Wednesday ∧ 
  Tuesday + Wednesday + Thursday = 91

theorem Matias_sales
  (Tuesday Wednesday Thursday : Nat) :
  books_sold Tuesday Wednesday Thursday := by
  sorry

end Matias_sales_l79_7956


namespace peter_total_miles_l79_7916

-- Definitions based on the conditions
def minutes_per_mile : ℝ := 20
def miles_walked_already : ℝ := 1
def additional_minutes : ℝ := 30

-- The value we want to prove
def total_miles_to_walk : ℝ := 2.5

-- Theorem statement corresponding to the proof problem
theorem peter_total_miles :
  (additional_minutes / minutes_per_mile) + miles_walked_already = total_miles_to_walk :=
sorry

end peter_total_miles_l79_7916


namespace boat_speed_still_water_l79_7968

variable (V_b V_s : ℝ)

def upstream : Prop := V_b - V_s = 10
def downstream : Prop := V_b + V_s = 40

theorem boat_speed_still_water (h1 : upstream V_b V_s) (h2 : downstream V_b V_s) : V_b = 25 :=
by
  sorry

end boat_speed_still_water_l79_7968


namespace leap_day_2040_is_friday_l79_7907

def leap_day_day_of_week (start_year : ℕ) (start_day : ℕ) (end_year : ℕ) : ℕ :=
  let num_years := end_year - start_year
  let num_leap_years := (num_years + 4) / 4 -- number of leap years including start and end year
  let total_days := 365 * (num_years - num_leap_years) + 366 * num_leap_years
  let day_of_week := (total_days % 7 + start_day) % 7
  day_of_week

theorem leap_day_2040_is_friday :
  leap_day_day_of_week 2008 5 2040 = 5 := 
  sorry

end leap_day_2040_is_friday_l79_7907


namespace avg_age_boys_class_l79_7948

-- Definitions based on conditions
def avg_age_students : ℝ := 15.8
def avg_age_girls : ℝ := 15.4
def ratio_boys_girls : ℝ := 1.0000000000000044

-- Using the given conditions to define the average age of boys
theorem avg_age_boys_class (B G : ℕ) (A_b : ℝ) 
  (h1 : avg_age_students = (B * A_b + G * avg_age_girls) / (B + G)) 
  (h2 : B = ratio_boys_girls * G) : 
  A_b = 16.2 :=
  sorry

end avg_age_boys_class_l79_7948


namespace curve_is_circle_l79_7903

theorem curve_is_circle (ρ θ : ℝ) (h : ρ = 5 * Real.sin θ) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), ∀ (x y : ℝ),
  (x, y) = (ρ * Real.cos θ, ρ * Real.sin θ) → 
  (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2 :=
by
  existsi (0, 5 / 2), 5 / 2
  sorry

end curve_is_circle_l79_7903


namespace angle_B_in_arithmetic_sequence_l79_7933

theorem angle_B_in_arithmetic_sequence (A B C : ℝ) (h_triangle_sum : A + B + C = 180) (h_arithmetic_sequence : 2 * B = A + C) : B = 60 := 
by 
  -- proof omitted
  sorry

end angle_B_in_arithmetic_sequence_l79_7933


namespace relationship_y1_y2_l79_7926

theorem relationship_y1_y2
  (x1 y1 x2 y2 : ℝ)
  (hA : y1 = 3 * x1 + 4)
  (hB : y2 = 3 * x2 + 4)
  (h : x1 < x2) :
  y1 < y2 :=
sorry

end relationship_y1_y2_l79_7926


namespace no_real_solutions_quadratic_solve_quadratic_eq_l79_7997

-- For Equation (1)

theorem no_real_solutions_quadratic (a b c : ℝ) (h_eq : a = 3 ∧ b = -4 ∧ c = 5 ∧ (b^2 - 4 * a * c < 0)) :
  ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 := 
by
  sorry

-- For Equation (2)

theorem solve_quadratic_eq {x : ℝ} (h_eq : (x + 1) * (x + 2) = 2 * x + 4) :
  x = -2 ∨ x = 1 :=
by
  sorry

end no_real_solutions_quadratic_solve_quadratic_eq_l79_7997


namespace find_third_root_l79_7981

theorem find_third_root (a b : ℚ) 
  (h1 : a * 1^3 + (a + 3 * b) * 1^2 + (b - 4 * a) * 1 + (6 - a) = 0)
  (h2 : a * (-3)^3 + (a + 3 * b) * (-3)^2 + (b - 4 * a) * (-3) + (6 - a) = 0)
  : ∃ c : ℚ, c = 7 / 13 :=
sorry

end find_third_root_l79_7981


namespace find_f_of_3_l79_7912

-- Define the function f and its properties
variable {f : ℝ → ℝ}

-- Define the properties given in the problem
axiom f_mono_increasing : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_of_f_minus_exp : ∀ x : ℝ, f (f x - 2^x) = 3

-- The main theorem to prove
theorem find_f_of_3 : f 3 = 9 := 
sorry

end find_f_of_3_l79_7912


namespace significant_figures_and_precision_l79_7949

-- Definition of the function to count significant figures
def significant_figures (n : Float) : Nat :=
  -- Implementation of a function that counts significant figures
  -- Skipping actual implementation, assuming it is correct.
  sorry

-- Definition of the function to determine precision
def precision (n : Float) : String :=
  -- Implementation of a function that returns the precision
  -- Skipping actual implementation, assuming it is correct.
  sorry

-- The target number
def num := 0.03020

-- The properties of the number 0.03020
theorem significant_figures_and_precision :
  significant_figures num = 4 ∧ precision num = "ten-thousandth" :=
by
  sorry

end significant_figures_and_precision_l79_7949


namespace percent_of_male_literate_l79_7947

noncomputable def female_percentage : ℝ := 0.6
noncomputable def total_employees : ℕ := 1500
noncomputable def literate_percentage : ℝ := 0.62
noncomputable def literate_female_employees : ℕ := 630

theorem percent_of_male_literate :
  let total_females := (female_percentage * total_employees)
  let total_males := total_employees - total_females
  let total_literate := literate_percentage * total_employees
  let literate_male_employees := total_literate - literate_female_employees
  let male_literate_percentage := (literate_male_employees / total_males) * 100
  male_literate_percentage = 50 := by
  sorry

end percent_of_male_literate_l79_7947


namespace quadratic_real_roots_condition_l79_7935

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 * x - 1 = 0) ↔ (a ≥ -4 ∧ a ≠ 0) :=
sorry

end quadratic_real_roots_condition_l79_7935


namespace train_speed_l79_7940

theorem train_speed (length time_speed: ℝ) (h1 : length = 400) (h2 : time_speed = 16) : length / time_speed = 25 := 
by
    sorry

end train_speed_l79_7940


namespace largest_number_l79_7954

theorem largest_number (n : ℕ) (digits : List ℕ) (h_digits : ∀ d ∈ digits, d = 5 ∨ d = 3 ∨ d = 1) (h_sum : digits.sum = 15) : n = 555 :=
by
  sorry

end largest_number_l79_7954


namespace haley_marbles_l79_7920

theorem haley_marbles (boys : ℕ) (marbles_per_boy : ℕ) (total_marbles : ℕ) 
  (h1 : boys = 11) (h2 : marbles_per_boy = 9) : total_marbles = 99 :=
by
  sorry

end haley_marbles_l79_7920


namespace triangle_third_side_max_length_l79_7941

theorem triangle_third_side_max_length (a b : ℕ) (ha : a = 5) (hb : b = 11) : ∃ (c : ℕ), c = 15 ∧ (a + c > b ∧ b + c > a ∧ a + b > c) :=
by 
  sorry

end triangle_third_side_max_length_l79_7941


namespace sum_invariant_under_permutation_l79_7982

theorem sum_invariant_under_permutation (b : List ℝ) (σ : List ℕ) (hσ : σ.Perm (List.range b.length)) :
  (List.sum b) = (List.sum (σ.map (b.get!))) := by
  sorry

end sum_invariant_under_permutation_l79_7982


namespace cost_each_side_is_56_l79_7908

-- Define the total cost and number of sides
def total_cost : ℕ := 224
def number_of_sides : ℕ := 4

-- Define the cost per side as the division of total cost by number of sides
def cost_per_side : ℕ := total_cost / number_of_sides

-- The theorem stating the cost per side is 56
theorem cost_each_side_is_56 : cost_per_side = 56 :=
by
  -- Proof would go here
  sorry

end cost_each_side_is_56_l79_7908


namespace combined_weight_of_Leo_and_Kendra_l79_7921

theorem combined_weight_of_Leo_and_Kendra :
  ∃ (K : ℝ), (92 + K = 160) ∧ (102 = 1.5 * K) :=
by
  sorry

end combined_weight_of_Leo_and_Kendra_l79_7921


namespace simplest_common_denominator_of_fractions_l79_7992

noncomputable def simplestCommonDenominator (a b : ℕ) (x y : ℕ) : ℕ := 6 * (x ^ 2) * (y ^ 3)

theorem simplest_common_denominator_of_fractions :
  simplestCommonDenominator 2 6 x y = 6 * x^2 * y^3 :=
by
  sorry

end simplest_common_denominator_of_fractions_l79_7992


namespace total_distance_traveled_l79_7905

noncomputable def travel_distance : ℝ :=
  1280 * Real.sqrt 2 + 640 * Real.sqrt (2 + Real.sqrt 2) + 640

theorem total_distance_traveled :
  let n := 8
  let r := 40
  let theta := 2 * Real.pi / n
  let d_2arcs := 2 * r * Real.sin (theta)
  let d_3arcs := r * (2 + Real.sqrt (2))
  let d_4arcs := 2 * r
  (8 * (4 * d_2arcs + 2 * d_3arcs + d_4arcs)) = travel_distance := by
  sorry

end total_distance_traveled_l79_7905


namespace susan_annual_percentage_increase_l79_7996

theorem susan_annual_percentage_increase :
  let initial_jerry := 14400
  let initial_susan := 6250
  let jerry_first_year := initial_jerry * (6 / 5 : ℝ)
  let jerry_second_year := jerry_first_year * (9 / 10 : ℝ)
  let jerry_third_year := jerry_second_year * (6 / 5 : ℝ)
  jerry_third_year = 18662.40 →
  (initial_susan : ℝ) * (1 + r)^3 = 18662.40 →
  r = 0.44 :=
by {
  sorry
}

end susan_annual_percentage_increase_l79_7996


namespace point_equidistant_l79_7930

def A : ℝ × ℝ × ℝ := (10, 0, 0)
def B : ℝ × ℝ × ℝ := (0, -6, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 8)
def D : ℝ × ℝ × ℝ := (0, 0, 0)
def P : ℝ × ℝ × ℝ := (5, -3, 4)

theorem point_equidistant : dist A P = dist B P ∧ dist B P = dist C P ∧ dist C P = dist D P :=
by
  sorry

end point_equidistant_l79_7930


namespace Mary_paid_on_Tuesday_l79_7999

theorem Mary_paid_on_Tuesday 
  (credit_limit total_spent paid_on_thursday remaining_payment paid_on_tuesday : ℝ)
  (h1 : credit_limit = 100)
  (h2 : total_spent = credit_limit)
  (h3 : paid_on_thursday = 23)
  (h4 : remaining_payment = 62)
  (h5 : total_spent = paid_on_thursday + remaining_payment + paid_on_tuesday) :
  paid_on_tuesday = 15 :=
sorry

end Mary_paid_on_Tuesday_l79_7999


namespace solve_fraction_l79_7938

variables (w x y : ℝ)

-- Conditions
def condition1 := w / x = 2 / 3
def condition2 := w / y = 6 / 15

-- Statement
theorem solve_fraction (h1 : condition1 w x) (h2 : condition2 w y) : (x + y) / y = 8 / 5 :=
sorry

end solve_fraction_l79_7938


namespace tangent_slope_at_one_l79_7952

def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem tangent_slope_at_one : f' 1 = 2 := by
  sorry

end tangent_slope_at_one_l79_7952


namespace total_quantity_before_adding_water_l79_7934

variable (x : ℚ)
variable (milk water : ℚ)
variable (added_water : ℚ)

-- Mixture contains milk and water in the ratio 3:2
def initial_ratio (milk water : ℚ) : Prop := milk / water = 3 / 2

-- Adding 10 liters of water
def added_amount : ℚ := 10

-- New ratio of milk to water becomes 2:3 after adding 10 liters of water
def new_ratio (milk water : ℚ) (added_water : ℚ) : Prop :=
  milk / (water + added_water) = 2 / 3

theorem total_quantity_before_adding_water
  (h_ratio : initial_ratio milk water)
  (h_added : added_water = 10)
  (h_new_ratio : new_ratio milk water added_water) :
  milk + water = 20 :=
by
  sorry

end total_quantity_before_adding_water_l79_7934


namespace smallest_number_is_neg1_l79_7904

-- Defining the list of numbers
def numbers := [0, -1, 1, 2]

-- Theorem statement to prove that the smallest number in the list is -1
theorem smallest_number_is_neg1 :
  ∀ x ∈ numbers, x ≥ -1 := 
sorry

end smallest_number_is_neg1_l79_7904


namespace hash_fn_triple_40_l79_7927

def hash_fn (N : ℝ) : ℝ := 0.6 * N + 2

theorem hash_fn_triple_40 : hash_fn (hash_fn (hash_fn 40)) = 12.56 := by
  sorry

end hash_fn_triple_40_l79_7927


namespace additional_people_needed_l79_7946

-- Definitions corresponding to the given conditions
def person_hours (n : ℕ) (t : ℕ) : ℕ := n * t
def initial_people : ℕ := 8
def initial_time : ℕ := 10
def total_person_hours := person_hours initial_people initial_time

-- Lean statement of the problem
theorem additional_people_needed (new_time : ℕ) (new_people : ℕ) : 
  new_time = 5 → person_hours new_people new_time = total_person_hours → new_people - initial_people = 8 :=
by
  intro h1 h2
  sorry

end additional_people_needed_l79_7946


namespace maximum_value_of_chords_l79_7969

noncomputable def max_sum_of_chords (r : ℝ) (h1 : r = 5) 
(h2 : ∃ PA PB PC : ℝ, PA = 2 * PB ∧ PA^2 + PB^2 + PC^2 = (2 * r)^2) : ℝ := 
  6 * Real.sqrt 10

theorem maximum_value_of_chords (P : Point) (r : ℝ) (h1 : r = 5) 
(h2 : ∃ PA PB PC : ℝ, PA = 2 * PB ∧ PA^2 + PB^2 + PC^2 = (2 * r)^2) : 
  PA + PB + PC ≤ 6 * Real.sqrt 10 :=
by
  sorry

end maximum_value_of_chords_l79_7969


namespace stockings_total_cost_l79_7928

-- Defining the conditions
def total_stockings : ℕ := 9
def original_price_per_stocking : ℝ := 20
def discount_rate : ℝ := 0.10
def monogramming_cost_per_stocking : ℝ := 5

-- Calculate the total cost of stockings
theorem stockings_total_cost :
  total_stockings * ((original_price_per_stocking * (1 - discount_rate)) + monogramming_cost_per_stocking) = 207 := 
by
  sorry

end stockings_total_cost_l79_7928


namespace find_m_l79_7975

theorem find_m (m : ℝ) (x : ℝ) (y : ℝ) (h_eq_parabola : y = m * x^2)
  (h_directrix : y = 1 / 8) : m = -2 :=
by
  sorry

end find_m_l79_7975


namespace neg_p_equivalent_to_forall_x2_ge_1_l79_7957

open Classical

variable {x : ℝ}

-- Definition of the original proposition p
def p : Prop := ∃ (x : ℝ), x^2 < 1

-- The negation of the proposition p
def not_p : Prop := ∀ (x : ℝ), x^2 ≥ 1

-- The theorem stating the equivalence
theorem neg_p_equivalent_to_forall_x2_ge_1 : ¬ p ↔ not_p := by
  sorry

end neg_p_equivalent_to_forall_x2_ge_1_l79_7957


namespace water_tank_capacity_l79_7976

theorem water_tank_capacity (C : ℝ) (h : 0.70 * C - 0.40 * C = 36) : C = 120 :=
sorry

end water_tank_capacity_l79_7976


namespace x_n_squared_leq_2007_l79_7919

def recurrence (x y : ℕ → ℝ) : Prop :=
  x 0 = 1 ∧ y 0 = 2007 ∧
  ∀ n, x (n + 1) = x n - (x n * y n + x (n + 1) * y (n + 1) - 2) * (y n + y (n + 1)) ∧
       y (n + 1) = y n - (x n * y n + x (n + 1) * y (n + 1) - 2) * (x n + x (n + 1))

theorem x_n_squared_leq_2007 (x y : ℕ → ℝ) (h : recurrence x y) : ∀ n, x n ^ 2 ≤ 2007 :=
by sorry

end x_n_squared_leq_2007_l79_7919


namespace max_intersections_pentagon_triangle_max_intersections_pentagon_quadrilateral_l79_7961

-- Define the pentagon and various other polygons
inductive PolygonType
| pentagon
| triangle
| quadrilateral

-- Define a function that calculates the maximum number of intersections
def max_intersections (K L : PolygonType) : ℕ :=
  match K, L with
  | PolygonType.pentagon, PolygonType.triangle => 10
  | PolygonType.pentagon, PolygonType.quadrilateral => 16
  | _, _ => 0  -- We only care about the cases specified in our problem

-- Theorem a): When L is a triangle, the intersections should be 10
theorem max_intersections_pentagon_triangle : max_intersections PolygonType.pentagon PolygonType.triangle = 10 :=
  by 
  -- provide proof here, but currently it is skipped with sorry
  sorry

-- Theorem b): When L is a quadrilateral, the intersections should be 16
theorem max_intersections_pentagon_quadrilateral : max_intersections PolygonType.pentagon PolygonType.quadrilateral = 16 :=
  by
  -- provide proof here, but currently it is skipped with sorry
  sorry

end max_intersections_pentagon_triangle_max_intersections_pentagon_quadrilateral_l79_7961


namespace david_completion_time_l79_7923

theorem david_completion_time :
  (∃ D : ℕ, ∀ t : ℕ, 6 * (1 / D) + 3 * ((1 / D) + (1 / t)) = 1 -> D = 12) :=
sorry

end david_completion_time_l79_7923


namespace distinct_zeros_abs_minus_one_l79_7925

theorem distinct_zeros_abs_minus_one : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (|x₁| - 1 = 0) ∧ (|x₂| - 1 = 0) := 
by
  sorry

end distinct_zeros_abs_minus_one_l79_7925


namespace probability_not_pulling_prize_twice_l79_7991

theorem probability_not_pulling_prize_twice
  (favorable : ℕ)
  (unfavorable : ℕ)
  (total : ℕ := favorable + unfavorable)
  (P_prize : ℚ := favorable / total)
  (P_not_prize : ℚ := 1 - P_prize)
  (P_not_prize_twice : ℚ := P_not_prize * P_not_prize) :
  P_not_prize_twice = 36 / 121 :=
by
  have favorable : ℕ := 5
  have unfavorable : ℕ := 6
  have total : ℕ := favorable + unfavorable
  have P_prize : ℚ := favorable / total
  have P_not_prize : ℚ := 1 - P_prize
  have P_not_prize_twice : ℚ := P_not_prize * P_not_prize
  sorry

end probability_not_pulling_prize_twice_l79_7991


namespace maximum_matches_l79_7902

theorem maximum_matches (A B C : ℕ) (h1 : A > B) (h2 : B > C) 
    (h3 : A ≥ B + 10) (h4 : B ≥ C + 10) (h5 : B + C > A) : 
    A + B + C - 1 ≤ 62 :=
sorry

end maximum_matches_l79_7902


namespace compute_gf3_l79_7987

def f (x : ℝ) : ℝ := x^3 - 3
def g (x : ℝ) : ℝ := 2 * x^2 - x + 4

theorem compute_gf3 : g (f 3) = 1132 := 
by 
  sorry

end compute_gf3_l79_7987


namespace common_root_and_param_l79_7971

theorem common_root_and_param :
  ∀ (x : ℤ) (P p : ℚ),
    (P = -((x^2 - x - 2) / (x - 1)) ∧ x ≠ 1) →
    (p = -((x^2 + 2*x - 1) / (x + 2)) ∧ x ≠ -2) →
    (-x + (2 / (x - 1)) = -x + (1 / (x + 2))) →
    x = -5 ∧ p = 14 / 3 :=
by
  intros x P p hP hp hroot
  sorry

end common_root_and_param_l79_7971


namespace general_formula_a_n_T_n_greater_than_S_n_l79_7977

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l79_7977


namespace thomas_total_training_hours_l79_7990

-- Define the conditions from the problem statement.
def training_hours_first_15_days : ℕ := 15 * 5
def training_hours_next_15_days : ℕ := (15 - 3) * (4 + 3)
def training_hours_next_12_days : ℕ := (12 - 2) * (4 + 3)

-- Prove that the total training hours equals 229.
theorem thomas_total_training_hours : 
  training_hours_first_15_days + training_hours_next_15_days + training_hours_next_12_days = 229 :=
by
  -- conditions as defined
  let t1 := 15 * 5
  let t2 := (15 - 3) * (4 + 3)
  let t3 := (12 - 2) * (4 + 3)
  show t1 + t2 + t3 = 229
  sorry

end thomas_total_training_hours_l79_7990


namespace difference_between_greatest_and_smallest_S_l79_7917

-- Conditions
def num_students := 47
def rows := 6
def columns := 8

-- The definition of position value calculation
def position_value (i j m n : ℕ) := i - m + (j - n)

-- The definition of S
def S (initial_empty final_empty : (ℕ × ℕ)) : ℤ :=
  let (i_empty, j_empty) := initial_empty
  let (i'_empty, j'_empty) := final_empty
  (i'_empty + j'_empty) - (i_empty + j_empty)

-- Main statement
theorem difference_between_greatest_and_smallest_S :
  let max_S := S (1, 1) (6, 8)
  let min_S := S (6, 8) (1, 1)
  max_S - min_S = 24 :=
sorry

end difference_between_greatest_and_smallest_S_l79_7917


namespace movie_ticket_final_price_l79_7942

noncomputable def final_ticket_price (initial_price : ℝ) : ℝ :=
  let price_year_1 := initial_price * 1.12
  let price_year_2 := price_year_1 * 0.95
  let price_year_3 := price_year_2 * 1.08
  let price_year_4 := price_year_3 * 0.96
  let price_year_5 := price_year_4 * 1.06
  let price_after_tax := price_year_5 * 1.07
  let final_price := price_after_tax * 0.90
  final_price

theorem movie_ticket_final_price :
  final_ticket_price 100 = 112.61 := by
  sorry

end movie_ticket_final_price_l79_7942


namespace reciprocal_sum_is_1_implies_at_least_one_is_2_l79_7939

-- Lean statement for the problem
theorem reciprocal_sum_is_1_implies_at_least_one_is_2 (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_sum : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1) : 
  a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 := 
sorry

end reciprocal_sum_is_1_implies_at_least_one_is_2_l79_7939
