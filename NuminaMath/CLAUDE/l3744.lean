import Mathlib

namespace symmetric_abs_function_l3744_374410

/-- A function f is symmetric about a point c if f(c+x) = f(c-x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetric_abs_function (m n : ℝ) :
  SymmetricAbout (fun x ↦ |x + m| + |n * x + 1|) 2 → m + n = -4 := by
  sorry

end symmetric_abs_function_l3744_374410


namespace tanika_cracker_sales_l3744_374435

theorem tanika_cracker_sales (saturday_sales : ℕ) : 
  saturday_sales = 60 → 
  (saturday_sales + (saturday_sales + saturday_sales / 2)) = 150 := by
sorry

end tanika_cracker_sales_l3744_374435


namespace complex_number_theorem_l3744_374422

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_theorem (z : ℂ) 
  (h1 : is_purely_imaginary z) 
  (h2 : is_purely_imaginary ((z + 2)^2 + 5)) : 
  z = Complex.I * 3 ∨ z = Complex.I * (-3) := by
  sorry

end complex_number_theorem_l3744_374422


namespace tv_power_consumption_l3744_374449

/-- Given a TV that runs for 4 hours a day, with electricity costing 14 cents per kWh,
    and the TV costing 49 cents to run for a week, prove that the TV uses 125 watts of electricity per hour. -/
theorem tv_power_consumption (hours_per_day : ℝ) (cost_per_kwh : ℝ) (weekly_cost : ℝ) :
  hours_per_day = 4 →
  cost_per_kwh = 0.14 →
  weekly_cost = 0.49 →
  ∃ (watts : ℝ), watts = 125 ∧ 
    (weekly_cost / cost_per_kwh) / (hours_per_day * 7) * 1000 = watts :=
by sorry

end tv_power_consumption_l3744_374449


namespace toy_factory_wage_calculation_l3744_374453

/-- Toy factory production and wage calculation -/
theorem toy_factory_wage_calculation 
  (planned_weekly_production : ℕ)
  (average_daily_production : ℕ)
  (deviations : List ℤ)
  (base_wage_per_toy : ℕ)
  (bonus_per_extra_toy : ℕ)
  (deduction_per_missing_toy : ℕ)
  (h1 : planned_weekly_production = 700)
  (h2 : average_daily_production = 100)
  (h3 : deviations = [5, -2, -4, 13, -6, 6, -3])
  (h4 : base_wage_per_toy = 20)
  (h5 : bonus_per_extra_toy = 5)
  (h6 : deduction_per_missing_toy = 4)
  : (planned_weekly_production + deviations.sum) * base_wage_per_toy + 
    (deviations.sum * (base_wage_per_toy + bonus_per_extra_toy)) = 14225 := by
  sorry

end toy_factory_wage_calculation_l3744_374453


namespace integer_solutions_count_l3744_374462

theorem integer_solutions_count : ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ |7*x - 4| ≤ 14) ∧ Finset.card S = 4 := by
  sorry

end integer_solutions_count_l3744_374462


namespace train_length_calculation_l3744_374488

/-- Represents a train with its length and the time it takes to cross two platforms -/
structure Train where
  length : ℝ
  time_platform1 : ℝ
  time_platform2 : ℝ

/-- The length of the first platform in meters -/
def platform1_length : ℝ := 120

/-- The length of the second platform in meters -/
def platform2_length : ℝ := 250

/-- Theorem stating that a train crossing two platforms of given lengths in specific times has a specific length -/
theorem train_length_calculation (t : Train) 
  (h1 : t.time_platform1 = 15) 
  (h2 : t.time_platform2 = 20) : 
  t.length = 270 := by
  sorry

end train_length_calculation_l3744_374488


namespace matrix_determinant_equality_l3744_374457

theorem matrix_determinant_equality 
  (A B : Matrix (Fin 4) (Fin 4) ℝ) 
  (h1 : A * B = B * A) 
  (h2 : Matrix.det (A^2 + A*B + B^2) = 0) : 
  Matrix.det (A + B) + 3 * Matrix.det (A - B) = 6 * Matrix.det A + 6 * Matrix.det B := by
  sorry

end matrix_determinant_equality_l3744_374457


namespace sector_area_from_arc_length_and_angle_l3744_374460

/-- Given an arc length of 4 cm corresponding to a central angle of 2 radians,
    the area of the sector formed by this central angle is 4 cm^2. -/
theorem sector_area_from_arc_length_and_angle (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = 2) :
  let r := s / θ
  (1 / 2) * r^2 * θ = 4 := by sorry

end sector_area_from_arc_length_and_angle_l3744_374460


namespace compare_quadratic_expressions_compare_fraction_sum_and_sum_l3744_374498

-- Statement 1
theorem compare_quadratic_expressions (m : ℝ) :
  3 * m^2 - m + 1 > 2 * m^2 + m - 3 := by
  sorry

-- Statement 2
theorem compare_fraction_sum_and_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 / b + b^2 / a ≥ a + b := by
  sorry

end compare_quadratic_expressions_compare_fraction_sum_and_sum_l3744_374498


namespace base_b_problem_l3744_374412

theorem base_b_problem : ∃! (b : ℕ), b > 1 ∧ (2 * b + 9)^2 = 7 * b^2 + 3 * b + 4 := by
  sorry

end base_b_problem_l3744_374412


namespace remainder_of_123456789012_mod_252_l3744_374408

theorem remainder_of_123456789012_mod_252 : 123456789012 % 252 = 24 := by
  sorry

end remainder_of_123456789012_mod_252_l3744_374408


namespace stan_pays_magician_l3744_374401

/-- The total amount Stan pays the magician -/
def total_payment (hourly_rate : ℕ) (hours_per_day : ℕ) (weeks : ℕ) : ℕ :=
  hourly_rate * hours_per_day * (weeks * 7)

/-- Proof that Stan pays the magician $2520 -/
theorem stan_pays_magician :
  total_payment 60 3 2 = 2520 := by
  sorry

end stan_pays_magician_l3744_374401


namespace meetings_percentage_of_workday_l3744_374459

def workday_hours : ℝ := 10
def first_meeting_minutes : ℝ := 30
def second_meeting_minutes : ℝ := 3 * first_meeting_minutes
def third_meeting_minutes : ℝ := 2 * second_meeting_minutes

def total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes + third_meeting_minutes
def workday_minutes : ℝ := workday_hours * 60

theorem meetings_percentage_of_workday :
  (total_meeting_minutes / workday_minutes) * 100 = 50 := by sorry

end meetings_percentage_of_workday_l3744_374459


namespace unique_solution_exponential_equation_l3744_374406

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (10 : ℝ)^(2*x) * (1000 : ℝ)^x = (10 : ℝ)^15 :=
by
  sorry

end unique_solution_exponential_equation_l3744_374406


namespace choir_size_l3744_374448

theorem choir_size :
  ∀ X : ℕ,
  (X / 2 : ℚ) - (X / 6 : ℚ) = 10 →
  X = 30 :=
by
  sorry

#check choir_size

end choir_size_l3744_374448


namespace bob_sandwich_options_l3744_374418

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents whether turkey is available. -/
def has_turkey : Prop := True

/-- Represents whether roast beef is available. -/
def has_roast_beef : Prop := True

/-- Represents whether Swiss cheese is available. -/
def has_swiss_cheese : Prop := True

/-- Represents whether rye bread is available. -/
def has_rye_bread : Prop := True

/-- Represents the number of sandwiches with turkey and Swiss cheese. -/
def turkey_swiss_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and roast beef. -/
def rye_roast_beef_combos : ℕ := num_cheeses

/-- Theorem stating the number of different sandwiches Bob could order. -/
theorem bob_sandwich_options : 
  num_breads * num_meats * num_cheeses - turkey_swiss_combos - rye_roast_beef_combos = 199 :=
sorry

end bob_sandwich_options_l3744_374418


namespace perpendicular_length_between_l3744_374421

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the relations and functions
variable (on_line : Point → Line → Prop)
variable (between : Point → Point → Point → Prop)
variable (perpendicular : Point → Point → Line → Prop)
variable (length : Point → Point → ℝ)

-- State the theorem
theorem perpendicular_length_between
  (a b : Line)
  (A₁ A₂ A₃ B₁ B₂ B₃ : Point)
  (h1 : on_line A₁ a)
  (h2 : on_line A₂ a)
  (h3 : on_line A₃ a)
  (h4 : between A₁ A₂ A₃)
  (h5 : perpendicular A₁ B₁ b)
  (h6 : perpendicular A₂ B₂ b)
  (h7 : perpendicular A₃ B₃ b) :
  (length A₁ B₁ ≤ length A₂ B₂ ∧ length A₂ B₂ ≤ length A₃ B₃) ∨
  (length A₃ B₃ ≤ length A₂ B₂ ∧ length A₂ B₂ ≤ length A₁ B₁) :=
sorry

end perpendicular_length_between_l3744_374421


namespace youngest_child_age_l3744_374490

/-- Represents a family with its members and ages -/
structure Family where
  members : Nat
  total_age : Nat

/-- The problem setup -/
def initial_family : Family := { members := 4, total_age := 96 }

/-- The current state of the family -/
def current_family : Family := { members := 6, total_age := 144 }

/-- The time passed since the initial state -/
def years_passed : Nat := 10

/-- The age difference between the two new children -/
def age_difference : Nat := 2

/-- Theorem stating that the youngest child's age is 3 years -/
theorem youngest_child_age :
  let youngest_age := (current_family.total_age - (initial_family.total_age + years_passed * initial_family.members)) / 2
  youngest_age = 3 := by sorry

end youngest_child_age_l3744_374490


namespace better_fit_for_lower_rss_l3744_374472

/-- Represents a model with its residual sum of squares -/
structure Model where
  rss : ℝ

/-- Definition of a better fit model -/
def better_fit (m1 m2 : Model) : Prop := m1.rss < m2.rss

theorem better_fit_for_lower_rss (model1 model2 : Model) 
  (h1 : model1.rss = 152.6) 
  (h2 : model2.rss = 159.8) : 
  better_fit model1 model2 := by
  sorry

#check better_fit_for_lower_rss

end better_fit_for_lower_rss_l3744_374472


namespace factorial_equivalences_l3744_374432

/-- The number of arrangements of n objects taken k at a time -/
def A (n k : ℕ) : ℕ := sorry

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := sorry

theorem factorial_equivalences (n : ℕ) : 
  (A n (n - 1) = factorial n) ∧ 
  ((1 / (n + 1 : ℚ)) * A (n + 1) (n + 1) = factorial n) := by sorry

end factorial_equivalences_l3744_374432


namespace bingbing_correct_qianqian_incorrect_l3744_374416

-- Define the basic parameters of the problem
def downstream_time : ℝ := 2
def upstream_time : ℝ := 2.5
def water_speed : ℝ := 3

-- Define Bingbing's equation
def bingbing_equation (x : ℝ) : Prop :=
  2 * (x + water_speed) = upstream_time * (x - water_speed)

-- Define Qianqian's equation
def qianqian_equation (x : ℝ) : Prop :=
  x / downstream_time - x / upstream_time = water_speed * downstream_time

-- Theorem stating that Bingbing's equation correctly models the problem
theorem bingbing_correct :
  ∃ (x : ℝ), bingbing_equation x ∧ x > 0 ∧ 
  (x * downstream_time = x * upstream_time) :=
sorry

-- Theorem stating that Qianqian's equation does not correctly model the problem
theorem qianqian_incorrect :
  ¬(∃ (x : ℝ), qianqian_equation x ∧ 
  (x * downstream_time = x * upstream_time ∨ x > 0)) :=
sorry

end bingbing_correct_qianqian_incorrect_l3744_374416


namespace solution_set_f_geq_5_range_of_a_l3744_374400

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Part I: Solution set of f(x) ≥ 5
theorem solution_set_f_geq_5 :
  {x : ℝ | f x ≥ 5} = Set.Iic (-3) ∪ Set.Ici 2 :=
sorry

-- Part II: Range of a for which f(x) > a^2 - 2a holds for all x
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x > a^2 - 2*a) ↔ a ∈ Set.Ioo (-1) 3 :=
sorry

end solution_set_f_geq_5_range_of_a_l3744_374400


namespace range_of_a_l3744_374477

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then a^x else (4 - a/2)*x + 2

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Icc 4 8 ∧ a ≠ 8 :=
sorry

end range_of_a_l3744_374477


namespace substance_mass_proof_l3744_374413

/-- The volume of 1 gram of the substance in cubic centimeters -/
def volume_per_gram : ℝ := 1.3333333333333335

/-- The number of cubic centimeters in 1 cubic meter -/
def cm3_per_m3 : ℝ := 1000000

/-- The number of grams in 1 kilogram -/
def grams_per_kg : ℝ := 1000

/-- The mass of 1 cubic meter of the substance in kilograms -/
def mass_per_m3 : ℝ := 750

theorem substance_mass_proof :
  mass_per_m3 = cm3_per_m3 / (grams_per_kg * volume_per_gram) := by
  sorry

end substance_mass_proof_l3744_374413


namespace pool_water_after_20_days_l3744_374470

/-- Calculates the remaining water in a swimming pool after a given number of days -/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (leak_rate : ℝ) (days : ℝ) : ℝ :=
  initial_amount - (evaporation_rate + leak_rate) * days

/-- Theorem stating the remaining water in the pool after 20 days -/
theorem pool_water_after_20_days :
  remaining_water 500 1.5 0.8 20 = 454 := by
  sorry

#eval remaining_water 500 1.5 0.8 20

end pool_water_after_20_days_l3744_374470


namespace paint_distribution_321_60_l3744_374433

/-- Given a paint mixture with a ratio of red:white:blue and a total number of cans,
    calculate the number of cans for each color. -/
def paint_distribution (red white blue total : ℕ) : ℕ × ℕ × ℕ :=
  let sum := red + white + blue
  let red_cans := total * red / sum
  let white_cans := total * white / sum
  let blue_cans := total * blue / sum
  (red_cans, white_cans, blue_cans)

/-- Prove that for a 3:2:1 ratio and 60 total cans, we get 30 red, 20 white, and 10 blue cans. -/
theorem paint_distribution_321_60 :
  paint_distribution 3 2 1 60 = (30, 20, 10) := by
  sorry

end paint_distribution_321_60_l3744_374433


namespace mrs_hilt_apple_pies_mrs_hilt_apple_pies_proof_l3744_374471

theorem mrs_hilt_apple_pies : ℝ → Prop :=
  fun apple_pies =>
    let pecan_pies : ℝ := 16.0
    let total_pies : ℝ := pecan_pies + apple_pies
    let new_total : ℝ := 150.0
    (5.0 * total_pies = new_total) → apple_pies = 14.0

-- The proof is omitted
theorem mrs_hilt_apple_pies_proof : mrs_hilt_apple_pies 14.0 := by
  sorry

end mrs_hilt_apple_pies_mrs_hilt_apple_pies_proof_l3744_374471


namespace limit_problem_l3744_374411

open Real
open Function

/-- The limit of the given function as x approaches 2 is -8ln(2)/5 -/
theorem limit_problem : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
  0 < |x - 2| ∧ |x - 2| < δ → 
    |(1 - 2^(4 - x^2)) / (2 * (sqrt (2*x) - sqrt (3*x^2 - 5*x + 2))) + 8*log 2 / 5| < ε :=
by
  sorry

end limit_problem_l3744_374411


namespace new_arithmetic_mean_l3744_374483

def original_count : ℕ := 60
def original_mean : ℝ := 45
def removed_numbers : List ℝ := [48, 58, 62]

theorem new_arithmetic_mean :
  let original_sum : ℝ := original_count * original_mean
  let removed_sum : ℝ := removed_numbers.sum
  let new_count : ℕ := original_count - removed_numbers.length
  let new_sum : ℝ := original_sum - removed_sum
  new_sum / new_count = 44.42 := by sorry

end new_arithmetic_mean_l3744_374483


namespace prob_two_non_defective_pens_l3744_374495

/-- The probability of selecting two non-defective pens from a box of 12 pens, where 6 are defective -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 12)
  (h2 : defective_pens = 6) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 5 / 22 := by
  sorry

#check prob_two_non_defective_pens

end prob_two_non_defective_pens_l3744_374495


namespace power_multiplication_l3744_374478

theorem power_multiplication (a : ℝ) : -a^4 * a^3 = -a^7 := by sorry

end power_multiplication_l3744_374478


namespace solution_set_part1_range_of_a_part2_l3744_374473

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x > -a) ↔ a > -3/2 :=
sorry

end solution_set_part1_range_of_a_part2_l3744_374473


namespace unique_a_for_system_solution_l3744_374404

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  2^(b*x) + (a+1)*b*y^2 = a^2 ∧ (a-1)*x^3 + y^3 = 1

-- State the theorem
theorem unique_a_for_system_solution :
  ∃! a : ℝ, ∀ b : ℝ, ∃ x y : ℝ, system a b x y ∧ a = -1 :=
sorry

end unique_a_for_system_solution_l3744_374404


namespace equation_solution_l3744_374431

theorem equation_solution : 
  ∀ x : ℝ, x * (2 * x - 1) = 4 * x - 2 ↔ x = 2 ∨ x = 1/2 := by
sorry

end equation_solution_l3744_374431


namespace monotonic_h_implies_a_leq_neg_one_l3744_374451

/-- Given functions f and g, prove that if h is monotonically increasing on [1,4],
    then a ≤ -1 -/
theorem monotonic_h_implies_a_leq_neg_one (a : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ Real.log x
  let g : ℝ → ℝ := λ x ↦ (1/2) * a * x^2 + 2*x
  let h : ℝ → ℝ := λ x ↦ f x - g x
  (∀ x ∈ Set.Icc 1 4, Monotone h) →
  a ≤ -1 := by
sorry

end monotonic_h_implies_a_leq_neg_one_l3744_374451


namespace tony_bought_seven_swords_l3744_374481

/-- Represents the purchase of toys by Tony -/
structure ToyPurchase where
  lego_cost : ℕ
  sword_cost : ℕ
  dough_cost : ℕ
  lego_sets : ℕ
  doughs : ℕ
  total_paid : ℕ

/-- Calculates the number of toy swords bought given a ToyPurchase -/
def calculate_swords (purchase : ToyPurchase) : ℕ :=
  let lego_total := purchase.lego_cost * purchase.lego_sets
  let dough_total := purchase.dough_cost * purchase.doughs
  let sword_total := purchase.total_paid - lego_total - dough_total
  sword_total / purchase.sword_cost

/-- Theorem stating that Tony bought 7 toy swords -/
theorem tony_bought_seven_swords : 
  ∀ (purchase : ToyPurchase), 
    purchase.lego_cost = 250 →
    purchase.sword_cost = 120 →
    purchase.dough_cost = 35 →
    purchase.lego_sets = 3 →
    purchase.doughs = 10 →
    purchase.total_paid = 1940 →
    calculate_swords purchase = 7 := by
  sorry

end tony_bought_seven_swords_l3744_374481


namespace meaningful_fraction_l3744_374452

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end meaningful_fraction_l3744_374452


namespace special_sequence_a11_l3744_374476

/-- A sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ+ → ℤ) : Prop :=
  (∀ p q : ℕ+, a (p + q) = a p + a q) ∧ (a 2 = -6)

/-- The theorem statement -/
theorem special_sequence_a11 (a : ℕ+ → ℤ) (h : SpecialSequence a) : a 11 = -33 := by
  sorry

end special_sequence_a11_l3744_374476


namespace rectangular_field_fencing_l3744_374437

theorem rectangular_field_fencing (area : ℝ) (fencing : ℝ) :
  area = 680 ∧ fencing = 74 →
  ∃ (length width : ℝ),
    length > 0 ∧ width > 0 ∧
    area = length * width ∧
    fencing = 2 * width + length ∧
    length = 40 := by
  sorry

end rectangular_field_fencing_l3744_374437


namespace trajectory_is_parabola_l3744_374464

/-- The set of points equidistant from a fixed point and a line forms a parabola -/
theorem trajectory_is_parabola (x y : ℝ) : 
  (∃ (C : ℝ × ℝ), C.1 = x ∧ C.2 = y ∧ 
    (C.1^2 + (C.2 - 3)^2)^(1/2) = |C.2 + 3|) →
  ∃ (a : ℝ), y = (1 / (4 * a)) * x^2 ∧ a ≠ 0 := by
  sorry

end trajectory_is_parabola_l3744_374464


namespace shelly_thread_needed_l3744_374409

def thread_per_keychain : ℕ := 12
def friends_in_classes : ℕ := 6
def friends_in_clubs : ℕ := friends_in_classes / 2

theorem shelly_thread_needed : 
  (friends_in_classes + friends_in_clubs) * thread_per_keychain = 108 := by
  sorry

end shelly_thread_needed_l3744_374409


namespace problem_solution_l3744_374466

theorem problem_solution (a b : ℤ) 
  (h1 : 4010 * a + 4014 * b = 4020) 
  (h2 : 4012 * a + 4016 * b = 4024) : 
  a - b = 2002 := by
sorry

end problem_solution_l3744_374466


namespace set_operations_l3744_374417

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ (A ∪ C)) = {6, 8, 10}) ∧
  ((U \ A) ∩ (U \ B) = {3}) := by
  sorry

end set_operations_l3744_374417


namespace largest_coefficient_term_l3744_374441

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The general term in the binomial expansion -/
def binomialTerm (n k : ℕ) (a b : ℝ) : ℝ := 
  (binomial n k : ℝ) * (a ^ (n - k)) * (b ^ k)

/-- The coefficient of the k-th term in the expansion of (2+3x)^10 -/
def coefficientTerm (k : ℕ) : ℝ := 
  (binomial 10 k : ℝ) * (2 ^ (10 - k)) * (3 ^ k)

theorem largest_coefficient_term :
  ∃ (k : ℕ), k = 5 ∧ 
  ∀ (j : ℕ), j ≠ k → coefficientTerm k ≥ coefficientTerm j :=
sorry

end largest_coefficient_term_l3744_374441


namespace derivative_of_f_l3744_374420

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x + 9

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x := by
  sorry

end derivative_of_f_l3744_374420


namespace final_alcohol_percentage_l3744_374475

/-- Given a mixture of 15 litres with 25% alcohol, prove that after removing 2 litres of alcohol
    and adding 3 litres of water, the final alcohol percentage is approximately 10.94%. -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (alcohol_removed : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 15)
  (h2 : initial_alcohol_percentage = 0.25)
  (h3 : alcohol_removed = 2)
  (h4 : water_added = 3) :
  let initial_alcohol := initial_volume * initial_alcohol_percentage
  let remaining_alcohol := initial_alcohol - alcohol_removed
  let final_volume := initial_volume - alcohol_removed + water_added
  let final_percentage := (remaining_alcohol / final_volume) * 100
  ∃ ε > 0, abs (final_percentage - 10.94) < ε :=
sorry

end final_alcohol_percentage_l3744_374475


namespace special_function_half_l3744_374482

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

/-- The main theorem stating the value of f(1/2) -/
theorem special_function_half (f : ℝ → ℝ) (h : special_function f) : f (1/2) = -1/2 := by
  sorry

end special_function_half_l3744_374482


namespace pentagon_c_y_coordinate_l3744_374497

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculates the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Checks if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

/-- The y-coordinate of vertex C in the given pentagon is 21 -/
theorem pentagon_c_y_coordinate :
  ∀ (p : Pentagon),
    p.A = (0, 0) →
    p.B = (0, 5) →
    p.D = (5, 5) →
    p.E = (5, 0) →
    hasVerticalSymmetry p →
    pentagonArea p = 65 →
    p.C.2 = 21 := by sorry

end pentagon_c_y_coordinate_l3744_374497


namespace complex_magnitude_problem_l3744_374427

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = 2 - Complex.I) :
  Complex.abs z = (3 * Real.sqrt 2) / 2 := by
  sorry

end complex_magnitude_problem_l3744_374427


namespace expand_difference_of_squares_l3744_374405

theorem expand_difference_of_squares (a : ℝ) : (a + 2) * (2 - a) = 4 - a^2 := by
  sorry

end expand_difference_of_squares_l3744_374405


namespace min_sum_of_squares_l3744_374480

theorem min_sum_of_squares (x₁ x₂ x₃ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3*x₂ + 4*x₃ = 100) : 
  ∀ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + 3*y₂ + 4*y₃ = 100 → 
  x₁^2 + x₂^2 + x₃^2 ≤ y₁^2 + y₂^2 + y₃^2 ∧ 
  ∃ z₁ z₂ z₃ : ℝ, z₁ > 0 ∧ z₂ > 0 ∧ z₃ > 0 ∧ z₁ + 3*z₂ + 4*z₃ = 100 ∧ 
  z₁^2 + z₂^2 + z₃^2 = 5000/13 := by
sorry

end min_sum_of_squares_l3744_374480


namespace parallelogram_smaller_angle_measure_l3744_374443

/-- 
Given a parallelogram where one angle exceeds the other by 40 degrees,
prove that the measure of the smaller angle is 70 degrees.
-/
theorem parallelogram_smaller_angle_measure : 
  ∀ (smaller_angle larger_angle : ℝ),
  -- Conditions
  (smaller_angle > 0) →  -- Angle measure is positive
  (larger_angle > 0) →  -- Angle measure is positive
  (larger_angle = smaller_angle + 40) →  -- One angle exceeds the other by 40
  (smaller_angle + larger_angle = 180) →  -- Adjacent angles are supplementary
  -- Conclusion
  smaller_angle = 70 := by
sorry

end parallelogram_smaller_angle_measure_l3744_374443


namespace ceiling_neg_sqrt_frac_l3744_374465

theorem ceiling_neg_sqrt_frac : ⌈-Real.sqrt (36 / 9)⌉ = -2 := by
  sorry

end ceiling_neg_sqrt_frac_l3744_374465


namespace amount_c_l3744_374474

/-- Given four amounts a, b, c, and d satisfying certain conditions, prove that c equals 225. -/
theorem amount_c (a b c d : ℕ) : 
  a + b + c + d = 750 →
  a + c = 350 →
  b + d = 450 →
  a + d = 400 →
  c + d = 500 →
  c = 225 := by
  sorry


end amount_c_l3744_374474


namespace find_n_l3744_374425

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 8) : n = 24 := by
  sorry

end find_n_l3744_374425


namespace fraction_relation_l3744_374438

theorem fraction_relation (a b c : ℚ) 
  (h1 : a / b = 2) 
  (h2 : b / c = 4 / 3) : 
  c / a = 3 / 8 := by
sorry

end fraction_relation_l3744_374438


namespace number_of_divisors_of_90_l3744_374414

theorem number_of_divisors_of_90 : Nat.card {d : ℕ | d ∣ 90} = 12 := by
  sorry

end number_of_divisors_of_90_l3744_374414


namespace fraction_power_four_l3744_374463

theorem fraction_power_four : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by sorry

end fraction_power_four_l3744_374463


namespace stereo_system_trade_in_john_stereo_trade_in_l3744_374444

theorem stereo_system_trade_in (old_cost : ℝ) (trade_in_percentage : ℝ) 
  (new_cost : ℝ) (discount_percentage : ℝ) : ℝ :=
  let trade_in_value := old_cost * trade_in_percentage
  let discounted_new_cost := new_cost * (1 - discount_percentage)
  discounted_new_cost - trade_in_value

theorem john_stereo_trade_in :
  stereo_system_trade_in 250 0.8 600 0.25 = 250 := by
  sorry

end stereo_system_trade_in_john_stereo_trade_in_l3744_374444


namespace pattern_perimeter_is_24_l3744_374468

/-- A pattern formed by squares, triangles, and a hexagon -/
structure Pattern where
  num_squares : ℕ
  num_triangles : ℕ
  square_side_length : ℝ
  triangle_perimeter_contribution : ℕ
  square_perimeter_contribution : ℕ

/-- Calculate the perimeter of the pattern -/
def pattern_perimeter (p : Pattern) : ℝ :=
  (p.num_triangles * p.triangle_perimeter_contribution +
   p.num_squares * p.square_perimeter_contribution) * p.square_side_length

/-- The specific pattern described in the problem -/
def specific_pattern : Pattern := {
  num_squares := 6,
  num_triangles := 6,
  square_side_length := 2,
  triangle_perimeter_contribution := 2,
  square_perimeter_contribution := 2
}

theorem pattern_perimeter_is_24 :
  pattern_perimeter specific_pattern = 24 := by
  sorry

end pattern_perimeter_is_24_l3744_374468


namespace geometric_sequence_property_l3744_374428

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Theorem statement
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_condition : a 2 * a 5 < 0) :
  a 1 * a 2 * a 3 * a 4 > 0 :=
by sorry

end geometric_sequence_property_l3744_374428


namespace light_bulb_probability_l3744_374491

/-- The probability of selecting a light bulb from Factory A and it passing the quality test -/
theorem light_bulb_probability (p_A : ℝ) (p_B : ℝ) (pass_A : ℝ) (pass_B : ℝ) 
  (h1 : p_A = 0.7) 
  (h2 : p_B = 0.3) 
  (h3 : p_A + p_B = 1) 
  (h4 : pass_A = 0.95) 
  (h5 : pass_B = 0.8) :
  p_A * pass_A = 0.665 := by
  sorry

end light_bulb_probability_l3744_374491


namespace max_points_on_ellipse_l3744_374445

/-- Represents an ellipse with semi-major axis a and focal distance c -/
structure Ellipse where
  a : ℝ
  c : ℝ

/-- Represents a sequence of points on an ellipse -/
structure PointSequence where
  n : ℕ
  d : ℝ

theorem max_points_on_ellipse (e : Ellipse) (seq : PointSequence) :
  e.a - e.c = 1 →
  e.a + e.c = 3 →
  seq.d > 1/100 →
  (∀ i : ℕ, i < seq.n → 1 + i * seq.d ≤ 3) →
  seq.n ≤ 200 := by
  sorry

end max_points_on_ellipse_l3744_374445


namespace menu_choices_l3744_374485

/-- The number of ways to choose one menu for lunch and one for dinner -/
def choose_menus (lunch_chinese : Nat) (lunch_japanese : Nat) (dinner_chinese : Nat) (dinner_japanese : Nat) : Nat :=
  (lunch_chinese + lunch_japanese) * (dinner_chinese + dinner_japanese)

theorem menu_choices : choose_menus 5 4 3 5 = 72 := by
  sorry

end menu_choices_l3744_374485


namespace cube_division_theorem_l3744_374487

/-- Represents the volume of the remaining solid after removal of marked cubes --/
def remaining_volume (k : ℕ) : ℚ :=
  if k % 2 = 0 then 1/2
  else (k+1)^2 * (2*k-1) / (4*k^3)

/-- Represents the surface area of the remaining solid after removal of marked cubes --/
def remaining_surface_area (k : ℕ) : ℚ :=
  if k % 2 = 0 then 3*(k+1) / 2
  else 3*(k+1)^2 / (2*k)

theorem cube_division_theorem (k : ℕ) :
  (k ≥ 65 → remaining_surface_area k > 100) ∧
  (k % 2 = 0 → remaining_volume k ≤ 1/2) :=
sorry

end cube_division_theorem_l3744_374487


namespace gcd_242_154_l3744_374469

theorem gcd_242_154 : Nat.gcd 242 154 = 22 := by
  sorry

end gcd_242_154_l3744_374469


namespace product_value_l3744_374436

theorem product_value : 
  (6 * 27^12 + 2 * 81^9) / 8000000^2 * (80 * 32^3 * 125^4) / (9^19 - 729^6) = 10 := by
  sorry

end product_value_l3744_374436


namespace hyperbola_midpoint_l3744_374454

def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem hyperbola_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ ∧
    ¬∃ (x₁' y₁' x₂' y₂' : ℝ),
      hyperbola x₁' y₁' ∧
      hyperbola x₂' y₂' ∧
      (is_midpoint 1 1 x₁' y₁' x₂' y₂' ∨
       is_midpoint (-1) 2 x₁' y₁' x₂' y₂' ∨
       is_midpoint 1 3 x₁' y₁' x₂' y₂') :=
by sorry

end hyperbola_midpoint_l3744_374454


namespace angle_E_measure_l3744_374494

structure Parallelogram where
  E : Real
  F : Real
  G : Real
  H : Real

def external_angle (p : Parallelogram) : Real := 50

theorem angle_E_measure (p : Parallelogram) :
  external_angle p = 50 → p.E = 130 := by
  sorry

end angle_E_measure_l3744_374494


namespace cyclist_distance_theorem_l3744_374458

/-- A cyclist travels in a straight line for two minutes. -/
def cyclist_travel (v1 v2 : ℝ) : ℝ := v1 * 60 + v2 * 60

/-- The theorem states that a cyclist traveling at 2 m/s for the first minute
    and 4 m/s for the second minute covers a total distance of 360 meters. -/
theorem cyclist_distance_theorem :
  cyclist_travel 2 4 = 360 := by sorry

end cyclist_distance_theorem_l3744_374458


namespace january_oil_bill_l3744_374496

theorem january_oil_bill (january_bill february_bill : ℚ) : 
  (february_bill / january_bill = 3 / 2) →
  ((february_bill + 20) / january_bill = 5 / 3) →
  january_bill = 120 := by
sorry

end january_oil_bill_l3744_374496


namespace wall_building_time_l3744_374489

/-- Given that 8 persons can build a 140 m long wall in 42 days, 
    prove that 30 persons can complete a 100 m long wall in 8 days -/
theorem wall_building_time 
  (persons_initial : ℕ) 
  (length_initial : ℕ) 
  (days_initial : ℕ) 
  (persons_new : ℕ) 
  (length_new : ℕ) 
  (h1 : persons_initial = 8) 
  (h2 : length_initial = 140) 
  (h3 : days_initial = 42) 
  (h4 : persons_new = 30) 
  (h5 : length_new = 100) : 
  (persons_initial * days_initial * length_new) / (persons_new * length_initial) = 8 := by
  sorry

end wall_building_time_l3744_374489


namespace only_valid_pair_is_tiger_and_leopard_l3744_374492

/-- Represents the animals that can participate in the forest competition. -/
inductive Animal : Type
  | Lion : Animal
  | Tiger : Animal
  | Leopard : Animal
  | Elephant : Animal

/-- Represents a pair of animals. -/
structure AnimalPair :=
  (first : Animal)
  (second : Animal)

/-- Checks if the given animal pair satisfies all the competition rules. -/
def satisfiesRules (pair : AnimalPair) : Prop :=
  -- If a lion is sent, a tiger must also be sent
  (pair.first = Animal.Lion ∨ pair.second = Animal.Lion) → 
    (pair.first = Animal.Tiger ∨ pair.second = Animal.Tiger) ∧
  -- If a leopard is not sent, a tiger cannot be sent
  (pair.first ≠ Animal.Leopard ∧ pair.second ≠ Animal.Leopard) → 
    (pair.first ≠ Animal.Tiger ∧ pair.second ≠ Animal.Tiger) ∧
  -- If a leopard participates, the elephant is not willing to go
  (pair.first = Animal.Leopard ∨ pair.second = Animal.Leopard) → 
    (pair.first ≠ Animal.Elephant ∧ pair.second ≠ Animal.Elephant)

/-- The theorem stating that the only valid pair is Tiger and Leopard. -/
theorem only_valid_pair_is_tiger_and_leopard :
  ∀ (pair : AnimalPair), 
    satisfiesRules pair ↔ 
      ((pair.first = Animal.Tiger ∧ pair.second = Animal.Leopard) ∨
       (pair.first = Animal.Leopard ∧ pair.second = Animal.Tiger)) :=
by sorry

end only_valid_pair_is_tiger_and_leopard_l3744_374492


namespace system_no_solution_l3744_374456

theorem system_no_solution (n : ℝ) : 
  (∃ (x y z : ℝ), nx + y = 1 ∧ ny + z = 1 ∧ x + nz = 1) ↔ n ≠ -1 :=
by sorry

end system_no_solution_l3744_374456


namespace annie_purchase_problem_l3744_374402

/-- Annie's hamburger and milkshake purchase problem -/
theorem annie_purchase_problem (hamburger_price milkshake_price hamburger_count milkshake_count remaining_money : ℕ) 
  (h1 : hamburger_price = 4)
  (h2 : milkshake_price = 5)
  (h3 : hamburger_count = 8)
  (h4 : milkshake_count = 6)
  (h5 : remaining_money = 70) :
  hamburger_price * hamburger_count + milkshake_price * milkshake_count + remaining_money = 132 := by
  sorry

end annie_purchase_problem_l3744_374402


namespace last_two_digits_2005_pow_base_3_representation_l3744_374450

-- Define the expression
def big_exp : ℕ := 2003^2004 + 3

-- Define the function to calculate the last two digits in base 3
def last_two_digits_base_3 (n : ℕ) : ℕ := n % 9

-- Theorem statement
theorem last_two_digits_2005_pow : last_two_digits_base_3 (2005^big_exp) = 4 := by
  sorry

-- Convert to base 3
theorem base_3_representation : (last_two_digits_base_3 (2005^big_exp)).digits 3 = [1, 1] := by
  sorry

end last_two_digits_2005_pow_base_3_representation_l3744_374450


namespace similar_cuts_possible_equilateral_cuts_impossible_l3744_374419

-- Define a triangular prism
structure TriangularPrism :=
  (base : Set (ℝ × ℝ × ℝ))
  (height : ℝ)

-- Define a cut on the prism
structure Cut :=
  (shape : Set (ℝ × ℝ × ℝ))
  (is_triangular : Bool)

-- Define similarity between two cuts
def are_similar (c1 c2 : Cut) : Prop := sorry

-- Define equality between two cuts
def are_equal (c1 c2 : Cut) : Prop := sorry

-- Define if a cut touches the base
def touches_base (c : Cut) (p : TriangularPrism) : Prop := sorry

-- Define if two cuts touch each other
def cuts_touch (c1 c2 : Cut) : Prop := sorry

-- Theorem for part (a)
theorem similar_cuts_possible (p : TriangularPrism) :
  ∃ (c1 c2 : Cut),
    c1.is_triangular ∧
    c2.is_triangular ∧
    are_similar c1 c2 ∧
    ¬are_equal c1 c2 ∧
    ¬touches_base c1 p ∧
    ¬touches_base c2 p ∧
    ¬cuts_touch c1 c2 := by sorry

-- Define an equilateral triangular cut
def is_equilateral_triangle (c : Cut) (side_length : ℝ) : Prop := sorry

-- Theorem for part (b)
theorem equilateral_cuts_impossible (p : TriangularPrism) :
  ¬∃ (c1 c2 : Cut),
    is_equilateral_triangle c1 1 ∧
    is_equilateral_triangle c2 2 ∧
    ¬touches_base c1 p ∧
    ¬touches_base c2 p ∧
    ¬cuts_touch c1 c2 := by sorry

end similar_cuts_possible_equilateral_cuts_impossible_l3744_374419


namespace dacids_physics_marks_l3744_374429

theorem dacids_physics_marks :
  let english_marks : ℕ := 73
  let math_marks : ℕ := 69
  let chemistry_marks : ℕ := 64
  let biology_marks : ℕ := 82
  let average_marks : ℕ := 76
  let num_subjects : ℕ := 5

  let total_marks : ℕ := average_marks * num_subjects
  let known_marks : ℕ := english_marks + math_marks + chemistry_marks + biology_marks
  let physics_marks : ℕ := total_marks - known_marks

  physics_marks = 92 :=
by
  sorry

end dacids_physics_marks_l3744_374429


namespace waiter_remaining_customers_l3744_374493

theorem waiter_remaining_customers 
  (initial_customers : Real) 
  (first_group_left : Real) 
  (second_group_left : Real) 
  (h1 : initial_customers = 36.0)
  (h2 : first_group_left = 19.0)
  (h3 : second_group_left = 14.0) : 
  initial_customers - first_group_left - second_group_left = 3.0 := by
sorry

end waiter_remaining_customers_l3744_374493


namespace james_sales_l3744_374430

/-- Given James' sales over two days, prove the total number of items sold --/
theorem james_sales (day1_houses day2_houses : ℕ) (day2_sale_rate : ℚ) : 
  day1_houses = 20 →
  day2_houses = 2 * day1_houses →
  day2_sale_rate = 4/5 →
  (day1_houses + (day2_houses : ℚ) * day2_sale_rate) * 2 = 104 := by
sorry

end james_sales_l3744_374430


namespace sum_mod_seven_l3744_374423

theorem sum_mod_seven : (4123 + 4124 + 4125 + 4126 + 4127) % 7 = 4 := by
  sorry

end sum_mod_seven_l3744_374423


namespace solution_mixture_problem_l3744_374446

theorem solution_mixture_problem (x : ℝ) :
  -- Solution 1 composition
  x + 80 = 100 →
  -- Solution 2 composition
  45 + 55 = 100 →
  -- Mixture composition (50% each solution)
  (x + 45) / 2 + (80 + 55) / 2 = 100 →
  -- Mixture contains 67.5% carbonated water
  (80 + 55) / 2 = 67.5 →
  -- Conclusion: Solution 1 is 20% lemonade
  x = 20 := by
sorry

end solution_mixture_problem_l3744_374446


namespace simplify_and_evaluate_l3744_374415

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  (1 + 1 / (a - 1)) / (a / (a^2 - 1)) = Real.sqrt 2 := by sorry

end simplify_and_evaluate_l3744_374415


namespace special_school_total_students_l3744_374403

/-- Represents a school with blind and deaf students -/
structure School where
  blind_students : ℕ
  deaf_students : ℕ

/-- The total number of students in the school -/
def total_students (s : School) : ℕ :=
  s.blind_students + s.deaf_students

/-- A special school with a specific ratio of deaf to blind students and a given number of blind students -/
def special_school : School :=
  { blind_students := 45,
    deaf_students := 3 * 45 }

theorem special_school_total_students :
  total_students special_school = 180 := by
  sorry

end special_school_total_students_l3744_374403


namespace matt_completes_in_100_days_l3744_374426

/-- The rate at which Matt and Peter complete work together -/
def combined_rate : ℚ := 1 / 20

/-- The rate at which Peter completes work alone -/
def peter_rate : ℚ := 1 / 25

/-- The rate at which Matt completes work alone -/
def matt_rate : ℚ := combined_rate - peter_rate

/-- The number of days Matt takes to complete the work alone -/
def matt_days : ℚ := 1 / matt_rate

theorem matt_completes_in_100_days : matt_days = 100 := by
  sorry

end matt_completes_in_100_days_l3744_374426


namespace student_A_received_A_grade_l3744_374499

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student

-- Define the grade levels
inductive Grade : Type
| A : Grade
| B : Grade
| C : Grade

-- Define a function to represent the actual grades received
def actual_grade : Student → Grade := sorry

-- Define a function to represent the correctness of predictions
def prediction_correct : Student → Prop := sorry

-- Theorem statement
theorem student_A_received_A_grade :
  -- Only one student received an A grade
  (∃! s : Student, actual_grade s = Grade.A) →
  -- A's prediction: C can only get a B or C
  (actual_grade Student.C ≠ Grade.A) →
  -- B's prediction: B will get an A
  (actual_grade Student.B = Grade.A) →
  -- C's prediction: C agrees with A's prediction
  (actual_grade Student.C ≠ Grade.A) →
  -- Only one prediction was inaccurate
  (∃! s : Student, ¬prediction_correct s) →
  -- Student A received an A grade
  actual_grade Student.A = Grade.A :=
sorry

end student_A_received_A_grade_l3744_374499


namespace line_perp_plane_implies_perp_line_l3744_374467

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_implies_perp_line 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perpendicular m α) 
  (h3 : subset n α) : 
  perpendicularLines m n :=
sorry

end line_perp_plane_implies_perp_line_l3744_374467


namespace negative_exponent_two_l3744_374442

theorem negative_exponent_two : 2⁻¹ = (1 : ℝ) / 2 := by sorry

end negative_exponent_two_l3744_374442


namespace unique_prime_six_digit_number_l3744_374434

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (B A : Nat) : Nat := 3000000 + B * 10000 + 1200 + A

theorem unique_prime_six_digit_number :
  ∃! (B A : Nat), B < 10 ∧ A < 10 ∧ 
    is_prime (six_digit_number B A) ∧
    B + A = 9 := by sorry

end unique_prime_six_digit_number_l3744_374434


namespace geometric_sequence_a3_l3744_374461

/-- Given a geometric sequence with common ratio 3, prove that a_3 = 3 if S_3 + S_4 = 53/3 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 3 * a n) →  -- Geometric sequence with common ratio 3
  (∀ n, S n = (a 1 * (3^n - 1)) / 2) →  -- Sum formula for geometric sequence
  S 3 + S 4 = 53 / 3 →  -- Given condition
  a 3 = 3 := by
sorry


end geometric_sequence_a3_l3744_374461


namespace nail_decoration_time_l3744_374407

theorem nail_decoration_time (total_time : ℕ) (num_coats : ℕ) (time_per_coat : ℕ) : 
  total_time = 120 →
  num_coats = 3 →
  total_time = num_coats * 2 * time_per_coat →
  time_per_coat = 20 := by
sorry

end nail_decoration_time_l3744_374407


namespace spending_calculation_l3744_374424

theorem spending_calculation (initial_amount : ℚ) : 
  let remaining_after_clothes : ℚ := initial_amount * (2/3)
  let remaining_after_food : ℚ := remaining_after_clothes * (4/5)
  let final_amount : ℚ := remaining_after_food * (3/4)
  final_amount = 300 → initial_amount = 750 := by
sorry

end spending_calculation_l3744_374424


namespace welders_left_correct_l3744_374447

/-- The number of welders who left for the other project after the first day -/
def welders_who_left : ℕ := 11

/-- The initial number of welders -/
def initial_welders : ℕ := 16

/-- The number of days to complete the order with all welders -/
def initial_days : ℕ := 8

/-- The additional days needed by remaining welders to complete the order -/
def additional_days : ℕ := 16

/-- The total amount of work to be done -/
def total_work : ℝ := initial_welders * initial_days

/-- The work done in the first day -/
def first_day_work : ℝ := initial_welders

/-- The remaining work after the first day -/
def remaining_work : ℝ := total_work - first_day_work

theorem welders_left_correct :
  (initial_welders - welders_who_left) * (initial_days + additional_days) = remaining_work :=
sorry

end welders_left_correct_l3744_374447


namespace purse_percentage_l3744_374484

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of pennies in Samantha's purse -/
def num_pennies : ℕ := 2

/-- The number of nickels in Samantha's purse -/
def num_nickels : ℕ := 3

/-- The number of dimes in Samantha's purse -/
def num_dimes : ℕ := 1

/-- The number of quarters in Samantha's purse -/
def num_quarters : ℕ := 2

/-- The total value of coins in Samantha's purse in cents -/
def total_cents : ℕ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

/-- The percentage of one dollar in Samantha's purse -/
theorem purse_percentage : (total_cents : ℚ) / 100 = 77 / 100 := by
  sorry

end purse_percentage_l3744_374484


namespace N_prime_iff_k_eq_two_l3744_374440

def N (k : ℕ) : ℕ := (10^(2*k) - 1) / 99

theorem N_prime_iff_k_eq_two :
  ∀ k : ℕ, k > 0 → (Nat.Prime (N k) ↔ k = 2) := by sorry

end N_prime_iff_k_eq_two_l3744_374440


namespace approximately_200_men_joined_l3744_374479

-- Define the initial number of men
def initial_men : ℕ := 1000

-- Define the initial duration of provisions in days
def initial_duration : ℚ := 20

-- Define the new duration of provisions in days
def new_duration : ℚ := 167/10  -- 16.67 as a rational number

-- Define a function to calculate the number of men who joined
def men_joined : ℚ := 
  (initial_men * initial_duration / new_duration) - initial_men

-- Theorem statement
theorem approximately_200_men_joined : 
  199 ≤ men_joined ∧ men_joined < 201 := by
  sorry


end approximately_200_men_joined_l3744_374479


namespace no_infinite_sequence_exists_l3744_374486

theorem no_infinite_sequence_exists : 
  ¬ (∃ (x : ℕ → ℝ), (∀ n : ℕ, x n > 0) ∧ 
    (∀ n : ℕ, x (n + 2) = Real.sqrt (x (n + 1)) - Real.sqrt (x n))) := by
  sorry

end no_infinite_sequence_exists_l3744_374486


namespace triangle_inequality_with_powers_l3744_374455

theorem triangle_inequality_with_powers (n : ℕ) (a b c : ℝ) 
  (hn : n > 1) 
  (hab : a > 0) (hbc : b > 0) (hca : c > 0)
  (hsum : a + b + c = 1)
  (htriangle : a < b + c ∧ b < a + c ∧ c < a + b) :
  (a^n + b^n)^(1/n : ℝ) + (b^n + c^n)^(1/n : ℝ) + (c^n + a^n)^(1/n : ℝ) < 1 + 2^(1/n : ℝ)/2 := by
  sorry

end triangle_inequality_with_powers_l3744_374455


namespace pets_problem_l3744_374439

theorem pets_problem (total_students : ℕ) 
  (students_with_dogs : ℕ) 
  (students_with_cats : ℕ) 
  (students_with_other_pets : ℕ) 
  (students_no_pets : ℕ) 
  (only_dogs : ℕ) 
  (only_cats : ℕ) 
  (only_other_pets : ℕ) 
  (dogs_and_cats : ℕ) 
  (dogs_and_other : ℕ) 
  (cats_and_other : ℕ) :
  total_students = 40 →
  students_with_dogs = 20 →
  students_with_cats = total_students / 4 →
  students_with_other_pets = 10 →
  students_no_pets = 5 →
  only_dogs = 15 →
  only_cats = 4 →
  only_other_pets = 5 →
  total_students = only_dogs + only_cats + only_other_pets + 
    dogs_and_cats + dogs_and_other + cats_and_other + 
    students_no_pets + (students_with_dogs + students_with_cats + 
    students_with_other_pets - (only_dogs + only_cats + only_other_pets + 
    dogs_and_cats + dogs_and_other + cats_and_other)) →
  students_with_dogs + students_with_cats + students_with_other_pets - 
    (only_dogs + only_cats + only_other_pets + 
    dogs_and_cats + dogs_and_other + cats_and_other) = 0 :=
by sorry

end pets_problem_l3744_374439
